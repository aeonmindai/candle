use crate::backend::{BackendDevice, BackendStorage};
use crate::{CpuStorage, CpuStorageRef, DType, Layout, Result, Shape};
pub use candle_kernels as kernels;
pub use cudarc;
use cudarc::driver::CudaFunction;
// NOTE: `HostSlice` is deliberately NOT imported. Bringing it into scope makes
// its `len` method compete with the inherent `[T]::len` at every call site in
// this file, including `kernels::ALL_IDS.len()` in const position -- which stops
// compiling, because a trait method is not const. `stream_synced_slice` is
// called through its fully-qualified path instead.
use float8::F8E4M3;
use half::{bf16, f16};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

use super::{CudaError, CudaStorage, CudaStorageSlice, WrapErr};

/// Capture-time host->device copies that had their source retained, since
/// process start. See [`arc_capture_retain_host`].
static ARC_CAPTURE_HTOD_RETAINED: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
/// Bytes retained by [`arc_capture_retain_host`].
static ARC_CAPTURE_HTOD_BYTES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// `(count, bytes)` of capture-time H2D sources retained so far.
///
/// This is the honest answer to "did the capture-safety fix do anything this
/// run": a run where the fix never fired reports `(0, 0)`, which is a different
/// statement from "capture succeeded". Assert on it; do not infer it.
pub fn arc_capture_htod_retained() -> (u64, u64) {
    use std::sync::atomic::Ordering;
    (
        ARC_CAPTURE_HTOD_RETAINED.load(Ordering::Relaxed),
        ARC_CAPTURE_HTOD_BYTES.load(Ordering::Relaxed),
    )
}

/// Retain the HOST source of a host->device copy **forever**, returning a
/// `'static` alias of the same bytes.
///
/// # Why (RUN-161 / ArcGraph)
///
/// `CudaStream::memcpy_htod` is `cuMemcpyHtoDAsync(dst, host_ptr, n, stream)`.
/// Under `cuStreamBeginCapture` that call is **recorded, not executed**: the
/// resulting graph gains a MEMCPY node that stores the *host pointer* and
/// dereferences it on the first `cuGraphLaunch` and on every replay afterwards.
///
/// Everything that builds a tensor from host data — `Tensor::new`,
/// `from_vec`, `from_slice`, `arange`, `full`, and every `CpuStorage` that
/// reaches [`CudaDevice::clone_htod`] — materialises its payload in a transient
/// `Vec` that is dropped as soon as the expression returns. By the time the
/// graph launches, that host allocation has been returned to the allocator and
/// very often unmapped, so the driver's launch-time validation of the node's
/// source region fails **synchronously**: `cuGraphLaunch` returns 700
/// (`CUDA_ERROR_ILLEGAL_ADDRESS`) on an otherwise clean context, before any
/// kernel runs. That synchronous-700-on-a-clean-context signature is the
/// fingerprint of this bug and is what distinguishes it from a device-side
/// fault, which would surface asynchronously at the next sync.
///
/// [`CudaDevice::htod_info`] already applied exactly this fix, but only for
/// kernel dims/strides. Nothing protected `clone_htod`/`memcpy_htod`, which is
/// the path every host-built tensor takes. This closes that gap for all of
/// them at once, so a site the manual sweep missed is covered too.
///
/// Bounded by construction: it only runs while `capture_mode()` is set, which is
/// a handful of forwards per process, and the payloads are index/position
/// vectors of at most a few KB.
pub fn arc_capture_retain_host<T>(src: &[T]) -> &'static [T] {
    use std::sync::atomic::Ordering;
    if src.is_empty() {
        return &[];
    }
    let bytes = std::mem::size_of_val(src);
    ARC_CAPTURE_HTOD_RETAINED.fetch_add(1, Ordering::Relaxed);
    ARC_CAPTURE_HTOD_BYTES.fetch_add(bytes as u64, Ordering::Relaxed);
    // A capture-time H2D of this size is not an index vector -- it is bulk data
    // that has no business inside the captured region. Retain it anyway (a
    // correct-but-fat graph beats a faulting one) but say so, loudly.
    if bytes > 1 << 20 {
        eprintln!(
            "[arc-htod] WARNING: retaining {bytes} B of host memory for a capture-time \
             H2D copy -- a payload this large inside the captured region is a bug"
        );
    }
    if std::env::var_os("ARC_HTOD_TRACE").is_some() {
        eprintln!(
            "[arc-htod] capture-time H2D retained: {bytes} B ({} x {})\n{}",
            src.len(),
            std::any::type_name::<T>(),
            std::backtrace::Backtrace::force_capture()
        );
    }
    // SAFETY: callers bound `T: DeviceRepr`, i.e. a plain-old-data type that is
    // valid to memcpy to the device. Copying its bytes into a fresh allocation
    // with the identical layout therefore yields a valid `[T]`. The allocation
    // is never freed, which is the point: the graph node reads it on every
    // replay for the lifetime of the process.
    unsafe {
        let layout = std::alloc::Layout::from_size_align(bytes, std::mem::align_of::<T>())
            .expect("arc_capture_retain_host: invalid layout");
        let p = std::alloc::alloc(layout) as *mut T;
        if p.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        std::ptr::copy_nonoverlapping(src.as_ptr(), p, src.len());
        std::slice::from_raw_parts(p, src.len())
    }
}

/// Capture-time device->host copies observed. See [`arc_capture_clone_dtoh`].
static ARC_CAPTURE_DTOH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Number of device->host copies issued while a graph capture was open.
///
/// Should be ZERO. Any non-zero value names a real defect: the captured forward
/// is reading a device value back to the host, which a graph cannot do.
pub fn arc_capture_dtoh_count() -> u64 {
    ARC_CAPTURE_DTOH.load(std::sync::atomic::Ordering::Relaxed)
}

/// `clone_dtoh` that cannot corrupt the host heap when issued during capture.
///
/// # Why (RUN-161 / ArcGraph)
///
/// `CudaStream::clone_dtoh` allocates a `Vec`, issues `cuMemcpyDtoHAsync` into
/// it, and returns **without synchronising**. Under `cuStreamBeginCapture` that
/// copy is recorded as a graph MEMCPY node whose *destination* is that `Vec`'s
/// heap buffer. The `Vec` is dropped as soon as the caller is done with it, so
/// every `cuGraphLaunch` afterwards has the driver **writing into freed heap
/// memory** — which is why the failure is `malloc_consolidate(): invalid chunk
/// size` on a later allocation, and `CUDA_ERROR_ILLEGAL_ADDRESS` when the page
/// has already been returned to the OS.
///
/// This is the mirror image of the host-source problem that
/// [`arc_capture_retain_host`] fixes, and it is the more destructive half: a
/// dangling *source* is only read, a dangling *destination* is written.
///
/// Note it does not fail capture the way a host round trip normally would,
/// because it never synchronises — so it is invisible to any "blocking D2H per
/// step" count. It has to be counted directly.
///
/// During capture the copy is redirected into a leaked buffer and the caller
/// gets uninitialised storage. That loses nothing: capture *records*, it does
/// not execute, so the caller's bytes were never meaningful on this pass. The
/// graph's replay output is separately gated against an eager forward before it
/// is trusted, so a forward that really does depend on a host readback is
/// caught there rather than silently returning garbage.
pub fn arc_capture_clone_dtoh<T: cudarc::driver::DeviceRepr + 'static>(
    dev: &CudaDevice,
    slice: &cudarc::driver::CudaSlice<T>,
) -> Result<Vec<T>> {
    let stream = slice.stream();
    if !dev.capture_mode() {
        return stream.clone_dtoh(slice).w();
    }
    let n = slice.len();
    ARC_CAPTURE_DTOH.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    eprintln!(
        "[arc-dtoh] WARNING: device->host copy of {n} x {} issued while a graph capture is \
         open. The captured graph would write into a freed host Vec on every replay; \
         redirecting to a leaked buffer. Set ARC_HTOD_TRACE=1 for the call site.",
        std::any::type_name::<T>()
    );
    if std::env::var_os("ARC_HTOD_TRACE").is_some() {
        eprintln!("{}", std::backtrace::Backtrace::force_capture());
    }
    // SAFETY: `T: DeviceRepr` is POD, and this mirrors what `clone_dtoh` itself
    // does (`with_capacity` + `set_len`) before overwriting via memcpy.
    let mut shadow: Vec<T> = Vec::with_capacity(n);
    unsafe { shadow.set_len(n) };
    let shadow: &'static mut [T] = Box::leak(shadow.into_boxed_slice());
    stream.memcpy_dtoh(slice, shadow).w()?;
    let mut out: Vec<T> = Vec::with_capacity(n);
    unsafe { out.set_len(n) };
    Ok(out)
}

/// Unique identifier for cuda devices.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceId(usize);

impl DeviceId {
    fn new() -> Self {
        // https://users.rust-lang.org/t/idiomatic-rust-way-to-generate-unique-id/33805
        use std::sync::atomic;
        static COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(1);
        Self(COUNTER.fetch_add(1, atomic::Ordering::Relaxed))
    }
}

struct CudaRng(cudarc::curand::CudaRng);
unsafe impl Send for CudaRng {}

pub struct ModuleStore {
    mdls: [Option<Arc<cudarc::driver::CudaModule>>; kernels::ALL_IDS.len()],
}

/// Opt-in CUDA caching allocator for graph-capture safety (RUN-161).
///
/// When `enabled`, freed device buffers are returned to `free` (keyed by byte
/// size) instead of `cuMemFreeAsync`, and `alloc` reuses them via
/// `upgrade_device_ptr` instead of `cuMemAllocAsync`. Warming the cache (a few
/// eager forwards at the capture shapes) then makes a captured forward
/// allocation-free: stable addresses (no MMU fault), no alloc/free graph nodes,
/// no cross-stream capture isolation. Default OFF -> candle's alloc/free is
/// byte-identical to upstream, so existing models are unaffected.
#[derive(Default)]
pub struct AllocCache {
    enabled: bool,
    /// While `capturing`, buffers freed (Drop -> cache_put) are NOT returned to
    /// `free` (which would let them be re-served within the SAME capture,
    /// recording two graph uses of one address -> aliasing corruption / MMU
    /// fault). They are parked in `deferred` and moved back to `free` only when
    /// capture mode ends. This guarantees every allocation during a captured
    /// forward gets a unique, stable address. (RUN-161, PyTorch-style.)
    capturing: bool,
    free: HashMap<usize, Vec<cudarc::driver::sys::CUdeviceptr>>,
    /// Buffers freed during capture, parked until capture mode ends.
    deferred: Vec<(usize, cudarc::driver::sys::CUdeviceptr)>,
    /// Debug: byte-sizes that have missed the cache (logged once each under
    /// ARC_CACHE_DEBUG). A new miss DURING capture = an allocation that becomes
    /// an unstable graph memory node -> the cause of the launch fault/corruption.
    ///
    /// This set is a LOG DEDUP, not a diagnostic. It silences the second miss of
    /// a size, so a size that first missed during a harmless deferred-free warm
    /// pass stays silent when it misses again during the real capture. Never
    /// gate a correctness check on it -- use `capture_misses`.
    missed: std::collections::HashSet<usize>,
    /// Per-size allocation demand of ONE capture-mode forward.
    ///
    /// `window` counts allocations of the capture-mode window currently open;
    /// closing the window folds it into `demand` with a per-size max. Because
    /// frees are deferred while capturing, the demand of a size is the window's
    /// TOTAL allocation count, not its peak-live count -- serving a captured
    /// forward needs one distinct buffer per allocation.
    ///
    /// The max over windows matters: buffers whose width cycles with a period
    /// (the V4 rolling-compressor tail cycles through `ratio` consecutive
    /// widths) present a DIFFERENT size on each step, so a single warm pass sees
    /// only one of them. Running `ratio` warm windows and taking the union of
    /// sizes with the max of counts covers the whole cycle.
    window: HashMap<usize, usize>,
    demand: HashMap<usize, usize>,
    /// Misses observed while `capturing`, since the last `reset_capture_misses`.
    /// Counted, never deduped: a miss during the real capture is an allocation
    /// served from the graph's private pool, i.e. an unstable graph memory node,
    /// i.e. CUDA_ERROR_ILLEGAL_ADDRESS on `cuGraphLaunch`. One is fatal.
    capture_misses: HashMap<usize, usize>,
}

#[derive(Clone)]
pub struct CudaDevice {
    id: DeviceId,
    context: Arc<cudarc::driver::CudaContext>,
    modules: Arc<std::sync::RwLock<ModuleStore>>,
    custom_modules: Arc<std::sync::RwLock<HashMap<String, Arc<cudarc::driver::CudaModule>>>>,
    stream: Arc<cudarc::driver::CudaStream>,
    pub(crate) blas: Arc<cudarc::cublas::CudaBlas>,
    curand: Arc<Mutex<CudaRng>>,
    seed_value: Arc<RwLock<u64>>,
    pub(crate) alloc_cache: Arc<Mutex<AllocCache>>,
}

impl CudaDevice {
    /// Enable/disable the opt-in caching allocator at runtime. Disabling drains
    /// and frees every cached buffer back to the driver. Keep OFF during model
    /// load (avoids hoarding transient load buffers); turn ON only around the
    /// decode warmup + CUDA-graph capture window.
    pub fn set_alloc_cache_enabled(&self, enabled: bool) {
        let drained: Vec<cudarc::driver::sys::CUdeviceptr> = {
            let mut cache = self.alloc_cache.lock().unwrap();
            cache.enabled = enabled;
            if enabled {
                Vec::new()
            } else {
                cache.capturing = false;
                // The open window described a pool that is about to stop
                // existing; folding it into the profile would record a demand
                // no buffer here can serve. `demand` itself survives: it is a
                // fact about the model's shapes, not about this pool.
                cache.window.clear();
                let mut d: Vec<_> = cache.free.drain().flat_map(|(_, v)| v).collect();
                d.extend(cache.deferred.drain(..).map(|(_, p)| p));
                d
            }
        };
        // Free drained buffers outside the lock (upgrade -> Drop frees via cudarc).
        for ptr in drained {
            drop(unsafe { self.stream.upgrade_device_ptr::<u8>(ptr, 0) });
        }
    }

    pub fn alloc_cache_enabled(&self) -> bool {
        self.alloc_cache.lock().unwrap().enabled
    }

    /// Enter/leave capture mode. While in capture mode, freed buffers are parked
    /// (not reusable) to avoid within-capture buffer aliasing. Leaving capture
    /// mode returns all parked buffers to the free pool.
    ///
    /// Protocol for an allocation-free capture:
    /// 1. `set_alloc_cache_enabled(true)`
    /// 2. eager warmup forwards (warm cuBLAS algos + kernels)
    /// 3. `set_capture_mode(true)`; one eager forward; `set_capture_mode(false)`
    ///    -- this "deferred-free" pass grows `free` to the full per-forward
    ///    allocation count (no reuse, so every alloc is distinct).
    /// 4. `set_capture_mode(true)`; begin_capture; forward; end_capture;
    ///    `set_capture_mode(false)` -- every alloc is now a cache hit at a stable
    ///    address, every free is deferred -> no graph memory nodes, no aliasing.
    /// Free EVERY cached buffer immediately, while the pool they were allocated
    /// from is still alive.
    ///
    /// # Why this exists (RUN-161 heap corruption)
    ///
    /// `AllocCache` stores `(bytes, ptr)` and **nothing about which memory pool
    /// a pointer came from**. It is device-global and lives for the process.
    /// Arc's CUDA-graph capture installs a *private* pool as the device default
    /// for the duration of a capture (`cuDeviceSetMemPool`), so every allocation
    /// that misses the cache during capture is served **from that private pool**
    /// — and is then parked in this cache when it is freed.
    ///
    /// When the private pool is destroyed, those cached pointers dangle. The
    /// cache keeps handing them out; `cuMemFreeAsync` is eventually called on a
    /// pointer whose pool no longer exists. `cuMemPoolDestroy` with outstanding
    /// allocations is undefined behaviour, and freeing into a destroyed pool
    /// corrupts the driver's host-side bookkeeping — which lives in the
    /// process's glibc arena. Measured symptom: `corrupted double-linked list`
    /// and `malloc_consolidate(): invalid chunk size`, at an arbitrary later
    /// allocation, only ever in capture-enabled runs.
    ///
    /// # Why it drains `free` as well as `deferred`
    ///
    /// Draining only `deferred` is not enough. A buffer still LIVE at capture
    /// end — the captured graph's own output tensor is the obvious one — is
    /// dropped later, when `capturing` is already false, so it lands in `free`.
    /// Both lists can therefore hold private-pool pointers, and without pool
    /// provenance the only sound answer is to drain both.
    ///
    /// # Caller contract
    ///
    /// Call this **before** `cuMemPoolDestroy`, and only when no captured graph
    /// may still replay — a live graph's baked addresses are exactly these
    /// buffers, so freeing them under a replayable graph trades one
    /// use-after-free for another. Destroy the `CUgraphExec` first, then drain,
    /// then destroy the pool.
    ///
    /// Cost is one cold warmup: the cache simply refills.
    pub fn drain_alloc_cache_and_free(&self) {
        let drained: Vec<cudarc::driver::sys::CUdeviceptr> = {
            let mut cache = self.alloc_cache.lock().unwrap();
            cache.capturing = false;
            cache.window.clear();
            let mut d: Vec<_> = cache.free.drain().flat_map(|(_, v)| v).collect();
            d.extend(cache.deferred.drain(..).map(|(_, p)| p));
            // Sizes that missed before mean nothing once every buffer is gone.
            cache.missed.clear();
            d
        };
        // Free outside the lock (upgrade -> Drop frees via cudarc).
        for ptr in drained {
            drop(unsafe { self.stream.upgrade_device_ptr::<u8>(ptr, 0) });
        }
    }

    pub fn set_capture_mode(&self, capturing: bool) {
        // RUN-161: ARC_NO_DEFERRED_FREE makes this a no-op -> buffers freed during
        // capture are recycled normally (within-capture reuse). Tests whether
        // single-stream recycling is capture-safe (it should be: a buffer is only
        // freed after its last use is submitted = stream-ordered). If so we drop
        // deferred-free entirely -> peak memory = normal forward peak (fits 80GB).
        if std::env::var_os("ARC_NO_DEFERRED_FREE").is_some() {
            return;
        }
        let mut cache = self.alloc_cache.lock().unwrap();
        cache.capturing = capturing;
        if capturing {
            cache.window.clear();
        } else {
            let window = std::mem::take(&mut cache.window);
            for (bytes, n) in window {
                let slot = cache.demand.entry(bytes).or_insert(0);
                *slot = (*slot).max(n);
            }
            let deferred = std::mem::take(&mut cache.deferred);
            for (bytes, ptr) in deferred {
                cache.free.entry(bytes).or_default().push(ptr);
            }
        }
    }

    /// Per-size allocation demand of one capture-mode forward, `(bytes, count)`,
    /// as observed over every capture-mode window so far. Sorted by size.
    pub fn capture_alloc_demand(&self) -> Vec<(usize, usize)> {
        let cache = self.alloc_cache.lock().unwrap();
        let mut v: Vec<(usize, usize)> = cache.demand.iter().map(|(&b, &n)| (b, n)).collect();
        v.sort_unstable();
        v
    }

    /// Clear the capture-miss ledger. Call immediately before entering the real
    /// capture so the count that follows describes that capture and nothing else.
    pub fn reset_capture_misses(&self) {
        self.alloc_cache.lock().unwrap().capture_misses.clear();
    }

    /// `(bytes, count)` for every size that missed the cache while capturing,
    /// since the last `reset_capture_misses`. Sorted by size.
    pub fn capture_misses(&self) -> Vec<(usize, usize)> {
        let cache = self.alloc_cache.lock().unwrap();
        let mut v: Vec<(usize, usize)> =
            cache.capture_misses.iter().map(|(&b, &n)| (b, n)).collect();
        v.sort_unstable();
        v
    }

    /// Total capture-mode misses since the last `reset_capture_misses`.
    /// Non-zero after a capture forward means the graph has at least one
    /// unstable memory node and MUST NOT be instantiated.
    pub fn capture_miss_count(&self) -> usize {
        self.alloc_cache
            .lock()
            .unwrap()
            .capture_misses
            .values()
            .sum()
    }

    /// Grow the free pool so that every size in the observed demand profile can
    /// be served `demand + slack` times without touching the driver.
    ///
    /// This is the pre-warm the capture protocol's step 3 approximates. Step 3
    /// leaves `free[S]` at the demand of the ONE step it ran; any size whose
    /// demand is higher on the step that gets captured -- or that only appears
    /// on that step -- misses, and a capture-time miss is served from the
    /// graph's private pool as an unstable memory node. Topping up from the
    /// profile closes both gaps, and `slack` absorbs a step that allocates a
    /// little more than any observed one.
    ///
    /// Returns `(sizes_topped_up, buffers_allocated)`. Call with capture mode
    /// OFF: a buffer released while capturing parks in `deferred` and cannot be
    /// served, so pre-warming inside a capture window warms nothing.
    pub fn prewarm_alloc_cache(&self, slack: usize) -> Result<(usize, usize)> {
        // Decide under the lock, allocate outside it: cuMemAllocAsync is slow
        // enough that holding the allocator mutex across it would serialise
        // every other thread's allocations behind the pre-warm.
        let todo: Vec<(usize, usize)> = {
            let cache = self.alloc_cache.lock().unwrap();
            if !cache.enabled {
                return Ok((0, 0));
            }
            cache
                .demand
                .iter()
                .filter_map(|(&bytes, &need)| {
                    let have = cache.free.get(&bytes).map_or(0, |v| v.len());
                    let want = need + slack;
                    (bytes > 0 && want > have).then_some((bytes, want - have))
                })
                .collect()
        };
        let mut buffers = 0usize;
        let mut sizes = 0usize;
        for (bytes, n) in todo {
            let mut made = Vec::with_capacity(n);
            for _ in 0..n {
                // Straight to the driver: `self.alloc` would consult the cache
                // and hand back a buffer that is already in it.
                let slice = unsafe { self.stream.alloc::<u8>(bytes) }.w()?;
                made.push(slice.leak());
            }
            let mut cache = self.alloc_cache.lock().unwrap();
            buffers += made.len();
            sizes += 1;
            cache.free.entry(bytes).or_default().extend(made);
        }
        Ok((sizes, buffers))
    }

    /// `(bytes, buffers_available)` for the free pool. Diagnostic only.
    pub fn alloc_cache_free_counts(&self) -> Vec<(usize, usize)> {
        let cache = self.alloc_cache.lock().unwrap();
        let mut v: Vec<(usize, usize)> = cache.free.iter().map(|(&b, v)| (b, v.len())).collect();
        v.sort_unstable();
        v
    }

    pub fn capture_mode(&self) -> bool {
        self.alloc_cache.lock().unwrap().capturing
    }

    fn cache_take(&self, bytes: usize) -> Option<cudarc::driver::sys::CUdeviceptr> {
        if bytes == 0 {
            return None;
        }
        let mut cache = self.alloc_cache.lock().unwrap();
        if !cache.enabled {
            return None;
        }
        if cache.capturing {
            // Demand accounting for the window currently open. Counted before
            // the lookup so a hit and a miss weigh the same: what the pre-warm
            // must supply is the number of allocations, not the number of
            // failures to serve them.
            *cache.window.entry(bytes).or_insert(0) += 1;
        }
        let hit = cache.free.get_mut(&bytes).and_then(|v| v.pop());
        if hit.is_none() && cache.capturing {
            *cache.capture_misses.entry(bytes).or_insert(0) += 1;
        }
        if hit.is_none() && cache.missed.insert(bytes) {
            if cache.capturing {
                // A miss DURING capture means this size was not pre-warmed: the
                // alloc becomes an unstable graph memory node -> launch fault or
                // corruption. Loud always (not gated) -- it is a correctness bug.
                eprintln!(
                    "[alloc-cache] WARNING: MISS during capture, size {bytes} bytes \
                     (not pre-warmed) -> graph will be unstable. Grow warmup coverage."
                );
            } else if std::env::var_os("ARC_CACHE_DEBUG").is_some() {
                eprintln!("[alloc-cache] MISS new size {bytes} bytes -> cuMemAllocAsync");
            }
        }
        hit
    }

    /// Return a freed buffer to the cache. Returns false if caching is off (the
    /// caller must then free it normally). During capture the buffer is parked
    /// in `deferred` (not reusable until capture mode ends) to prevent
    /// within-capture aliasing.
    pub(crate) fn cache_put(&self, bytes: usize, ptr: cudarc::driver::sys::CUdeviceptr) -> bool {
        if bytes == 0 {
            return false;
        }
        let mut cache = self.alloc_cache.lock().unwrap();
        if !cache.enabled {
            return false;
        }
        if cache.capturing {
            cache.deferred.push((bytes, ptr));
        } else {
            cache.free.entry(bytes).or_default().push(ptr);
        }
        true
    }

    pub(crate) fn cuda_stream_ref(&self) -> &Arc<cudarc::driver::CudaStream> {
        &self.stream
    }
}

impl std::fmt::Debug for CudaDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CudaDevice({:?})", self.id)
    }
}

impl CudaDevice {
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn alloc<T: cudarc::driver::DeviceRepr>(
        &self,
        len: usize,
    ) -> Result<cudarc::driver::CudaSlice<T>> {
        let bytes = len * std::mem::size_of::<T>();
        if let Some(ptr) = self.cache_take(bytes) {
            // Reuse a cached buffer (no cuMemAllocAsync -> graph-capture safe).
            return Ok(unsafe { self.stream.upgrade_device_ptr::<T>(ptr, len) });
        }
        self.stream.alloc::<T>(len).w()
    }

    pub fn alloc_zeros<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits>(
        &self,
        len: usize,
    ) -> Result<cudarc::driver::CudaSlice<T>> {
        let bytes = len * std::mem::size_of::<T>();
        if let Some(ptr) = self.cache_take(bytes) {
            let mut slice = unsafe { self.stream.upgrade_device_ptr::<T>(ptr, len) };
            self.stream.memset_zeros(&mut slice).w()?;
            return Ok(slice);
        }
        self.stream.alloc_zeros::<T>(len).w()
    }

    pub fn memcpy_htod<
        // `'static`: capture-time sources are retained for the life of the graph,
        // which outlives any borrow. Every `DeviceRepr` is a concrete POD, so
        // this bound excludes nothing that could reach here.
        T: cudarc::driver::DeviceRepr + 'static,
        Src: cudarc::driver::HostSlice<T> + ?Sized,
        Dst: cudarc::driver::DevicePtrMut<T>,
    >(
        &self,
        src: &Src,
        dst: &mut Dst,
    ) -> Result<()> {
        // While capturing, the copy is recorded against the HOST pointer and
        // re-read on every replay -- see `arc_capture_retain_host`.
        if self.capture_mode() {
            let (s, _guard) =
                unsafe { cudarc::driver::HostSlice::stream_synced_slice(src, &self.stream) };
            let retained = arc_capture_retain_host(s);
            return self.stream.memcpy_htod(retained, dst).w();
        }
        self.stream.memcpy_htod(src, dst).w()
    }

    pub fn clone_dtoh<T: cudarc::driver::DeviceRepr, Src: cudarc::driver::DevicePtr<T>>(
        &self,
        src: &Src,
    ) -> Result<Vec<T>> {
        self.stream.clone_dtoh(src).w()
    }

    pub fn memcpy_dtod<
        T,
        Src: cudarc::driver::DevicePtr<T>,
        Dst: cudarc::driver::DevicePtrMut<T>,
    >(
        &self,
        src: &Src,
        dst: &mut Dst,
    ) -> Result<()> {
        self.stream.memcpy_dtod(src, dst).w()
    }

    pub fn memcpy_dtoh<
        T: cudarc::driver::DeviceRepr,
        Src: cudarc::driver::DevicePtr<T>,
        Dst: cudarc::driver::HostSlice<T>,
    >(
        &self,
        src: &Src,
        dst: &mut Dst,
    ) -> Result<()> {
        self.stream.memcpy_dtoh(src, dst).w()
    }

    pub fn clone_htod<
        T: cudarc::driver::DeviceRepr + 'static,
        Src: cudarc::driver::HostSlice<T> + ?Sized,
    >(
        &self,
        src: &Src,
    ) -> Result<cudarc::driver::CudaSlice<T>> {
        // RUN-161: route the allocation through the caching allocator. candle's
        // reductions/indexing upload per-op layout via
        // `dev.clone_htod([dims, strides])`; the raw `stream.clone_htod` does a
        // cuMemAllocAsync + free-on-drop, which during CUDA-graph capture leaves
        // an UNSTABLE info buffer -> the kernel reads out-of-bounds dims/strides
        // -> MMU fault on launch. Allocating from the cache gives a stable,
        // deferred-free address; the small htod copy itself is graph-safe (the
        // driver stages tiny host->device copies into the graph). When the cache
        // is OFF this is identical to alloc + memcpy (no behavior change).
        //
        // Allocating from the cache fixes the DEVICE side. The HOST side needs
        // the same treatment: a captured `cuMemcpyHtoDAsync` records the source
        // POINTER, and every host-built tensor (`Tensor::{new,from_vec,
        // from_slice,arange,full}` -> `storage_from_cpu_storage` -> here) hands
        // it a `Vec` that dies with the expression. That is a synchronous 700
        // out of the first `cuGraphLaunch`. See `arc_capture_retain_host`.
        let len = cudarc::driver::HostSlice::len(src);
        let mut dst = unsafe { self.alloc::<T>(len)? };
        if self.capture_mode() {
            let (s, _guard) =
                unsafe { cudarc::driver::HostSlice::stream_synced_slice(src, &self.stream) };
            let retained = arc_capture_retain_host(s);
            self.stream.memcpy_htod(retained, &mut dst).w()?;
            return Ok(dst);
        }
        self.stream.memcpy_htod(src, &mut dst).w()?;
        Ok(dst)
    }

    /// Upload per-op layout/info (dims/strides) for a kernel, returning an
    /// `InfoBuf` whose Drop returns the device buffer to the caching allocator
    /// (stable address under CUDA-graph capture). Used for metadata that is NOT
    /// wrapped in a `CudaStorage` (reductions, indexing, etc.).
    ///
    /// During capture the HOST source is leaked: candle builds the metadata as a
    /// transient `[dims, strides].concat()` Vec that is freed when the op
    /// returns, but a captured `cuMemcpyHtoDAsync` records the host POINTER and
    /// re-reads it on every (re)launch -- a freed Vec yields garbage dims -> the
    /// reduce kernel indexes out of bounds -> MMU fault. Leaking a copy keeps the
    /// source valid for the first launch and all replays. Bounded + tiny (a few
    /// dozen bytes per distinct reduce/index site, only while capturing).
    pub fn htod_info<T: cudarc::driver::DeviceRepr + Copy + 'static>(
        &self,
        src: &[T],
    ) -> Result<super::InfoBuf<T>> {
        let mut dst = unsafe { self.alloc::<T>(src.len())? };
        if self.capture_mode() {
            // Same retention as every other capture-time H2D, through the one
            // helper, so `arc_capture_htod_retained` counts all of them.
            let leaked = arc_capture_retain_host(src);
            self.stream.memcpy_htod(leaked, &mut dst).w()?;
        } else {
            self.stream.memcpy_htod(src, &mut dst).w()?;
        }
        Ok(super::InfoBuf::new(self.clone(), dst))
    }
}

pub struct CudaFunc {
    func: CudaFunction,
    stream: Arc<cudarc::driver::CudaStream>,
}

impl std::ops::Deref for CudaFunc {
    type Target = CudaFunction;

    fn deref(&self) -> &Self::Target {
        &self.func
    }
}

impl CudaFunc {
    pub fn into_cuda_function(self) -> CudaFunction {
        self.func
    }
}

#[macro_export]
macro_rules! builder_arg {
    ($b:ident, $($arg:expr),*) => {
        $(
            let __arg = $arg;
            $b.arg(&__arg);
        )*
    };
}

impl CudaFunc {
    pub fn builder(&self) -> cudarc::driver::LaunchArgs<'_> {
        self.stream.launch_builder(&self.func)
    }
}

impl CudaDevice {
    pub fn cuda_stream(&self) -> Arc<cudarc::driver::CudaStream> {
        self.stream.clone()
    }

    /// When turned on, all cuda tensors **created after calling this function** will
    /// not track uses via cuda events.
    ///
    /// # Safety
    ///
    /// It is up to the user to ensure proper synchronization between multiple streams:
    /// - Ensure that no tensor is freed before a use on another stream is finished.
    /// - Ensure that a tensor is not used on another stream before allocation on the
    ///   allocating stream finishes.
    /// - Ensure that a tensor is not written two concurrently by multiple streams.
    pub unsafe fn disable_event_tracking(&self) {
        self.context.disable_event_tracking()
    }

    pub fn is_event_tracking(&self) -> bool {
        self.context.is_event_tracking()
    }

    #[cfg(all(feature = "ug", not(target_arch = "wasm32")))]
    pub fn compile(
        &self,
        func_name: &'static str,
        kernel: candle_ug::lang::ssa::Kernel,
    ) -> Result<CudaFunc> {
        let mut buf = vec![];
        candle_ug::cuda::code_gen::gen(&mut buf, func_name, &kernel)?;
        let cuda_code = String::from_utf8(buf)?;
        let opts = cudarc::nvrtc::CompileOptions {
            use_fast_math: Some(true),
            ..Default::default()
        };
        let ptx = cudarc::nvrtc::safe::compile_ptx_with_opts(cuda_code, opts).w()?;
        let module = self.context.load_module(ptx).w()?;
        let func = module.load_function(func_name).w()?;
        Ok(CudaFunc {
            func,
            stream: self.stream.clone(),
        })
    }

    pub fn id(&self) -> DeviceId {
        self.id
    }

    pub fn get_or_load_custom_func(
        &self,
        fn_name: &str,
        module_name: &str,
        ptx: &str,
    ) -> Result<CudaFunc> {
        let ms = self.custom_modules.read().unwrap();
        if let Some(mdl) = ms.get(module_name).as_ref() {
            let func = mdl.load_function(fn_name).w()?;
            return Ok(CudaFunc {
                func,
                stream: self.stream.clone(),
            });
        }
        drop(ms);
        let mut ms = self.custom_modules.write().unwrap();
        let cuda_module = self.context.load_module(ptx.into()).w()?;
        ms.insert(module_name.to_string(), cuda_module.clone());
        let func = cuda_module.load_function(fn_name).w()?;
        Ok(CudaFunc {
            func,
            stream: self.stream.clone(),
        })
    }

    pub fn get_or_load_func(&self, fn_name: &str, mdl: &kernels::Module) -> Result<CudaFunc> {
        let ms = self.modules.read().unwrap();
        if let Some(mdl) = ms.mdls[mdl.index()].as_ref() {
            let func = mdl.load_function(fn_name).w()?;
            return Ok(CudaFunc {
                func,
                stream: self.stream.clone(),
            });
        }
        drop(ms);
        let mut ms = self.modules.write().unwrap();
        let cuda_module = self.context.load_module(mdl.ptx().into()).w()?;
        ms.mdls[mdl.index()] = Some(cuda_module.clone());
        let func = cuda_module.load_function(fn_name).w()?;
        Ok(CudaFunc {
            func,
            stream: self.stream.clone(),
        })
    }

    pub fn cublas_handle(&self) -> Arc<cudarc::cublas::CudaBlas> {
        self.blas.clone()
    }
}

impl CudaDevice {
    pub fn new_with_stream(ordinal: usize) -> Result<Self> {
        let context = cudarc::driver::CudaContext::new(ordinal).w()?;
        // RUN-161: disable cudarc's per-slice event tracking. We run everything
        // on this one stream, so events are unnecessary (the stream serializes
        // ops). Crucially they break CUDA-graph capture: a cached buffer's
        // Drop -> leak() does cudaEventDestroy + stream.wait() mid-capture,
        // corrupting the graph (MMU fault on launch); and event-based deps
        // trigger CUDA_ERROR_STREAM_CAPTURE_ISOLATION. Disabling makes
        // alloc/free clean for capture.
        unsafe { context.disable_event_tracking() };
        let stream = context.new_stream().w()?;
        let blas = cudarc::cublas::CudaBlas::new(stream.clone()).w()?;
        // RUN-161: bind a persistent, user-owned cuBLAS workspace. cuBLAS's
        // default workspace is lazily allocated on the stream and is NOT
        // CUDA-graph-capture safe: a gemm captured into a graph produces
        // corrupted output (the workspace pointer baked into the graph is not
        // stable / the lazy alloc is not recorded). A fixed, leaked workspace
        // makes cuBLAS gemm capturable. Hopper recommends 32 MiB.
        {
            let ws_bytes = 32 * 1024 * 1024usize;
            let ws = stream.alloc_zeros::<u8>(ws_bytes).w()?;
            let ws_ptr = ws.leak();
            unsafe {
                cudarc::cublas::sys::cublasSetWorkspace_v2(
                    *blas.handle(),
                    ws_ptr as *mut core::ffi::c_void,
                    ws_bytes,
                )
                .result()
                .w()?;
            }
        }
        let curand = cudarc::curand::CudaRng::new(299792458, stream.clone()).w()?;
        let module_store = ModuleStore {
            mdls: [const { None }; kernels::ALL_IDS.len()],
        };
        Ok(Self {
            id: DeviceId::new(),
            context,
            stream,
            blas: Arc::new(blas),
            curand: Arc::new(Mutex::new(CudaRng(curand))),
            modules: Arc::new(std::sync::RwLock::new(module_store)),
            custom_modules: Arc::new(std::sync::RwLock::new(HashMap::new())),
            seed_value: Arc::new(RwLock::new(299792458)),
            alloc_cache: Arc::new(Mutex::new(AllocCache::default())),
        })
    }
}

impl BackendDevice for CudaDevice {
    type Storage = CudaStorage;

    fn new(ordinal: usize) -> Result<Self> {
        let context = cudarc::driver::CudaContext::new(ordinal).w()?;
        let stream = context.default_stream();
        let blas = cudarc::cublas::CudaBlas::new(stream.clone()).w()?;
        let curand = cudarc::curand::CudaRng::new(299792458, stream.clone()).w()?;
        let module_store = ModuleStore {
            mdls: [const { None }; kernels::ALL_IDS.len()],
        };
        Ok(Self {
            id: DeviceId::new(),
            context,
            stream,
            blas: Arc::new(blas),
            curand: Arc::new(Mutex::new(CudaRng(curand))),
            modules: Arc::new(std::sync::RwLock::new(module_store)),
            custom_modules: Arc::new(std::sync::RwLock::new(HashMap::new())),
            seed_value: Arc::new(RwLock::new(299792458)),
            alloc_cache: Arc::new(Mutex::new(AllocCache::default())),
        })
    }

    fn set_seed(&self, seed: u64) -> Result<()> {
        // We do not call set_seed but instead create a new curand object. This ensures that the
        // state will be identical and the same random numbers will be generated.
        let mut curand = self.curand.lock().unwrap();
        curand.0 = cudarc::curand::CudaRng::new(seed, self.stream.clone()).w()?;
        *self.seed_value.write().unwrap() = seed;
        Ok(())
    }

    fn get_current_seed(&self) -> Result<u64> {
        Ok(*self.seed_value.read().unwrap())
    }

    fn location(&self) -> crate::DeviceLocation {
        crate::DeviceLocation::Cuda {
            gpu_id: self.context.ordinal(),
        }
    }

    fn same_device(&self, rhs: &Self) -> bool {
        self.id == rhs.id
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<CudaStorage> {
        let elem_count = shape.elem_count();
        let slice = match dtype {
            DType::U8 => {
                let data = self.alloc_zeros::<u8>(elem_count)?;
                CudaStorageSlice::U8(data)
            }
            DType::U32 => {
                let data = self.alloc_zeros::<u32>(elem_count)?;
                CudaStorageSlice::U32(data)
            }
            DType::I16 => {
                let data = self.alloc_zeros::<i16>(elem_count)?;
                CudaStorageSlice::I16(data)
            }
            DType::I32 => {
                let data = self.alloc_zeros::<i32>(elem_count)?;
                CudaStorageSlice::I32(data)
            }
            DType::I64 => {
                let data = self.alloc_zeros::<i64>(elem_count)?;
                CudaStorageSlice::I64(data)
            }
            DType::BF16 => {
                let data = self.alloc_zeros::<bf16>(elem_count)?;
                CudaStorageSlice::BF16(data)
            }
            DType::F16 => {
                let data = self.alloc_zeros::<f16>(elem_count)?;
                CudaStorageSlice::F16(data)
            }
            DType::F32 => {
                let data = self.alloc_zeros::<f32>(elem_count)?;
                CudaStorageSlice::F32(data)
            }
            DType::F64 => {
                let data = self.alloc_zeros::<f64>(elem_count)?;
                CudaStorageSlice::F64(data)
            }
            DType::F8E4M3 => {
                let data = self.alloc_zeros::<F8E4M3>(elem_count)?;
                CudaStorageSlice::F8E4M3(data)
            }
            DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::F8E8M0 => {
                return Err(
                    CudaError::InternalError("Dummy types not supported in CUDA backend").into(),
                )
            }
        };
        Ok(CudaStorage {
            slice: std::mem::ManuallyDrop::new(slice),
            device: self.clone(),
        })
    }

    fn rand_uniform(&self, shape: &Shape, dtype: DType, lo: f64, up: f64) -> Result<CudaStorage> {
        let elem_count = shape.elem_count();
        let curand = self.curand.lock().unwrap();
        let slice = match dtype {
            // TODO: Add support for F16 and BF16 though this is likely to require some upstream
            // cudarc changes.
            DType::U8
            | DType::U32
            | DType::I16
            | DType::I32
            | DType::I64
            | DType::F16
            | DType::BF16 => Err(CudaError::UnsupportedDtype {
                dtype,
                op: "rand_uniform",
            })
            .w()?,
            DType::F32 => {
                let mut data = unsafe { self.alloc::<f32>(elem_count)? };
                curand.0.fill_with_uniform(&mut data).w()?;
                CudaStorageSlice::F32(data)
            }
            DType::F64 => {
                let mut data = unsafe { self.alloc::<f64>(elem_count)? };
                curand.0.fill_with_uniform(&mut data).w()?;
                CudaStorageSlice::F64(data)
            }
            DType::F8E4M3 | DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::F8E8M0 => {
                Err(CudaError::UnsupportedDtype {
                    dtype,
                    op: "rand_uniform",
                })
                .w()?
            }
        };
        let slice = if lo == 0. && up == 1.0 {
            slice
        } else {
            use super::utils::Map1;
            let layout = Layout::contiguous(shape);
            super::Affine(up - lo, lo).map(&slice, self, &layout)?
        };
        Ok(CudaStorage {
            slice: std::mem::ManuallyDrop::new(slice),
            device: self.clone(),
        })
    }

    fn rand_normal(&self, shape: &Shape, dtype: DType, mean: f64, std: f64) -> Result<CudaStorage> {
        // TODO: Add support for F16 and BF16 though this is likely to require some upstream
        // cudarc changes.
        let elem_count = shape.elem_count();
        let curand = self.curand.lock().unwrap();
        // curand can only generate an odd number of values.
        // https://github.com/huggingface/candle/issues/734
        let elem_count_round = if elem_count % 2 == 1 {
            elem_count + 1
        } else {
            elem_count
        };
        let slice = match dtype {
            DType::U8
            | DType::U32
            | DType::I16
            | DType::I32
            | DType::I64
            | DType::F16
            | DType::BF16 => Err(CudaError::UnsupportedDtype {
                dtype,
                op: "rand_normal",
            })
            .w()?,
            DType::F32 => {
                let mut data = unsafe { self.alloc::<f32>(elem_count_round)? };
                curand
                    .0
                    .fill_with_normal(&mut data, mean as f32, std as f32)
                    .w()?;
                CudaStorageSlice::F32(data)
            }
            DType::F64 => {
                let mut data = unsafe { self.alloc::<f64>(elem_count_round)? };
                curand.0.fill_with_normal(&mut data, mean, std).w()?;
                CudaStorageSlice::F64(data)
            }
            DType::F8E4M3 | DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::F8E8M0 => {
                Err(CudaError::UnsupportedDtype {
                    dtype,
                    op: "rand_normal",
                })
                .w()?
            }
        };
        Ok(CudaStorage {
            slice: std::mem::ManuallyDrop::new(slice),
            device: self.clone(),
        })
    }

    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let elem_count = shape.elem_count();
        let slice = match dtype {
            DType::U8 => {
                let data = self.alloc::<u8>(elem_count)?;
                CudaStorageSlice::U8(data)
            }
            DType::U32 => {
                let data = self.alloc::<u32>(elem_count)?;
                CudaStorageSlice::U32(data)
            }
            DType::I16 => {
                let data = self.alloc::<i16>(elem_count)?;
                CudaStorageSlice::I16(data)
            }
            DType::I32 => {
                let data = self.alloc::<i32>(elem_count)?;
                CudaStorageSlice::I32(data)
            }
            DType::I64 => {
                let data = self.alloc::<i64>(elem_count)?;
                CudaStorageSlice::I64(data)
            }
            DType::BF16 => {
                let data = self.alloc::<bf16>(elem_count)?;
                CudaStorageSlice::BF16(data)
            }
            DType::F16 => {
                let data = self.alloc::<f16>(elem_count)?;
                CudaStorageSlice::F16(data)
            }
            DType::F32 => {
                let data = self.alloc::<f32>(elem_count)?;
                CudaStorageSlice::F32(data)
            }
            DType::F64 => {
                let data = self.alloc::<f64>(elem_count)?;
                CudaStorageSlice::F64(data)
            }
            DType::F8E4M3 => {
                let data = self.alloc::<F8E4M3>(elem_count)?;
                CudaStorageSlice::F8E4M3(data)
            }
            DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::F8E8M0 => {
                return Err(
                    CudaError::InternalError("Dummy types not supported in CUDA backend").into(),
                )
            }
        };
        Ok(CudaStorage {
            slice: std::mem::ManuallyDrop::new(slice),
            device: self.clone(),
        })
    }

    fn storage_from_slice<T: crate::WithDType>(&self, s: &[T]) -> Result<Self::Storage> {
        let slice = match T::cpu_storage_ref(s) {
            CpuStorageRef::U8(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::U8(data)
            }
            CpuStorageRef::U32(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::U32(data)
            }
            CpuStorageRef::I16(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::I16(data)
            }
            CpuStorageRef::I32(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::I32(data)
            }
            CpuStorageRef::I64(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::I64(data)
            }
            CpuStorageRef::BF16(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::BF16(data)
            }
            CpuStorageRef::F16(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::F16(data)
            }
            CpuStorageRef::F32(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::F32(data)
            }
            CpuStorageRef::F64(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::F64(data)
            }
            CpuStorageRef::F8E4M3(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::F8E4M3(data)
            }
            CpuStorageRef::F4(_)
            | CpuStorageRef::F6E2M3(_)
            | CpuStorageRef::F6E3M2(_)
            | CpuStorageRef::F8E8M0(_) => {
                return Err(CudaError::UnsupportedDtype {
                    dtype: T::DTYPE,
                    op: "storage_from_slice",
                }
                .into());
            }
        };
        Ok(CudaStorage {
            slice: std::mem::ManuallyDrop::new(slice),
            device: self.clone(),
        })
    }

    fn storage_from_cpu_storage(&self, storage: &CpuStorage) -> Result<CudaStorage> {
        let slice = match storage {
            CpuStorage::U8(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::U8(data)
            }
            CpuStorage::U32(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::U32(data)
            }
            CpuStorage::I16(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::I16(data)
            }
            CpuStorage::I32(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::I32(data)
            }
            CpuStorage::I64(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::I64(data)
            }
            CpuStorage::BF16(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::BF16(data)
            }
            CpuStorage::F16(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::F16(data)
            }
            CpuStorage::F32(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::F32(data)
            }
            CpuStorage::F64(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::F64(data)
            }
            CpuStorage::F8E4M3(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::F8E4M3(data)
            }
            CpuStorage::F4(_)
            | CpuStorage::F6E2M3(_)
            | CpuStorage::F6E3M2(_)
            | CpuStorage::F8E8M0(_) => {
                return Err(CudaError::UnsupportedDtype {
                    dtype: storage.dtype(),
                    op: "storage_from_cpu_storage",
                }
                .into());
            }
        };
        Ok(CudaStorage {
            slice: std::mem::ManuallyDrop::new(slice),
            device: self.clone(),
        })
    }

    fn storage_from_cpu_storage_owned(&self, storage: CpuStorage) -> Result<CudaStorage> {
        let slice = match storage {
            CpuStorage::U8(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::U8(data)
            }
            CpuStorage::U32(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::U32(data)
            }
            CpuStorage::I16(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::I16(data)
            }
            CpuStorage::I32(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::I32(data)
            }
            CpuStorage::I64(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::I64(data)
            }
            CpuStorage::BF16(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::BF16(data)
            }
            CpuStorage::F16(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::F16(data)
            }
            CpuStorage::F32(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::F32(data)
            }
            CpuStorage::F64(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::F64(data)
            }
            CpuStorage::F8E4M3(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::F8E4M3(data)
            }
            CpuStorage::F4(_)
            | CpuStorage::F6E2M3(_)
            | CpuStorage::F6E3M2(_)
            | CpuStorage::F8E8M0(_) => {
                return Err(CudaError::UnsupportedDtype {
                    dtype: storage.dtype(),
                    op: "storage_from_cpu_storage_owned",
                }
                .into());
            }
        };
        Ok(CudaStorage {
            slice: std::mem::ManuallyDrop::new(slice),
            device: self.clone(),
        })
    }

    fn synchronize(&self) -> Result<()> {
        self.stream.synchronize().map_err(crate::Error::wrap)?;
        Ok(())
    }
}
