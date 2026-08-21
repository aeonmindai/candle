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

// ---------------------------------------------------------------------------
// Size-class ladder
// ---------------------------------------------------------------------------

/// Quarter-power-of-two size classes, the standard ladder shape (jemalloc,
/// PyTorch's `CUDACachingAllocator`): four evenly spaced classes per octave.
///
/// This cache does **not** serve a request from a different size than it asks
/// for — see [`AllocCache::take`] for why that would be unsound here — so the
/// ladder is not used for matching. It is used to keep the diagnostic `missed`
/// set bounded: byte counts track KV length and are unbounded, size classes are
/// not (221 of them on a 64-bit target).
const LADDER_BASE_LOG2: u32 = 9; // 512 B
const SUBCLASSES: usize = 4;

/// The size class `bytes` falls in. Monotonic non-decreasing in `bytes`.
#[inline]
fn bucket_index(bytes: usize) -> usize {
    if bytes < (1usize << LADDER_BASE_LOG2) {
        return 0;
    }
    // floor(log2(bytes)); >= LADDER_BASE_LOG2 by the branch above.
    let k = usize::BITS - 1 - bytes.leading_zeros();
    // Which quarter of [2^k, 2^(k+1)) `bytes` sits in.
    let sub = (bytes - (1usize << k)) >> (k - 2);
    1 + ((k - LADDER_BASE_LOG2) as usize) * SUBCLASSES + sub as usize
}

/// Cached buffers of one exact byte size, plus the recency stamp that orders
/// them for eviction.
#[derive(Default)]
struct SizeClass {
    ptrs: Vec<cudarc::driver::sys::CUdeviceptr>,
    /// Monotonic stamp of the last put or hit at this size. Eviction order.
    tick: u64,
}

/// Counters for the caching allocator. Every number here is a count of a real
/// driver call or a real retained byte — nothing is inferred.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AllocCacheStats {
    /// Allocations served from the cache (no `cuMemAllocAsync`).
    pub hits: u64,
    /// Allocations that fell through to `cuMemAllocAsync` **while the cache was
    /// enabled**. This is "allocations per step" for a decode loop.
    pub misses: u64,
    /// Buffers accepted into the cache instead of being freed.
    pub puts: u64,
    /// Buffers handed back to the driver by capacity eviction.
    pub evicted: u64,
    /// Buffers handed back to the driver by `set_alloc_cache_enabled(false)` or
    /// `drain_alloc_cache_and_free()`.
    pub drained: u64,
    /// Buffers allocated straight from the driver by `prewarm_alloc_cache` and
    /// filed into the cache. These are real `cuMemAllocAsync` calls, but they
    /// are neither `misses` (nothing fell through the cache) nor `puts`
    /// (nothing was freed), so they get their own counter rather than putting a
    /// one-time spike into a per-step number.
    pub prewarmed: u64,
    /// Eviction victims whose byte size is in the capture demand profile.
    ///
    /// **Should be zero while an instantiated graph can still replay.** LRU
    /// cannot see which addresses a graph baked; a non-zero value here is the
    /// fingerprint of a replay-time use-after-free waiting to happen. See
    /// `AllocCache::evict_if_over_capacity`.
    pub evicted_at_demand_size: u64,
    /// Bytes currently retained (`free` + `deferred`), pre-warmed buffers
    /// included.
    pub cached_bytes: usize,
    /// Buffers currently retained.
    pub cached_buffers: usize,
    /// Largest `cached_bytes` ever reached. The number that says whether the
    /// capacity is doing anything.
    ///
    /// Note the pre-warm widens this: `put` overshoots the cap by at most one
    /// allocation, but `prewarm_alloc_cache` installs a whole demand profile
    /// without evicting, so the peak can exceed the cap by the size of that
    /// profile until the first ordinary put after the capture window.
    pub high_water_bytes: usize,
    /// The cap. `usize::MAX` means unbounded.
    pub capacity_bytes: usize,
    /// Distinct (size class, physical size) groups currently held.
    pub size_classes: usize,
    /// Distinct byte sizes in the capture demand profile.
    pub demand_sizes: usize,
    /// Total entries in the capture-miss ledger since the last
    /// `reset_capture_misses`. Same number as `capture_miss_count()`.
    pub capture_misses: usize,
}

impl AllocCacheStats {
    /// Buffers returned to the driver, by any route. "Frees per step".
    pub fn frees(&self) -> u64 {
        self.evicted + self.drained
    }

    /// Real `cuMemAllocAsync` calls the cache caused: allocations that fell
    /// through it, plus the ones the pre-warm made on purpose.
    pub fn driver_allocs(&self) -> u64 {
        self.misses + self.prewarmed
    }
}

/// Default cap: 1 GiB. Chosen to be far larger than a decode step's working set
/// and far smaller than any card this runs on, so the cap only ever bites the
/// unbounded-growth case it exists to stop. Override with
/// `CANDLE_ALLOC_CACHE_MAX_MB` (`0` = unbounded, which is the old behaviour).
const DEFAULT_CAPACITY_BYTES: usize = 1024 * 1024 * 1024;

/// Opt-in CUDA caching allocator for graph-capture safety (RUN-161).
///
/// When `enabled`, freed device buffers are returned to `free` instead of
/// `cuMemFreeAsync`, and `alloc` reuses them via `upgrade_device_ptr` instead of
/// `cuMemAllocAsync`. Warming the cache (a few eager forwards at the capture
/// shapes) then makes a captured forward allocation-free: stable addresses (no
/// MMU fault), no alloc/free graph nodes, no cross-stream capture isolation.
/// Default OFF -> candle's alloc/free is byte-identical to upstream, so existing
/// models are unaffected.
///
/// # Bounded, and why it has to be
///
/// The first version of this cache had **no capacity and no eviction**: the only
/// ways memory went back to the driver were `set_alloc_cache_enabled(false)` and
/// `drain_alloc_cache_and_free()`. That is fatal for a long generation, because
/// a decode step allocates buffers whose size tracks the KV length — measured on
/// DeepSeek-V4, a family of ~132 buffers stepping by 8 KiB per token. Each one
/// is a byte size nothing ever asks for again, so the cache filed ~132
/// permanently-dead entries per token and freed nothing. Measured growth: **6.04
/// MiB per decoded token with no plateau** (`memory.used`, 2 Hz, 2 600 tokens),
/// against exactly 0.00 for the same run with the cache off.
///
/// So retention is now capped. `cached_bytes` is tracked exactly; crossing
/// `capacity` frees least-recently-used buffers back to the driver until
/// retention is under the low-water mark. Buffers reused every step keep a fresh
/// tick and survive; the per-step-unique sizes go stale and are evicted first.
/// That bounds retention unconditionally, whatever the shape traffic looks like.
///
/// # Why sizes are matched exactly and not by class
///
/// The obvious next step — group nearby sizes so a 106 496 byte buffer can serve
/// a 114 688 byte request — is **not sound here**, and the reason is worth
/// recording so it is not re-attempted.
///
/// A buffer's size is not stored anywhere. It is recomputed on free, in
/// `CudaStorageSlice::byte_len_and_leak`, as `slice.len() * size_of::<T>()` —
/// the length of the slice the *requester* was handed. Serve a 128-byte buffer
/// to a 120-byte request and it comes back recorded as 120. Serve that to a
/// 100-byte request and it comes back as 100. The recorded size ratchets down
/// on every reuse while the real allocation stays 128, so `cached_bytes`
/// silently under-counts and the cap stops bounding anything. Class-based
/// matching therefore needs per-pointer physical sizes (a hash lookup on every
/// free, ~11 k of them per token) before it is safe at all.
///
/// It also buys much less than it looks like it does. Class matching would drive
/// the ~132 misses per step towards zero — worth roughly 0.26 ms/token of
/// `cuMemAllocAsync` — while the 3.73 ms/token this cache is actually worth
/// comes from absorbing the other ~11 300 allocations, which exact matching
/// already does. The cap is what fixes the leak; classes were the hypothesis.
pub struct AllocCache {
    enabled: bool,
    /// While `capturing`, buffers freed (Drop -> cache_put) are NOT returned to
    /// `free` (which would let them be re-served within the SAME capture,
    /// recording two graph uses of one address -> aliasing corruption / MMU
    /// fault). They are parked in `deferred` and moved back to `free` only when
    /// capture mode ends. This guarantees every allocation during a captured
    /// forward gets a unique, stable address. (RUN-161, PyTorch-style.)
    capturing: bool,
    /// Free buffers, keyed by their exact byte size.
    free: HashMap<usize, SizeClass>,
    /// Buffers freed during capture, parked until capture mode ends.
    deferred: Vec<(usize, cudarc::driver::sys::CUdeviceptr)>,
    /// Debug: size classes that have missed the cache (logged once each under
    /// ARC_CACHE_DEBUG). A new miss DURING capture = an allocation that becomes
    /// an unstable graph memory node -> the cause of the launch fault/corruption.
    /// Keyed by class rather than byte count so it cannot itself grow without
    /// bound.
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
    ///
    /// Bounded only by the caller's discipline: it is cleared by
    /// `reset_capture_misses`, never automatically. See that method for why the
    /// reset is deliberately not folded into `set_capture_mode(true)`.
    capture_misses: HashMap<usize, usize>,
    /// Bytes retained in `free` + `deferred`. Maintained incrementally; the
    /// tests assert it against a full recount.
    cached_bytes: usize,
    /// Buffers retained in `free` + `deferred`.
    cached_buffers: usize,
    /// Hard cap on `cached_bytes`. `usize::MAX` = unbounded.
    capacity: usize,
    /// Monotonic stamp source for LRU.
    tick: u64,
    hits: u64,
    misses: u64,
    puts: u64,
    evicted: u64,
    drained: u64,
    /// Buffers installed by `prewarm_alloc_cache`. Counted separately from
    /// `puts` and `misses` on purpose: a prewarmed buffer was never freed by
    /// anyone (so it is not a `put`) and was not an allocation that fell
    /// through the cache (so it is not a `miss`), but it *is* a real
    /// `cuMemAllocAsync`. Folding it into either would put a one-time spike in
    /// a per-step number and make the cache look broken on the step it was
    /// warmed.
    prewarmed: u64,
    /// Eviction victims whose byte size appears in the capture demand profile.
    ///
    /// Diagnostic for the one interaction this merge cannot settle on host
    /// state: LRU eviction does not know that a size in `demand` may be baked
    /// into an instantiated graph's memory nodes. Non-zero while a graph is
    /// live is the fingerprint of a replay-time use-after-free. See
    /// `evict_if_over_capacity`.
    evicted_at_demand_size: u64,
    high_water_bytes: usize,
}

impl Default for AllocCache {
    fn default() -> Self {
        let capacity = match std::env::var("CANDLE_ALLOC_CACHE_MAX_MB") {
            Ok(v) => match v.trim().parse::<usize>() {
                // 0 means "no cap" — the pre-bounding behaviour, kept reachable
                // so a regression can be A/B'd against it.
                Ok(0) => usize::MAX,
                Ok(mb) => mb.saturating_mul(1024 * 1024),
                Err(_) => DEFAULT_CAPACITY_BYTES,
            },
            Err(_) => DEFAULT_CAPACITY_BYTES,
        };
        Self {
            enabled: false,
            capturing: false,
            free: HashMap::new(),
            deferred: Vec::new(),
            missed: std::collections::HashSet::new(),
            window: HashMap::new(),
            demand: HashMap::new(),
            capture_misses: HashMap::new(),
            cached_bytes: 0,
            cached_buffers: 0,
            capacity,
            tick: 0,
            hits: 0,
            misses: 0,
            puts: 0,
            evicted: 0,
            drained: 0,
            prewarmed: 0,
            evicted_at_demand_size: 0,
            high_water_bytes: 0,
        }
    }
}

impl AllocCache {
    /// Take a buffer of exactly `bytes` from the cache, or `None`.
    ///
    /// Exact match only. The returned buffer is therefore always exactly the
    /// requested size, which is the invariant `cached_bytes` accounting — and so
    /// the capacity bound — depends on. See the type docs for why a size-class
    /// match would break it.
    ///
    /// While `capturing`, this is also where the two capture instruments are
    /// fed. They count different things and neither substitutes for the other:
    ///
    /// - `window` counts **allocations**, hit or miss, because what the pre-warm
    ///   must supply is the number of buffers the captured forward asks for, not
    ///   the number of times the cache failed to supply one.
    /// - `capture_misses` counts **misses only**, never deduped, because one
    ///   miss inside the real capture is one unstable graph memory node.
    ///
    /// `misses` (master's driver-call counter) is incremented on the same path
    /// and is a strict superset: `capture_misses` is the slice of it that
    /// happened while `capturing`. They are separate fields, so there is no
    /// double counting in either.
    fn take(&mut self, bytes: usize) -> Option<cudarc::driver::sys::CUdeviceptr> {
        self.tick += 1;
        let tick = self.tick;
        if self.capturing {
            // Demand accounting for the window currently open. Counted before
            // the lookup so a hit and a miss weigh the same.
            *self.window.entry(bytes).or_insert(0) += 1;
        }
        // Note the shape: every path that does not return a buffer must count a
        // miss, including "the size is known but its stack is empty". `?` on the
        // lookup would skip the counter and make `misses` — the number this
        // change is judged on — quietly wrong.
        let hit = self
            .free
            .get_mut(&bytes)
            .and_then(|sc| sc.ptrs.pop().inspect(|_| sc.tick = tick));
        match hit {
            Some(ptr) => {
                self.cached_bytes -= bytes;
                self.cached_buffers -= 1;
                self.hits += 1;
                Some(ptr)
            }
            None => {
                self.misses += 1;
                if self.capturing {
                    *self.capture_misses.entry(bytes).or_insert(0) += 1;
                }
                None
            }
        }
    }

    /// Park `ptr` (exactly `bytes` long) in the cache. Returns pointers the
    /// caller must hand back to the driver: eviction victims, if this put
    /// pushed retention over capacity.
    fn put(
        &mut self,
        bytes: usize,
        ptr: cudarc::driver::sys::CUdeviceptr,
    ) -> Vec<cudarc::driver::sys::CUdeviceptr> {
        self.tick += 1;
        let tick = self.tick;
        self.puts += 1;
        if self.capturing {
            self.deferred.push((bytes, ptr));
        } else {
            let sc = self.free.entry(bytes).or_default();
            sc.ptrs.push(ptr);
            sc.tick = tick;
        }
        self.cached_bytes += bytes;
        self.cached_buffers += 1;
        self.high_water_bytes = self.high_water_bytes.max(self.cached_bytes);
        self.evict_if_over_capacity()
    }

    /// Free least-recently-used buffers until retention is back under the
    /// low-water mark (7/8 of capacity, so a cache sitting exactly at the cap
    /// does not evict on every single put).
    ///
    /// `deferred` is never evicted: those buffers were freed *during* a capture
    /// and handing their addresses back to the driver mid-capture is precisely
    /// the aliasing the deferral exists to prevent. Captures are short and
    /// bounded, so nothing is lost by waiting.
    ///
    /// # Hazard this function does NOT handle
    ///
    /// LRU has no idea which addresses an *instantiated* graph baked into its
    /// memory nodes. After a capture, the buffers the captured forward used are
    /// dropped by the eager tensors that held them and land right back in
    /// `free`, indistinguishable from any other cached buffer. Evicting one
    /// hands the driver back memory the next `cuGraphLaunch` will write into.
    ///
    /// That hazard predates the pre-warm — it follows from eviction plus
    /// replay — but the pre-warm makes it far likelier to bite, because it
    /// deliberately installs a large, long-lived, replay-critical set of
    /// buffers at exactly the sizes in `demand`. Nothing here prevents it;
    /// `evicted_at_demand_size` only makes it *visible*, so the condition can be
    /// asserted on rather than inferred from a crash.
    fn evict_if_over_capacity(&mut self) -> Vec<cudarc::driver::sys::CUdeviceptr> {
        if self.capacity == usize::MAX || self.cached_bytes <= self.capacity || self.capturing {
            return Vec::new();
        }
        // Evict to 7/8 of capacity rather than exactly to it, so a cache sitting
        // at the cap does not run this sweep on every single put.
        let low_water = self.capacity / 8 * 7;
        // Oldest size first.
        let mut order: Vec<(u64, usize)> = self
            .free
            .iter()
            .filter(|(_, sc)| !sc.ptrs.is_empty())
            .map(|(bytes, sc)| (sc.tick, *bytes))
            .collect();
        order.sort_unstable();
        let mut victims = Vec::new();
        let mut demand_victims = 0u64;
        for (_, bytes) in order {
            if self.cached_bytes <= low_water {
                break;
            }
            let in_demand = self.demand.contains_key(&bytes);
            let Some(sc) = self.free.get_mut(&bytes) else {
                continue;
            };
            while self.cached_bytes > low_water {
                match sc.ptrs.pop() {
                    Some(p) => {
                        victims.push(p);
                        self.cached_bytes -= bytes;
                        self.cached_buffers -= 1;
                        self.evicted += 1;
                        if in_demand {
                            demand_victims += 1;
                        }
                    }
                    None => break,
                }
            }
        }
        self.evicted_at_demand_size += demand_victims;
        // Drop emptied sizes so the map does not grow without bound either — it
        // is keyed by byte count, and those track KV length.
        self.free.retain(|_, sc| !sc.ptrs.is_empty());
        victims
    }

    /// Hand every retained buffer back. Used by `set_alloc_cache_enabled(false)`
    /// and `drain_alloc_cache_and_free()`.
    ///
    /// The open capture window is discarded here rather than folded into
    /// `demand`: it describes a pool that is about to stop existing, so
    /// recording it would state a demand no buffer in this cache can serve.
    /// `demand` itself survives — it is a fact about the model's shapes, not
    /// about this pool — and so does `capture_misses`, which is the caller's to
    /// clear via `reset_capture_misses`.
    fn drain_all(&mut self) -> Vec<cudarc::driver::sys::CUdeviceptr> {
        self.capturing = false;
        self.window.clear();
        let mut d = Vec::with_capacity(self.cached_buffers);
        for (_, sc) in self.free.drain() {
            d.extend(sc.ptrs);
        }
        d.extend(self.deferred.drain(..).map(|(_, p)| p));
        self.drained += d.len() as u64;
        self.cached_bytes = 0;
        self.cached_buffers = 0;
        d
    }

    /// Open a capture-mode window: the per-size allocation tally starts empty.
    fn open_capture_window(&mut self) {
        self.window.clear();
    }

    /// Close a capture-mode window, folding its per-size allocation counts into
    /// the demand profile with a per-size **max** — so N warm windows cover a
    /// period-N size cycle rather than the last window overwriting the others.
    fn close_capture_window(&mut self) {
        let window = std::mem::take(&mut self.window);
        for (bytes, n) in window {
            let slot = self.demand.entry(bytes).or_insert(0);
            *slot = (*slot).max(n);
        }
    }

    /// How many buffers of each size the pre-warm must allocate to bring `free`
    /// up to `demand + slack`. Read-only: the allocation happens outside the
    /// lock, and `install_prewarmed` commits the result.
    fn prewarm_plan(&self, slack: usize) -> Vec<(usize, usize)> {
        self.demand
            .iter()
            .filter_map(|(&bytes, &need)| {
                let have = self.free.get(&bytes).map_or(0, |sc| sc.ptrs.len());
                let want = need + slack;
                (bytes > 0 && want > have).then_some((bytes, want - have))
            })
            .collect()
    }

    /// File freshly-allocated buffers into `free`, with full retention
    /// accounting and **without eviction**.
    ///
    /// # Why this does not go through `put`, and does not evict
    ///
    /// `put` calls `evict_if_over_capacity`, and eviction inside a pre-warm is
    /// self-defeating in two independent ways.
    ///
    /// 1. **It reclaims what the pre-warm just installed.** LRU orders by
    ///    `SizeClass::tick`, and the pre-warm walks the demand profile in
    ///    `HashMap` order. A sweep triggered by the last size installed would
    ///    take its victims from the *oldest* ticks — the demand-profile sizes
    ///    topped up earlier in the same pass. The pre-warm would then hand back
    ///    a cache that provably cannot serve the capture, non-deterministically,
    ///    and the capture-miss gate would blame the profile for it.
    /// 2. **The addresses are replay-critical.** These are precisely the buffers
    ///    the about-to-be-captured graph will bake into its memory nodes. If a
    ///    graph from an earlier capture is still instantiated, its addresses are
    ///    in this same pool, and freeing one is a device-side use-after-free on
    ///    its next replay.
    ///
    /// Retention may therefore exceed `capacity` from here until the first
    /// ordinary `put` after the capture window closes. That is the same bounded
    /// exception `set_capture_mode` already takes, for the same reason: inside
    /// the capture protocol, `drain_alloc_cache_and_free` must be the only thing
    /// that hands buffers back. The caller is told about the overshoot rather
    /// than left to discover it — see `CudaDevice::prewarm_alloc_cache`.
    ///
    /// Counted as `prewarmed`, not as `puts` or `misses`: see the field docs.
    fn install_prewarmed(&mut self, bytes: usize, ptrs: Vec<cudarc::driver::sys::CUdeviceptr>) {
        if ptrs.is_empty() {
            return;
        }
        self.tick += 1;
        let tick = self.tick;
        let n = ptrs.len();
        let sc = self.free.entry(bytes).or_default();
        sc.ptrs.extend(ptrs);
        // A freshly pre-warmed size is the hottest thing in the cache: it is
        // about to be consumed by the capture. Stamping it keeps it at the back
        // of the eviction queue for as long as LRU can manage.
        sc.tick = tick;
        self.cached_bytes += bytes * n;
        self.cached_buffers += n;
        self.prewarmed += n as u64;
        self.high_water_bytes = self.high_water_bytes.max(self.cached_bytes);
    }

    fn stats(&self) -> AllocCacheStats {
        AllocCacheStats {
            hits: self.hits,
            misses: self.misses,
            puts: self.puts,
            evicted: self.evicted,
            drained: self.drained,
            prewarmed: self.prewarmed,
            evicted_at_demand_size: self.evicted_at_demand_size,
            cached_bytes: self.cached_bytes,
            cached_buffers: self.cached_buffers,
            high_water_bytes: self.high_water_bytes,
            capacity_bytes: self.capacity,
            size_classes: self.free.values().filter(|sc| !sc.ptrs.is_empty()).count(),
            demand_sizes: self.demand.len(),
            capture_misses: self.capture_misses.values().sum(),
        }
    }

    /// Recount `cached_bytes`/`cached_buffers` from scratch. Only for tests —
    /// the incremental counters are what the allocator actually uses, so a test
    /// that recomputes them is the thing that proves they never drift.
    #[cfg(test)]
    fn recount(&self) -> (usize, usize) {
        let mut bytes = 0;
        let mut n = 0;
        for (sz, sc) in self.free.iter() {
            bytes += sz * sc.ptrs.len();
            n += sc.ptrs.len();
        }
        for (b, _) in self.deferred.iter() {
            bytes += b;
            n += 1;
        }
        (bytes, n)
    }
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
                // Also drops the open capture window and clears `capturing` —
                // see `AllocCache::drain_all`. `demand` survives on purpose.
                cache.drain_all()
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

    /// Counters for the caching allocator: driver allocations, driver frees,
    /// bytes retained, high-water mark. Cheap enough to poll every decode step.
    ///
    /// `misses` is the number of real `cuMemAllocAsync` calls made while the
    /// cache was on, and `frees()` the number of real `cuMemFreeAsync` calls it
    /// caused. A cache that is working shows a low `misses` delta per step and a
    /// **non-zero** `frees()` delta once it reaches capacity; a cache that is
    /// leaking shows `frees() == 0` and `cached_bytes` climbing without bound.
    pub fn alloc_cache_stats(&self) -> AllocCacheStats {
        self.alloc_cache.lock().unwrap().stats()
    }

    /// Set the retention cap in bytes. `usize::MAX` disables the bound.
    ///
    /// Applied immediately: lowering it below current retention evicts down to
    /// the new low-water mark before returning.
    pub fn set_alloc_cache_capacity(&self, capacity_bytes: usize) {
        let victims = {
            let mut cache = self.alloc_cache.lock().unwrap();
            cache.capacity = capacity_bytes;
            cache.evict_if_over_capacity()
        };
        for ptr in victims {
            drop(unsafe { self.stream.upgrade_device_ptr::<u8>(ptr, 0) });
        }
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
            // Sizes that missed before mean nothing once every buffer is gone.
            cache.missed.clear();
            // `drain_all` also clears `capturing` and the open capture window.
            // It does NOT clear `capture_misses`: that ledger is scoped to a
            // capture by the caller's `reset_capture_misses`, and a drain in
            // the middle of one would silently erase the evidence the gate
            // exists to read.
            cache.drain_all()
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
        //
        // ⚠ This early return switches off the CAPTURE INSTRUMENTS as well, and
        // it does so silently. `capturing` never becomes true, so:
        //   * no capture window ever opens -> `demand` stays empty -> a
        //     `prewarm_alloc_cache` after it plans nothing and reports (0, 0);
        //   * no miss is ever filed in `capture_misses` -> `capture_miss_count()`
        //     returns 0 and the caller's "refuse to instantiate on a miss" gate
        //     passes unconditionally.
        // A run with ARC_NO_DEFERRED_FREE set therefore looks exactly like a
        // clean capture and is not one. Do not read a zero from either
        // instrument without checking this variable first. (Behaviour predates
        // both merged lines; recorded here, not changed here.)
        if std::env::var_os("ARC_NO_DEFERRED_FREE").is_some() {
            return;
        }
        // Leaving capture returns the parked buffers to the free pool.
        //
        // Deliberately WITHOUT eviction, even though this can leave retention
        // over the cap until the next ordinary put. Capture installs a private
        // memory pool as the device default, and buffers allocated during a
        // capture come from it; `AllocCache` records no pool provenance, so it
        // cannot tell those apart from default-pool buffers. Freeing one after
        // its pool is destroyed corrupts the driver's host-side bookkeeping —
        // which lives in the process's glibc arena, and surfaces as `corrupted
        // size vs. prev_size` at an arbitrary later allocation. See
        // `drain_alloc_cache_and_free`.
        //
        // The capture path already drains the cache explicitly before every
        // `cuMemPoolDestroy`. That discipline only works if draining is the
        // *only* thing that hands buffers back, so eviction — which fires on
        // whatever put happens to cross the cap — must not run inside the
        // capture window. Retention is bounded again on the first decode put
        // after capture, by which point the drain has removed any private-pool
        // pointer.
        //
        // The window that opens/closes here is the *demand profile's* window,
        // not the miss ledger's. The two are deliberately scoped differently:
        // the profile wants the union over every warm window (that is what
        // covers a period-N size cycle), while the ledger must describe exactly
        // one capture. So the profile is folded automatically here and the
        // ledger is cleared only by `reset_capture_misses`, which the caller
        // invokes once, immediately before the real capture.
        let mut cache = self.alloc_cache.lock().unwrap();
        cache.capturing = capturing;
        if capturing {
            cache.open_capture_window();
        } else {
            cache.close_capture_window();
            let deferred = std::mem::take(&mut cache.deferred);
            // Already counted in `cached_bytes` when parked, so only the
            // recency stamp changes hands here.
            cache.tick += 1;
            let tick = cache.tick;
            for (bytes, ptr) in deferred {
                let sc = cache.free.entry(bytes).or_default();
                sc.ptrs.push(ptr);
                sc.tick = tick;
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
    ///
    /// # The ordering is load-bearing, in both directions
    ///
    /// The ledger counts every miss taken while `capturing`, and the capture
    /// protocol sets `capturing` for the deferred-free **warm** passes too —
    /// where missing is the entire point, since that is how `free` grows. So:
    ///
    /// - Reset too early (before the warm passes) and the ledger carries their
    ///   deliberate misses into the gate, which then refuses to instantiate a
    ///   perfectly good graph.
    /// - Reset too late (after the capture forward has begun allocating) and
    ///   real misses are erased, and the gate waves through a graph with
    ///   unstable memory nodes.
    ///
    /// The one correct place is: after `prewarm_alloc_cache`, before
    /// `set_capture_mode(true)` for the real capture.
    ///
    /// This is deliberately **not** folded into `set_capture_mode(true)`.
    /// Auto-resetting there would be indistinguishable from the correct call for
    /// today's caller, and would therefore silently paper over a caller that
    /// forgot it — while also destroying the only way to accumulate a ledger
    /// across more than one window. A counter whose scope is set implicitly is
    /// a counter nobody can reason about.
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
    ///
    /// # Interaction with the capacity bound
    ///
    /// The pre-warm files its buffers **without eviction** — see
    /// `AllocCache::install_prewarmed` for the two reasons — so it can leave
    /// retention above `capacity`. That overshoot is intentional and bounded:
    /// the first ordinary `cache_put` after the capture window closes runs the
    /// sweep and brings retention back under the cap, exactly as
    /// `set_capture_mode` already relies on.
    ///
    /// It is not silent. If the pre-warm pushes retention past the cap this
    /// says so once, with the numbers, because the alternative — a pre-warm that
    /// quietly installs a profile the cap will shred on the next decode step —
    /// looks identical to success right up until a graph replays into freed
    /// memory. If you see it, raise `CANDLE_ALLOC_CACHE_MAX_MB` above
    /// `prewarm bytes + the decode working set`.
    pub fn prewarm_alloc_cache(&self, slack: usize) -> Result<(usize, usize)> {
        // Decide under the lock, allocate outside it: cuMemAllocAsync is slow
        // enough that holding the allocator mutex across it would serialise
        // every other thread's allocations behind the pre-warm.
        let todo: Vec<(usize, usize)> = {
            let cache = self.alloc_cache.lock().unwrap();
            if !cache.enabled {
                return Ok((0, 0));
            }
            if cache.capturing {
                // Contract violation, and one whose symptom is silence: every
                // buffer this installs would be reachable, but the capture that
                // follows would still miss, and the gate would blame the profile.
                eprintln!(
                    "[alloc-cache] WARNING: prewarm_alloc_cache called with capture mode ON. \
                     Buffers freed while capturing park in `deferred` and cannot be served, \
                     so this pre-warm will not do what it looks like it did."
                );
            }
            cache.prewarm_plan(slack)
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
            cache.install_prewarmed(bytes, made);
        }
        let s = self.alloc_cache_stats();
        if s.cached_bytes > s.capacity_bytes {
            eprintln!(
                "[alloc-cache] WARNING: pre-warm left retention at {} B, over the {} B cap. \
                 The first ordinary put after the capture window will evict back under it — \
                 including, possibly, buffers this capture is about to bake into its graph. \
                 Raise CANDLE_ALLOC_CACHE_MAX_MB and watch `evicted_at_demand_size`.",
                s.cached_bytes, s.capacity_bytes
            );
        }
        Ok((sizes, buffers))
    }

    /// `(bytes, buffers_available)` for the free pool. Diagnostic only.
    pub fn alloc_cache_free_counts(&self) -> Vec<(usize, usize)> {
        let cache = self.alloc_cache.lock().unwrap();
        let mut v: Vec<(usize, usize)> = cache
            .free
            .iter()
            .map(|(&b, sc)| (b, sc.ptrs.len()))
            .collect();
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
        // `take` counts hits/misses AND, while capturing, feeds `window` (the
        // demand profile) and `capture_misses` (the ledger). Keeping all of it
        // inside `AllocCache` is what makes it testable without a device.
        let hit = cache.take(bytes);
        // Keyed by size class, not byte count: the byte counts are unbounded
        // (they track KV length) and this set used to grow with them.
        if hit.is_none() && cache.missed.insert(bucket_index(bytes)) {
            if cache.capturing {
                // A miss DURING capture means this size was not pre-warmed: the
                // alloc becomes an unstable graph memory node -> launch fault or
                // corruption. Loud always (not gated) -- it is a correctness bug.
                //
                // Note this print is DEDUPED by size class, so it under-reports:
                // it is a log, not the gate. `capture_misses` / the
                // `capture_miss_count()` the caller refuses to instantiate on
                // is the counted, never-deduped ledger. Do not read a single
                // warning as "one miss".
                eprintln!(
                    "[alloc-cache] WARNING: MISS during capture, size {bytes} bytes \
                     (not pre-warmed) -> graph will be unstable. Grow warmup coverage."
                );
            } else if std::env::var_os("ARC_CACHE_DEBUG").is_some() {
                eprintln!("[alloc-cache] MISS new size class for {bytes} bytes -> cuMemAllocAsync");
            }
        }
        hit
    }

    /// Return a freed buffer to the cache. Returns false if caching is off (the
    /// caller must then free it normally). During capture the buffer is parked
    /// in `deferred` (not reusable until capture mode ends) to prevent
    /// within-capture aliasing.
    ///
    /// Accepting `ptr` may push retention over capacity, in which case the
    /// least-recently-used buffers are freed here — which is the only reason
    /// this cache ever hands memory back during steady-state decode. Returning
    /// `true` therefore means "`ptr` is the cache's problem now", not "nothing
    /// was freed".
    pub(crate) fn cache_put(&self, bytes: usize, ptr: cudarc::driver::sys::CUdeviceptr) -> bool {
        if bytes == 0 {
            return false;
        }
        let victims = {
            let mut cache = self.alloc_cache.lock().unwrap();
            if !cache.enabled {
                return false;
            }
            cache.put(bytes, ptr)
        };
        // Free outside the lock (upgrade -> Drop frees via cudarc).
        for p in victims {
            drop(unsafe { self.stream.upgrade_device_ptr::<u8>(p, 0) });
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

/// The caching allocator's bound, tested without a GPU.
///
/// `AllocCache` is pure host state — byte counts and `u64` pointer values — so
/// everything that makes it *bounded* can be asserted here. That matters: before
/// this module there were no tests of the allocator at all, in this repo or in
/// the one that consumes it, because every path that exercised it needed a
/// device.
///
/// These assert **counts**, not the absence of a crash. A cache that quietly
/// declined to cache anything would satisfy "did not OOM"; it would not satisfy
/// `hits == 3` or `frees() == 0 before the cap, > 0 after`.
#[cfg(test)]
mod alloc_cache_tests {
    use super::*;

    /// `ptr` values are opaque `u64`s to the cache — it never dereferences them
    /// — so a counter stands in for the driver.
    fn cache(capacity: usize) -> AllocCache {
        let mut c = AllocCache::default();
        c.enabled = true;
        c.capacity = capacity;
        c
    }

    const UNBOUNDED: usize = usize::MAX;

    #[test]
    fn a_put_then_a_take_of_the_same_size_is_a_hit() {
        let mut c = cache(UNBOUNDED);
        assert!(c.put(4096, 0x1000).is_empty());
        assert_eq!(c.take(4096), Some(0x1000));
        let s = c.stats();
        assert_eq!((s.hits, s.misses, s.puts), (1, 0, 1));
        assert_eq!(s.cached_bytes, 0, "the buffer left the cache");
    }

    /// The property the capacity bound rests on: a request is never served a
    /// buffer that was allocated at a different size. If this ever fails, either
    /// callers can overrun a short buffer, or `cached_bytes` stops matching the
    /// bytes actually held and the cap stops bounding anything.
    #[test]
    fn a_different_size_is_never_served() {
        let mut c = cache(UNBOUNDED);
        c.put(4096, 0x1000);
        assert_eq!(c.take(4097), None, "must not serve a larger request");
        assert_eq!(c.take(4095), None, "must not serve a smaller request");
        assert_eq!(c.take(8192), None);
        assert_eq!(c.stats().misses, 3);
        // The buffer is still there, untouched, for its own size.
        assert_eq!(c.take(4096), Some(0x1000));
    }

    /// The defect this change exists to fix, in miniature: sizes that step
    /// upward every call — as the KV-length-dependent buffers do — are never
    /// reused, so an unbounded cache retains all of them and frees nothing.
    #[test]
    fn unbounded_growing_sizes_retain_everything_and_free_nothing() {
        let mut c = cache(UNBOUNDED);
        let mut expect = 0usize;
        for step in 0..500u64 {
            let bytes = 65536 + step as usize * 8192; // the measured V4 pattern
            assert_eq!(c.take(bytes), None, "a never-seen size cannot hit");
            c.put(bytes, 0x1_0000 + step);
            expect += bytes;
        }
        let s = c.stats();
        assert_eq!(s.frees(), 0, "this is the leak: nothing is ever freed");
        assert_eq!(s.cached_bytes, expect);
        assert_eq!(s.cached_buffers, 500);
        assert_eq!(c.recount(), (s.cached_bytes, s.cached_buffers));
    }

    /// The fix. Same traffic, with a cap: retention stops at the cap and frees
    /// become non-zero. Both halves are asserted — "did not grow" alone would
    /// also be satisfied by a cache that never stored anything.
    #[test]
    fn a_capped_cache_bounds_retention_and_frees() {
        const CAP: usize = 8 * 1024 * 1024;
        let mut c = cache(CAP);
        let mut freed = 0u64;
        let mut largest = 0usize;
        for step in 0..500u64 {
            let bytes = 65536 + step as usize * 8192;
            largest = largest.max(bytes);
            c.take(bytes);
            freed += c.put(bytes, 0x1_0000 + step).len() as u64;
            assert!(
                c.stats().cached_bytes <= CAP,
                "retention passed the cap at step {step}: {} > {CAP}",
                c.stats().cached_bytes
            );
        }
        let s = c.stats();
        assert!(s.frees() > 0, "frees must be non-zero once the cap bites");
        assert_eq!(s.frees(), freed, "every victim was handed to the caller");
        assert_eq!(s.evicted, freed);
        // A buffer is taken in before the sweep that makes room for it, so the
        // peak overshoots the cap by at most one allocation and never more. Not
        // papered over: this is the number a VRAM budget has to leave headroom
        // for. Measured on V4 at a 1024 MiB cap, the peak was 1026.1 MiB.
        assert!(
            s.high_water_bytes <= CAP + largest,
            "peak {} exceeded cap {CAP} by more than one allocation ({largest})",
            s.high_water_bytes
        );
        assert!(s.high_water_bytes > CAP - largest, "the cap was actually reached");
        assert_eq!(c.recount(), (s.cached_bytes, s.cached_buffers));
    }

    /// Eviction must not evict the working set. A hot size touched every step
    /// keeps a fresh tick, so it survives while the one-shot sizes around it are
    /// reclaimed — otherwise the cap would cost the 7.80 ms/token the cache is
    /// worth. Measured on V4: a 1024 MiB cap holds a 0.9961 hit rate against
    /// 0.9996 unbounded.
    ///
    /// The cap has to leave room for the working set for this to hold. It is a
    /// cache, not a miracle: if one cold allocation is itself a large fraction
    /// of the cap, LRU will evict the hot buffer to make room and the hit rate
    /// collapses. `cold_max + HOT` here is a small fraction of the low-water
    /// mark, which is the regime a 1 GiB cap puts V4 in.
    #[test]
    fn eviction_keeps_the_hot_size_and_drops_the_cold_ones() {
        const HOT: usize = 1024 * 1024;
        const CAP: usize = 4 * 1024 * 1024;
        let mut c = cache(CAP);
        c.put(HOT, 0xAAAA);
        let mut cold_max = 0;
        for step in 0..400u64 {
            // The hot buffer is taken and returned every step, as a per-layer
            // activation would be.
            let hot = c.take(HOT).expect("the hot size must stay resident");
            c.put(HOT, hot);
            // A cold, never-repeated size, small next to the cap.
            let cold = 4096 + step as usize * 512;
            cold_max = cold;
            c.take(cold);
            c.put(cold, 0x2_0000 + step);
        }
        assert!(HOT + cold_max < CAP / 8 * 7, "fixture must fit under low water");
        assert_eq!(c.stats().hits, 400, "every hot take hit");
        assert!(c.stats().evicted > 0, "the cold sizes were reclaimed");
        assert!(c.stats().cached_bytes <= CAP);
    }

    /// Capacity 0 is a legitimate setting — cache nothing — and must not park a
    /// buffer it has no room for. Guards against an off-by-one that would retain
    /// one buffer per put forever.
    #[test]
    fn zero_capacity_retains_nothing() {
        let mut c = cache(0);
        let victims = c.put(4096, 0x1000);
        assert_eq!(victims, vec![0x1000]);
        assert_eq!(c.stats().cached_bytes, 0);
        assert_eq!(c.stats().cached_buffers, 0);
        assert_eq!(c.take(4096), None);
    }

    /// Draining hands back every buffer exactly once and zeroes the accounting.
    /// A drain that lost a pointer would leak it past the process's own
    /// bookkeeping, where nothing could ever find it again.
    #[test]
    fn drain_returns_every_buffer_once_and_zeroes_the_accounting() {
        let mut c = cache(UNBOUNDED);
        for i in 0..64u64 {
            c.put(4096 + (i as usize % 8) * 512, 0x3_0000 + i);
        }
        c.capturing = true;
        for i in 0..16u64 {
            c.put(2048, 0x4_0000 + i);
        }
        let mut drained = c.drain_all();
        assert_eq!(drained.len(), 80, "64 free + 16 deferred");
        drained.sort_unstable();
        drained.dedup();
        assert_eq!(drained.len(), 80, "no pointer handed back twice");
        let s = c.stats();
        assert_eq!((s.cached_bytes, s.cached_buffers), (0, 0));
        assert_eq!(s.drained, 80);
        assert_eq!(c.recount(), (0, 0));
    }

    /// During capture, frees are parked rather than recycled — re-serving an
    /// address inside one capture records two graph uses of it — but they still
    /// count against retention, and the cap must not evict them: those addresses
    /// may be baked into the graph being recorded.
    #[test]
    fn capture_parks_buffers_and_never_evicts_them() {
        let mut c = cache(64 * 1024);
        c.capturing = true;
        for i in 0..64u64 {
            let victims = c.put(65536, 0x5_0000 + i);
            assert!(victims.is_empty(), "capture must not evict");
        }
        let s = c.stats();
        assert_eq!(s.cached_buffers, 64);
        assert_eq!(s.cached_bytes, 64 * 65536);
        assert!(s.cached_bytes > s.capacity_bytes, "over the cap, deliberately");
        assert_eq!(s.evicted, 0);
        // Nothing parked is reusable until capture ends.
        assert_eq!(c.take(65536), None);
    }

    /// Sizes are the map's keys and they track KV length, so an emptied size
    /// must not leave its key behind — that is a host-side leak with the same
    /// unbounded shape as the device-side one.
    #[test]
    fn emptied_sizes_are_dropped_from_the_map() {
        let mut c = cache(1024 * 1024);
        for step in 0..2000u64 {
            let bytes = 4096 + step as usize * 512;
            c.put(bytes, 0x6_0000 + step);
        }
        assert!(
            c.free.len() < 600,
            "the size map grew unbounded: {} keys",
            c.free.len()
        );
        assert_eq!(c.stats().size_classes, c.free.len());
    }

    /// The `missed` diagnostic set is keyed by size class, not byte count, so it
    /// is bounded by the ladder (221 classes) no matter how many distinct sizes
    /// go through it.
    #[test]
    fn size_classes_are_monotonic_and_few() {
        let mut last = 0;
        for bytes in [0usize, 1, 511, 512, 513, 640, 768, 1023, 1024, 1 << 20] {
            let b = bucket_index(bytes);
            assert!(b >= last, "not monotonic at {bytes}");
            last = b;
        }
        assert_eq!(bucket_index(0), bucket_index(511), "tiny is one class");
        assert!(bucket_index(512) > bucket_index(511));
        // 512..=usize::MAX spans 4 classes per octave, plus the tiny class.
        let top = bucket_index(usize::MAX);
        assert!(top < 256, "ladder is small and fixed: {top}");
        // The measured V4 pattern — 2 000 distinct byte sizes stepping by 8 KiB
        // from 64 KiB — collapses to the octaves it spans. 65 536 is 2^16 and
        // the largest, 16 441 344, is under 2^24: eight octaves, four classes
        // each. This is the bound that keeps `missed` from growing with KV
        // length the way the byte-keyed version did.
        let sizes: Vec<usize> = (0..2000).map(|s| 65536 + s * 8192).collect();
        assert!(*sizes.last().unwrap() < (1 << 24));
        let classes: std::collections::HashSet<usize> =
            sizes.iter().map(|s| bucket_index(*s)).collect();
        assert_eq!(classes.len(), 8 * SUBCLASSES, "one class per octave-quarter");
    }

    // -----------------------------------------------------------------------
    // The merge: bounded allocator (LRU + capacity) meets the capture pre-warm
    // (demand profile + miss ledger). These are the interactions, not the
    // features — each side is already covered above and on its own line.
    // -----------------------------------------------------------------------

    /// A capture-mode window counts **allocations**, not failures: a hit weighs
    /// exactly as much as a miss. The pre-warm has to supply one buffer per
    /// allocation, so a profile built from misses alone would under-supply by
    /// however much the previous warm pass happened to cover.
    #[test]
    fn the_demand_window_counts_hits_and_misses_alike() {
        let mut c = cache(UNBOUNDED);
        c.put(4096, 0x1000); // one buffer available, so take #1 hits
        c.capturing = true;
        c.take(4096); // hit
        c.take(4096); // miss
        c.take(4096); // miss
        c.capturing = false;
        c.close_capture_window();
        assert_eq!(c.demand.get(&4096), Some(&3), "3 allocations, not 2 misses");
    }

    /// The profile is a union over windows with a per-size **max**, which is
    /// what covers a size cycle whose period exceeds one warm pass. A last-write
    /// -wins fold would silently forget the sizes only the earlier passes saw —
    /// and forgetting is precisely what makes a capture fault later.
    #[test]
    fn windows_union_by_max_not_by_overwrite() {
        let mut c = cache(UNBOUNDED);
        // Window 1: size A twice, size B once.
        c.capturing = true;
        c.take(1024);
        c.take(1024);
        c.take(2048);
        c.capturing = false;
        c.close_capture_window();
        // Window 2: size A once, size C once. A must NOT drop to 1, B must
        // survive not being seen at all.
        c.capturing = true;
        c.take(1024);
        c.take(4096);
        c.capturing = false;
        c.close_capture_window();
        assert_eq!(c.demand.get(&1024), Some(&2), "max over windows, not last");
        assert_eq!(c.demand.get(&2048), Some(&1), "a size seen once survives");
        assert_eq!(c.demand.get(&4096), Some(&1));
    }

    /// The ledger is scoped by `reset`, never by the window. The warm passes run
    /// with `capturing` set and miss on purpose; if those misses reached the
    /// gate it would refuse every graph. And within one capture the ledger is
    /// never deduped — three misses at one size read as three.
    #[test]
    fn the_miss_ledger_is_scoped_by_reset_and_never_deduped() {
        let mut c = cache(UNBOUNDED);
        // A warm pass: everything misses, by design.
        c.capturing = true;
        for _ in 0..5 {
            c.take(8192);
        }
        c.capturing = false;
        c.close_capture_window();
        assert_eq!(c.capture_misses.values().sum::<usize>(), 5);
        // The caller scopes the ledger to the real capture.
        c.capture_misses.clear();
        assert_eq!(c.capture_misses.values().sum::<usize>(), 0);
        c.capturing = true;
        c.take(8192);
        c.take(8192);
        c.take(9999);
        assert_eq!(
            c.capture_misses.get(&8192),
            Some(&2),
            "counted, not deduped"
        );
        assert_eq!(c.capture_misses.get(&9999), Some(&1));
        assert_eq!(c.stats().capture_misses, 3);
    }

    /// Misses outside a capture window never reach the ledger, so a decode step
    /// cannot inflate the gate's number.
    #[test]
    fn misses_outside_capture_do_not_reach_the_ledger() {
        let mut c = cache(UNBOUNDED);
        for i in 0..10u64 {
            c.take(4096 + i as usize);
        }
        assert_eq!(c.stats().misses, 10, "they are real driver allocations");
        assert_eq!(c.stats().capture_misses, 0, "but not capture misses");
    }

    /// The whole reason this merge is not mechanical. Pre-warming installs the
    /// demand profile; the cap says retention must come down. If the pre-warm
    /// evicted, LRU would reclaim the sizes it topped up earlier in the same
    /// pass — non-deterministically, since it walks a `HashMap` — and hand back
    /// a cache that provably cannot serve the capture it was built for.
    #[test]
    fn prewarm_installs_over_the_cap_without_evicting() {
        const CAP: usize = 64 * 1024;
        let mut c = cache(CAP);
        // A profile that does not fit: 8 sizes x 4 buffers x 64 KiB = 2 MiB.
        for i in 0..8usize {
            c.demand.insert(65536 + i * 8192, 4);
        }
        let plan = c.prewarm_plan(0);
        assert_eq!(plan.len(), 8);
        let mut installed = 0usize;
        let mut ptr = 0x10_0000u64;
        for (bytes, n) in plan {
            let ptrs: Vec<_> = (0..n)
                .map(|_| {
                    ptr += 1;
                    ptr
                })
                .collect();
            installed += ptrs.len();
            c.install_prewarmed(bytes, ptrs);
        }
        let s = c.stats();
        assert_eq!(installed, 32);
        assert_eq!(s.prewarmed, 32);
        assert_eq!(s.evicted, 0, "the pre-warm must not evict");
        assert_eq!(s.puts, 0, "a pre-warm is not a put");
        assert_eq!(s.misses, 0, "and not a miss");
        assert!(
            s.cached_bytes > s.capacity_bytes,
            "deliberately over the cap: {} <= {CAP}",
            s.cached_bytes
        );
        assert_eq!(c.recount(), (s.cached_bytes, s.cached_buffers));
        // Every profiled size can now be served exactly `demand` times.
        for i in 0..8usize {
            let bytes = 65536 + i * 8192;
            for k in 0..4 {
                assert!(c.take(bytes).is_some(), "size {bytes} short at {k}");
            }
        }
        assert_eq!(c.stats().hits, 32);
    }

    /// And the other half of that bargain: the cap is re-imposed by the first
    /// ordinary put after the window, exactly as `set_capture_mode` already
    /// relies on. The overshoot is bounded, not permanent.
    #[test]
    fn the_cap_is_reimposed_by_the_first_ordinary_put_after_prewarm() {
        const CAP: usize = 64 * 1024;
        let mut c = cache(CAP);
        c.install_prewarmed(65536, (0..32u64).map(|i| 0x20_0000 + i).collect());
        assert!(c.stats().cached_bytes > CAP);
        let victims = c.put(4096, 0x30_0000);
        assert!(!victims.is_empty(), "the sweep ran");
        assert!(c.stats().cached_bytes <= CAP, "retention bounded again");
    }

    /// The instrument for the hazard this merge surfaces but does not fix:
    /// eviction cannot see which addresses an instantiated graph baked, and the
    /// pre-warmed profile is exactly that set. Non-zero here while a graph can
    /// replay is a use-after-free waiting to happen; the counter is what lets a
    /// GPU run assert on it instead of waiting for the crash.
    #[test]
    fn eviction_at_a_profiled_size_is_counted_separately() {
        const CAP: usize = 64 * 1024;
        let mut c = cache(CAP);
        c.demand.insert(65536, 8);
        c.install_prewarmed(65536, (0..8u64).map(|i| 0x40_0000 + i).collect());
        assert_eq!(c.stats().evicted_at_demand_size, 0);
        c.put(4096, 0x50_0000);
        let s = c.stats();
        assert!(s.evicted > 0);
        assert_eq!(
            s.evicted_at_demand_size, s.evicted,
            "every victim came from the profiled size"
        );
    }

    /// A drain discards the open window (it describes a pool that is going
    /// away) but keeps the profile (a fact about the model's shapes) and keeps
    /// the ledger (the caller's to scope). Getting any of the three wrong is
    /// silent: the capture still runs, it just gets warmed from the wrong facts.
    #[test]
    fn a_drain_discards_the_window_but_keeps_the_profile_and_the_ledger() {
        let mut c = cache(UNBOUNDED);
        c.capturing = true;
        c.take(1024); // window + ledger entry
        c.close_capture_window(); // profile now knows 1024
        c.capturing = true;
        c.take(2048); // an OPEN window, mid-capture
        c.put(4096, 0x60_0000);
        let drained = c.drain_all();
        assert_eq!(drained.len(), 1);
        assert!(c.window.is_empty(), "the open window is discarded");
        assert_eq!(c.demand.get(&1024), Some(&1), "the profile survives");
        assert_eq!(
            c.capture_misses.values().sum::<usize>(),
            2,
            "the ledger survives; only reset_capture_misses clears it"
        );
        assert!(!c.capturing, "drain leaves capture mode");
        assert_eq!(c.recount(), (0, 0));
    }

    /// A pre-warm that finds the pool already stocked allocates nothing. Guards
    /// the `have >= want` branch: getting it wrong would re-allocate the whole
    /// profile on every capture and grow retention without bound — the exact
    /// defect the cap exists to stop, reintroduced through the other door.
    #[test]
    fn prewarm_is_idempotent_when_the_pool_already_covers_the_profile() {
        let mut c = cache(UNBOUNDED);
        c.demand.insert(4096, 3);
        let first = c.prewarm_plan(1);
        assert_eq!(first, vec![(4096, 4)], "demand 3 + slack 1");
        c.install_prewarmed(4096, vec![1, 2, 3, 4]);
        assert!(
            c.prewarm_plan(1).is_empty(),
            "a second pre-warm must allocate nothing"
        );
        assert_eq!(c.stats().prewarmed, 4);
    }
}
