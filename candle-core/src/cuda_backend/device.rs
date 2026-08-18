use crate::backend::{BackendDevice, BackendStorage};
use crate::{CpuStorage, CpuStorageRef, DType, Layout, Result, Shape};
pub use candle_kernels as kernels;
pub use cudarc;
use cudarc::driver::CudaFunction;
use float8::F8E4M3;
use half::{bf16, f16};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

use super::{CudaError, CudaStorage, CudaStorageSlice, WrapErr};

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
    missed: std::collections::HashSet<usize>,
    /// Hard cap on bytes parked in `free` + `deferred`. 0 = unlimited. A put
    /// that would exceed it is refused (the caller frees normally), so the
    /// arena's device footprint is bounded by construction rather than by hope.
    /// Set via `set_alloc_cache_cap_bytes`.
    cap_bytes: u64,
    /// Content-addressed layout-info table: dims+strides bytes -> a device
    /// buffer holding exactly those bytes, owned for the process lifetime.
    ///
    /// SAFETY / aliasing: every `info` parameter in candle-kernels is declared
    /// `const size_t *` (checked: zero non-const `size_t *info` in the kernel
    /// sources), so these buffers are read-only to the device. Two ops with
    /// byte-identical dims/strides therefore cannot observe a difference
    /// between sharing one buffer and holding two copies of the same bytes.
    /// Interning is also strictly SAFER than the previous path, which freed the
    /// buffer at `InfoBuf::drop` — i.e. potentially before the kernel that
    /// reads it has retired, relying entirely on stream ordering.
    intern: HashMap<Box<[usize]>, cudarc::driver::sys::CUdeviceptr>,
    intern_bytes: u64,
    /// 0 = interning off (upstream behaviour: one alloc + one H2D per op).
    intern_max_entries: usize,
    stats: AllocStats,
}

/// Counters for the caching allocator and the layout-info intern table.
///
/// These exist to answer one question with a number instead of an opinion:
/// **how many `cuMemAllocAsync` and how many dims/strides H2D copies does one
/// decode step actually issue?** `driver_allocs` counts the calls an nsys trace
/// would show, so the counter and the trace are two independent instruments
/// that must agree.
#[derive(Default, Clone, Copy, Debug, PartialEq, Eq)]
pub struct AllocStats {
    /// `cuMemAllocAsync` actually issued (cache miss, or cache disabled).
    pub driver_allocs: u64,
    pub driver_alloc_bytes: u64,
    /// Allocations served from the free list — no driver call.
    pub cache_hits: u64,
    /// Buffers accepted back into the cache.
    pub cache_puts: u64,
    /// Puts refused because the cap was reached; the caller freed instead.
    pub cache_puts_refused: u64,
    /// `cuMemFreeAsync` actually issued (cache off, put refused, or drain).
    pub driver_frees: u64,
    /// dims/strides H2D copies actually issued.
    pub info_uploads: u64,
    /// dims/strides H2D copies elided by the intern table.
    pub info_hits: u64,
    /// H2D copies of real tensor data (`clone_htod` from a CpuStorage). Kept
    /// separate so the trace's total H2D count can be attributed rather than
    /// guessed at.
    pub data_htod: u64,
    /// Bytes currently parked in `free` + `deferred`, maintained incrementally.
    pub cached_bytes: u64,
    /// High-water mark of `cached_bytes` — the arena's peak device footprint.
    pub cached_bytes_hwm: u64,
    pub intern_entries: u64,
    pub intern_bytes: u64,
}

/// Result of [`CudaDevice::verify_alloc_accounting`].
///
/// D33: an instrument that cannot report the bad answer is not an instrument.
/// `cached_bytes` is maintained incrementally on every put/take; `recomputed`
/// walks the free list and the deferred list and adds the sizes up from
/// scratch. If a put, a take, a drain or a cap refusal ever fails to update the
/// running total, the two disagree. They cannot both be right and disagree.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AllocAccounting {
    pub running: u64,
    pub recomputed: u64,
    pub buffers: u64,
}

/// Round an allocation up to a reuse bucket.
///
/// # Why bucketing is not optional
///
/// The free list is keyed by size. A decode step whose tensors depend on the
/// KV length asks for a slightly *different* size every step, so an
/// exact-size cache misses every step (allocations never reach zero) **and**
/// retains one buffer per distinct length — a leak that grows with context.
/// Rounding to a bucket collapses that family onto a bounded set of keys.
///
/// # The correctness constraint
///
/// `bucket(n) >= n` for all `n`, and a buffer is physically allocated at its
/// bucket size on a miss (see `alloc`). So any request served from bucket `k`
/// asks for at most `k` bytes and the buffer physically holds `k`. Allocating
/// the *requested* size while filing it under a larger key would hand out a
/// short buffer — a silent out-of-bounds write, not a crash.
///
/// Granularity is an eighth of the enclosing octave: at most 12.5% waste, and
/// eight buckets per octave.
fn bucket(bytes: usize) -> usize {
    const MIN: usize = 128;
    if bytes == 0 {
        return 0;
    }
    if bytes <= MIN {
        return MIN;
    }
    let floor_pow2 = 1usize << (usize::BITS - 1 - bytes.leading_zeros());
    let gran = (floor_pow2 >> 3).max(MIN);
    bytes.div_ceil(gran) * gran
}

#[cfg(test)]
mod bucket_tests {
    use super::bucket;

    #[test]
    fn bucket_never_shrinks_a_request() {
        // The safety property the whole scheme rests on.
        for n in 1usize..100_000 {
            assert!(bucket(n) >= n, "bucket({n}) = {} < {n}", bucket(n));
        }
        for shift in 10..40 {
            for delta in [-3i64, -1, 0, 1, 3] {
                let n = ((1i64 << shift) + delta) as usize;
                assert!(bucket(n) >= n, "bucket({n}) = {} < {n}", bucket(n));
            }
        }
    }

    #[test]
    fn bucket_is_idempotent() {
        // put() and take() both round; if rounding were not idempotent a
        // buffer could be filed under a key larger than its physical size.
        for n in [
            1usize,
            127,
            128,
            129,
            1024,
            1025,
            4096,
            1 << 20,
            (1 << 20) + 7,
        ] {
            assert_eq!(bucket(bucket(n)), bucket(n), "not idempotent at {n}");
        }
    }

    #[test]
    fn bucket_waste_is_bounded() {
        for n in 129usize..200_000 {
            let b = bucket(n);
            assert!(
                b <= n + n / 8 + 128,
                "bucket({n}) = {b} wastes more than 12.5%"
            );
        }
    }

    #[test]
    fn bucket_collapses_a_growing_kv_family() {
        // The motivating case: a tensor that grows by one KV slot per step must
        // not produce a new key every step.
        let keys: std::collections::HashSet<usize> =
            (0..4096).map(|step| bucket(65536 + step * 128)).collect();
        assert!(
            keys.len() < 64,
            "{} distinct buckets for 4096 KV lengths -- still a per-step leak",
            keys.len()
        );
    }
}

impl AllocAccounting {
    pub fn agrees(&self) -> bool {
        self.running == self.recomputed
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
                cache.capturing = false;
                let mut d: Vec<_> = cache.free.drain().flat_map(|(_, v)| v).collect();
                d.extend(cache.deferred.drain(..).map(|(_, p)| p));
                cache.stats.cached_bytes = 0;
                cache.stats.driver_frees += d.len() as u64;
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

    /// Bound the arena's device footprint. `cap_bytes == 0` means unlimited
    /// (the previous behaviour). Once `free` + `deferred` hold `cap_bytes`, a
    /// further put is refused and the caller frees normally — so the arena can
    /// never grow past a number you chose, and `AllocStats::cache_puts_refused`
    /// tells you if the cap is actually binding rather than leaving you to
    /// wonder.
    pub fn set_alloc_cache_cap_bytes(&self, cap_bytes: u64) {
        self.alloc_cache.lock().unwrap().cap_bytes = cap_bytes;
    }

    /// Enable content-addressed interning of dims/strides uploads.
    ///
    /// `max_entries == 0` disables it (upstream behaviour: one allocation and
    /// one `cuMemcpyHtoDAsync` per op that needs layout metadata). Otherwise a
    /// layout whose bytes have been uploaded before is served from the table
    /// with **no allocation and no copy**. Entries are owned for the process
    /// lifetime; the table is capped because an unbounded map keyed on tensor
    /// shapes is a leak on a workload with unbounded shape variety.
    ///
    /// A decode step at fixed batch and fixed head count issues the same
    /// handful of distinct layouts every step, which is exactly the case this
    /// serves. `AllocStats::intern_entries` reports how many distinct layouts
    /// were actually seen, so "the table is small" is a measurement, not a
    /// premise.
    pub fn set_info_intern_max_entries(&self, max_entries: usize) {
        self.alloc_cache.lock().unwrap().intern_max_entries = max_entries;
    }

    /// Snapshot the allocator/H2D counters.
    pub fn alloc_stats(&self) -> AllocStats {
        let cache = self.alloc_cache.lock().unwrap();
        let mut s = cache.stats;
        s.intern_entries = cache.intern.len() as u64;
        s.intern_bytes = cache.intern_bytes;
        s
    }

    /// Zero the counters (not the pools). Call between benchmark legs so a
    /// per-step rate is measured over a known number of steps.
    pub fn reset_alloc_stats(&self) {
        let mut cache = self.alloc_cache.lock().unwrap();
        let hwm = cache.stats.cached_bytes_hwm;
        let cached = cache.stats.cached_bytes;
        cache.stats = AllocStats::default();
        // Footprint is a level, not a rate: carry it across a reset.
        cache.stats.cached_bytes = cached;
        cache.stats.cached_bytes_hwm = hwm;
    }

    /// Recompute the cached-byte total from the pools and return it alongside
    /// the incrementally maintained one. See [`AllocAccounting`] — these two
    /// numbers are produced by disjoint code paths and cannot both be right and
    /// disagree, which is the point.
    pub fn verify_alloc_accounting(&self) -> AllocAccounting {
        let cache = self.alloc_cache.lock().unwrap();
        let mut recomputed = 0u64;
        let mut buffers = 0u64;
        for (bytes, ptrs) in cache.free.iter() {
            recomputed += (*bytes as u64) * (ptrs.len() as u64);
            buffers += ptrs.len() as u64;
        }
        for (bytes, _) in cache.deferred.iter() {
            recomputed += *bytes as u64;
            buffers += 1;
        }
        AllocAccounting {
            running: cache.stats.cached_bytes,
            recomputed,
            buffers,
        }
    }

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
            let mut d: Vec<_> = cache.free.drain().flat_map(|(_, v)| v).collect();
            d.extend(cache.deferred.drain(..).map(|(_, p)| p));
            // Sizes that missed before mean nothing once every buffer is gone.
            cache.missed.clear();
            // The pools are empty, so the running byte total must say so --
            // otherwise `verify_alloc_accounting` reports a mismatch that is
            // this function's bookkeeping rather than a real one.
            cache.stats.cached_bytes = 0;
            cache.stats.driver_frees += d.len() as u64;
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
        if !capturing {
            let deferred = std::mem::take(&mut cache.deferred);
            for (bytes, ptr) in deferred {
                cache.free.entry(bytes).or_default().push(ptr);
            }
        }
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
        let bytes = bucket(bytes);
        let hit = cache.free.get_mut(&bytes).and_then(|v| v.pop());
        if hit.is_some() {
            cache.stats.cache_hits += 1;
            cache.stats.cached_bytes = cache.stats.cached_bytes.saturating_sub(bytes as u64);
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
        let bytes = bucket(bytes);
        // Bounded by construction: refuse rather than grow past the cap. The
        // caller then frees normally, so the arena's footprint is the cap and
        // `cache_puts_refused` says whether the cap ever bound.
        if cache.cap_bytes != 0 && cache.stats.cached_bytes + bytes as u64 > cache.cap_bytes {
            cache.stats.cache_puts_refused += 1;
            return false;
        }
        if cache.capturing {
            cache.deferred.push((bytes, ptr));
        } else {
            cache.free.entry(bytes).or_default().push(ptr);
        }
        cache.stats.cache_puts += 1;
        cache.stats.cached_bytes += bytes as u64;
        if cache.stats.cached_bytes > cache.stats.cached_bytes_hwm {
            cache.stats.cached_bytes_hwm = cache.stats.cached_bytes;
        }
        true
    }

    /// Look up an already-uploaded dims/strides buffer by its exact contents.
    fn info_intern_get(&self, key: &[usize]) -> Option<cudarc::driver::sys::CUdeviceptr> {
        let cache = self.alloc_cache.lock().unwrap();
        if cache.intern_max_entries == 0 {
            return None;
        }
        cache.intern.get(key).copied()
    }

    /// Record a freshly uploaded dims/strides buffer. Returns false if the table
    /// is full or interning is off, in which case the caller keeps ownership and
    /// frees/caches the buffer the normal way.
    fn info_intern_put(&self, key: &[usize], ptr: cudarc::driver::sys::CUdeviceptr) -> bool {
        let mut cache = self.alloc_cache.lock().unwrap();
        if cache.intern_max_entries == 0 || cache.intern.len() >= cache.intern_max_entries {
            return false;
        }
        // Losing a race must not orphan a buffer. Overwriting the entry would
        // make the winner's pointer unreachable -- never freed, never reused --
        // so if the layout is already interned we decline and the caller keeps
        // its own buffer, which its Drop returns to the arena. Both buffers
        // hold the same bytes, so which one the kernel reads is immaterial.
        if cache.intern.contains_key(key) {
            return false;
        }
        cache.intern.insert(key.into(), ptr);
        cache.intern_bytes += (key.len() * std::mem::size_of::<usize>()) as u64;
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
        if let Some(ptr) = self.alloc_bucketed(bytes)? {
            return Ok(unsafe { self.stream.upgrade_device_ptr::<T>(ptr, len) });
        }
        self.count_driver_alloc(bytes);
        self.stream.alloc::<T>(len).w()
    }

    /// Cache-miss allocation while the arena is enabled: allocate the BUCKET
    /// size, not the requested size, so that when this buffer is later filed
    /// under its bucket key it genuinely holds that many bytes. Returns `None`
    /// when the arena is off, in which case the caller allocates exactly the
    /// requested size (byte-identical to upstream candle).
    fn alloc_bucketed(&self, bytes: usize) -> Result<Option<cudarc::driver::sys::CUdeviceptr>> {
        if bytes == 0 || !self.alloc_cache_enabled() {
            return Ok(None);
        }
        let b = bucket(bytes);
        self.count_driver_alloc(b);
        // SAFETY: same contract as `CudaDevice::alloc` -- the bytes are
        // uninitialised and every caller writes before it reads. Allocated as
        // `u8` at the BUCKET size so the buffer physically holds what its
        // free-list key claims.
        let raw = unsafe { self.stream.alloc::<u8>(b) }.w()?;
        Ok(Some(raw.leak()))
    }

    /// One `cuMemAllocAsync` is about to be issued. This counter is the whole
    /// point of the instrumentation: it counts exactly the call an nsys trace
    /// records, so counter and trace are two independent instruments that must
    /// agree — and it is incremented on the MISS path, which means it reports
    /// the large number when the arena is off. An instrument that can only
    /// report zero is not evidence.
    fn count_driver_alloc(&self, bytes: usize) {
        let mut cache = self.alloc_cache.lock().unwrap();
        cache.stats.driver_allocs += 1;
        cache.stats.driver_alloc_bytes += bytes as u64;
    }

    pub(crate) fn count_driver_free(&self) {
        self.alloc_cache.lock().unwrap().stats.driver_frees += 1;
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
        if let Some(ptr) = self.alloc_bucketed(bytes)? {
            let mut slice = unsafe { self.stream.upgrade_device_ptr::<T>(ptr, len) };
            self.stream.memset_zeros(&mut slice).w()?;
            return Ok(slice);
        }
        self.count_driver_alloc(bytes);
        self.stream.alloc_zeros::<T>(len).w()
    }

    pub fn memcpy_htod<
        T: cudarc::driver::DeviceRepr,
        Src: cudarc::driver::HostSlice<T> + ?Sized,
        Dst: cudarc::driver::DevicePtrMut<T>,
    >(
        &self,
        src: &Src,
        dst: &mut Dst,
    ) -> Result<()> {
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

    pub fn clone_htod<T: cudarc::driver::DeviceRepr, Src: cudarc::driver::HostSlice<T> + ?Sized>(
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
        let len = cudarc::driver::HostSlice::len(src);
        let mut dst = unsafe { self.alloc::<T>(len)? };
        self.alloc_cache.lock().unwrap().stats.data_htod += 1;
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
    ///
    /// # Interning (Arc budget chain: 2,811 H2D copies per decode token)
    ///
    /// A B=1 decode step re-uploads the *same* dims/strides bytes every step —
    /// the shapes do not change. Measured on the V4 H200 trace: 2,509 of the
    /// 2,811 H2D copies per token are `usize` layout arrays averaging 72 B.
    /// With `set_info_intern_max_entries(n)` a layout whose bytes were uploaded
    /// before is served from a device buffer that already holds them: **no
    /// allocation and no copy**.
    ///
    /// Safety of sharing one buffer between ops: every `info` parameter in
    /// candle-kernels is `const size_t *` (verified: no non-const `size_t *info`
    /// exists in the kernel sources), so the buffer is read-only to the device
    /// and two ops with byte-identical layouts cannot tell a shared buffer from
    /// two private ones. It is also strictly safer than the non-interned path,
    /// which hands the buffer back at `InfoBuf::drop` — i.e. possibly before the
    /// kernel reading it has retired.
    pub fn htod_info<T: cudarc::driver::DeviceRepr + Copy + 'static>(
        &self,
        src: &[T],
    ) -> Result<super::InfoBuf<T>> {
        // Only `usize` layout arrays are interned. Restricting by TypeId keeps
        // the byte-level key sound: `usize` has no padding, so equal bytes mean
        // equal values, which is not guaranteed for an arbitrary `DeviceRepr`.
        let internable = std::any::TypeId::of::<T>() == std::any::TypeId::of::<usize>();
        if internable {
            // SAFETY: guarded by the TypeId check above, so T is exactly usize.
            let key: &[usize] =
                unsafe { std::slice::from_raw_parts(src.as_ptr().cast::<usize>(), src.len()) };
            if let Some(ptr) = self.info_intern_get(key) {
                let mut cache = self.alloc_cache.lock().unwrap();
                cache.stats.info_hits += 1;
                drop(cache);
                let slice = unsafe { self.stream.upgrade_device_ptr::<T>(ptr, src.len()) };
                return Ok(super::InfoBuf::interned(self.clone(), slice));
            }
        }
        let mut dst = unsafe { self.alloc::<T>(src.len())? };
        self.alloc_cache.lock().unwrap().stats.info_uploads += 1;
        if self.capture_mode() {
            let leaked: &'static [T] = Vec::leak(src.to_vec());
            self.stream.memcpy_htod(leaked, &mut dst).w()?;
        } else {
            self.stream.memcpy_htod(src, &mut dst).w()?;
        }
        if internable {
            // SAFETY: as above, T is exactly usize.
            let key: &[usize] =
                unsafe { std::slice::from_raw_parts(src.as_ptr().cast::<usize>(), src.len()) };
            // `leak` consumes the slice without freeing, so the table can own
            // the buffer for the process lifetime.
            let len = src.len();
            let ptr = dst.leak();
            let slice = unsafe { self.stream.upgrade_device_ptr::<T>(ptr, len) };
            return if self.info_intern_put(key, ptr) {
                Ok(super::InfoBuf::interned(self.clone(), slice))
            } else {
                // Table full: hand ownership back so it frees/caches normally.
                Ok(super::InfoBuf::new(self.clone(), slice))
            };
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
