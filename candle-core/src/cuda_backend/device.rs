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
    /// Bytes currently retained (`free` + `deferred`).
    pub cached_bytes: usize,
    /// Buffers currently retained.
    pub cached_buffers: usize,
    /// Largest `cached_bytes` ever reached. The number that says whether the
    /// capacity is doing anything.
    pub high_water_bytes: usize,
    /// The cap. `usize::MAX` means unbounded.
    pub capacity_bytes: usize,
    /// Distinct (size class, physical size) groups currently held.
    pub size_classes: usize,
}

impl AllocCacheStats {
    /// Buffers returned to the driver, by any route. "Frees per step".
    pub fn frees(&self) -> u64 {
        self.evicted + self.drained
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
    missed: std::collections::HashSet<usize>,
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
            cached_bytes: 0,
            cached_buffers: 0,
            capacity,
            tick: 0,
            hits: 0,
            misses: 0,
            puts: 0,
            evicted: 0,
            drained: 0,
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
    fn take(&mut self, bytes: usize) -> Option<cudarc::driver::sys::CUdeviceptr> {
        self.tick += 1;
        let tick = self.tick;
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
        for (_, bytes) in order {
            if self.cached_bytes <= low_water {
                break;
            }
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
                    }
                    None => break,
                }
            }
        }
        // Drop emptied sizes so the map does not grow without bound either — it
        // is keyed by byte count, and those track KV length.
        self.free.retain(|_, sc| !sc.ptrs.is_empty());
        victims
    }

    /// Hand every retained buffer back. Used by `set_alloc_cache_enabled(false)`
    /// and `drain_alloc_cache_and_free()`.
    fn drain_all(&mut self) -> Vec<cudarc::driver::sys::CUdeviceptr> {
        self.capturing = false;
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

    fn stats(&self) -> AllocCacheStats {
        AllocCacheStats {
            hits: self.hits,
            misses: self.misses,
            puts: self.puts,
            evicted: self.evicted,
            drained: self.drained,
            cached_bytes: self.cached_bytes,
            cached_buffers: self.cached_buffers,
            high_water_bytes: self.high_water_bytes,
            capacity_bytes: self.capacity,
            size_classes: self.free.values().filter(|sc| !sc.ptrs.is_empty()).count(),
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
        let mut cache = self.alloc_cache.lock().unwrap();
        cache.capturing = capturing;
        if !capturing {
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
        let hit = cache.take(bytes);
        // Keyed by size class, not byte count: the byte counts are unbounded
        // (they track KV length) and this set used to grow with them.
        if hit.is_none() && cache.missed.insert(bucket_index(bytes)) {
            if cache.capturing {
                // A miss DURING capture means this size was not pre-warmed: the
                // alloc becomes an unstable graph memory node -> launch fault or
                // corruption. Loud always (not gated) -- it is a correctness bug.
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
            let leaked: &'static [T] = Vec::leak(src.to_vec());
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
}
