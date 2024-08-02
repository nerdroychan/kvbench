//! Spawn-join functionality.
//!
//! **You may not need to check this if it is OK to run benchmarks with [`std::thread`].**
//!
//! A key-value store is generally passive. However, some store may act like a server with
//! active threads. In that case, one may employ its own implementation of spawn-join. If that is
//! the case, their join handle (like [`std::thread::JoinHandle`]) should implement the
//! [`JoinHandle`] trait and the spawn struct needs to implement [`Thread`].
//!
//! Note that for simplicity, the function spawn is generic over should not have a return value. So
//! it is with the [`JoinHandle`]. Because the purpose is not general spawn-join but solely for
//! benchmark code, which does not use any return values.

pub trait JoinHandle {
    fn join(self);
}

pub trait Thread: Send + Clone + 'static {
    type JoinHandle: JoinHandle;

    fn spawn(&self, f: impl FnOnce() + Send + 'static) -> Self::JoinHandle;

    fn yield_now(&self);

    fn pin(&self, core: usize);
}

#[derive(Clone)]
pub(crate) struct DefaultThread;

impl JoinHandle for std::thread::JoinHandle<()> {
    fn join(self) {
        let _ = self.join();
    }
}

impl Thread for DefaultThread {
    type JoinHandle = std::thread::JoinHandle<()>;

    fn spawn(&self, f: impl FnOnce() + Send + 'static) -> Self::JoinHandle {
        std::thread::spawn(f)
    }

    fn yield_now(&self) {
        std::thread::yield_now();
    }

    fn pin(&self, core: usize) {
        let cores = core_affinity::get_core_ids().unwrap();
        core_affinity::set_for_current(cores[core % cores.len()]);
    }
}
