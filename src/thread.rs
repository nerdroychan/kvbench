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
    fn join(self: Box<Self>);
}

pub trait Thread {
    fn spawn(&self, f: Box<dyn FnOnce() + Send>) -> Box<dyn JoinHandle>;

    fn yield_now(&self);

    fn pin(&self, core: usize);
}

#[derive(Clone)]
pub(crate) struct DefaultThread;

pub(crate) struct DefaultJoinHandle(std::thread::JoinHandle<()>);

impl JoinHandle for DefaultJoinHandle {
    fn join(self: Box<Self>) {
        let handle = self.0;
        assert!(handle.join().is_ok());
    }
}

impl Thread for DefaultThread {
    fn spawn(&self, f: Box<dyn FnOnce() + Send>) -> Box<dyn JoinHandle> {
        let handle = std::thread::spawn(f);
        Box::new(DefaultJoinHandle(handle))
    }

    fn yield_now(&self) {
        std::thread::yield_now();
    }

    fn pin(&self, core: usize) {
        let cores = core_affinity::get_core_ids().unwrap();
        core_affinity::set_for_current(cores[core % cores.len()]);
    }
}
