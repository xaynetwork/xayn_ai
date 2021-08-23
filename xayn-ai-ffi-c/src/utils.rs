//! Utilities.

use std::{ffi::CStr, fmt::Display, sync::Once};

use rayon::{ThreadPoolBuildError, ThreadPoolBuilder};
use xayn_ai_ffi::{CCode, Error};

use crate::result::{call_with_result, error::CError};
#[cfg(doc)]
pub use crate::slice::CBoxedSlice;

/// Reads a string slice from the borrowed bytes pointer.
///
/// # Errors
/// Fails on null pointer and invalid utf8 encoding. The error is constructed from the `code`
/// and `context`.
///
/// # Safety
/// The behavior is undefined if:
/// - A non-null `bytes` doesn't point to an aligned, contiguous area of memory with a terminating
/// null byte.
pub(crate) unsafe fn as_str<'a>(
    bytes: Option<&'a u8>,
    code: CCode,
    context: impl Display,
) -> Result<&'a str, Error> {
    let pointer = bytes
        .ok_or_else(|| code.with_context(format!("{}: The {} is null", context, code)))?
        as *const u8;
    unsafe { CStr::from_ptr::<'a>(pointer.cast()) }
        .to_str()
        .map_err(|cause| {
            code.with_context(format!(
                "{}: The {} contains invalid utf8: {}",
                context, code, cause,
            ))
        })
}

/// Conversion of Rust values into C-compatible values.
///
/// # Safety
/// The behavior is undefined if:
/// - The `Value` is not compatible with the C ABI.
/// - The `Value` is accessed after its lifetime has expired.
pub(crate) unsafe trait IntoRaw {
    /// A C-compatible value. Usually some kind of `#[repr(C)]` and `'static`/owned.
    type Value: Default + Send + Sized;

    /// Converts the Rust value into the C value. Usually leaks memory for heap allocated values.
    fn into_raw(self) -> Self::Value;
}

unsafe impl IntoRaw for () {
    // Safety: This is a no-op.
    type Value = ();

    #[inline]
    fn into_raw(self) -> Self::Value {}
}

/// Initializes the global thread pool. The thread pool is used by the AI to
/// parallel some of its tasks.
///
/// The number of threads used by the pool depends on `num_cpus`:
///
/// On a single core system the thread pool consists of only one thread.
/// On a multicore system the thread pool consists of (the number of logical cores - 1) threads.
///
/// # Error
/// - If the initialization of the thread pool has failed.
/// - An unexpected panic happened during the initialization of the thread pool.
///
/// # Safety
///
/// It is safe to call this function multiple times but it must be invoked
/// before calling any of the `xaynai_*` functions.
///
/// The behavior is undefined if:
/// - A non-null `error` doesn't point to an aligned, contiguous area of memory with a [`CError`].
#[no_mangle]
pub unsafe extern "C" fn xaynai_init_thread_pool(num_cpus: u64, error: Option<&mut CError>) {
    static mut INIT_ERROR: Result<(), ThreadPoolBuildError> = Ok(());
    static INIT: Once = Once::new();

    let init_pool = || unsafe {
        INIT.call_once(|| {
            INIT_ERROR = init_thread_pool(num_cpus as usize);
        });

        if let Err(cause) = INIT_ERROR.as_ref() {
            Err(CCode::InitGlobalThreadPool
                .with_context(format!("Failed to initialize thread pool: {}", cause)))
        } else {
            Ok(())
        }
    };

    call_with_result(init_pool, error)
}

/// See [`xaynai_init_thread_pool()`] for more.
fn init_thread_pool(num_cpus: usize) -> Result<(), ThreadPoolBuildError> {
    let num_threads = if num_cpus > 1 { num_cpus - 1 } else { num_cpus };

    ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Assuming the reference points to the first byte in a CStr converts it into a &str.
    ///
    /// # Panics
    ///
    /// Panics if `None` is passed in or the if it's not in a  valid utf8 encoding.
    ///
    /// # Safety
    ///
    /// The behavior is undefined if:
    ///
    /// - The byte reference passed in is not the beginning of a null terminated
    ///   c-string.
    ///
    /// As we accept a `&u8` we already have the guarantees that the pointer is
    /// not dangling as else the creation of the `Option<&u8>` would have been
    /// invalid.
    pub unsafe fn as_str_unchecked<'a>(bytes: Option<&'a u8>) -> &'a str {
        unsafe { CStr::from_ptr::<'a>((bytes.unwrap() as *const u8).cast()) }
            .to_str()
            .unwrap()
    }

    /// Nullable pointer conversions.
    pub trait AsPtr {
        /// Casts as a borrowed pointer.
        #[inline]
        fn as_ptr(&self) -> Option<&Self> {
            Some(self)
        }

        /// Casts as a mutable borrowed pointer.
        #[inline]
        fn as_mut_ptr(&mut self) -> Option<&mut Self> {
            Some(self)
        }

        /// Casts into an owned pointer.
        #[inline]
        fn into_ptr(self: Box<Self>) -> Option<Box<Self>> {
            Some(self)
        }
    }
}
