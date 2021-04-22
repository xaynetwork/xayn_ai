use std::{
    marker::PhantomData,
    ptr::{null, null_mut},
    slice::{from_raw_parts, from_raw_parts_mut},
};

use ffi_support::{ExternError, IntoFfi};

use crate::result::call_with_result;

/// A bytes buffer.
pub struct Bytes(pub(crate) Vec<u8>);

/// A raw slice of bytes.
#[repr(C)]
pub struct CBytes<'a> {
    /// The raw pointer to the bytes.
    pub data: *const u8,
    /// The number of bytes.
    pub len: u32,
    // lifetime attached to the foreign raw slice of bytes (only use 'static for 'a if it's owned)
    _lifetime: PhantomData<&'a [u8]>,
}

unsafe impl IntoFfi for Bytes {
    type Value = *mut CBytes<'static>;

    #[inline]
    fn ffi_default() -> Self::Value {
        null_mut()
    }

    #[inline]
    fn into_ffi_value(self) -> Self::Value {
        let len = self.0.len() as u32;
        let data = if self.0.is_empty() {
            null()
        } else {
            self.0.leak().as_ptr()
        };
        let bytes = CBytes {
            data,
            len,
            _lifetime: PhantomData,
        };

        Box::into_raw(Box::new(bytes))
    }
}

impl Bytes {
    /// See [`bytes_new()`] for more.
    unsafe fn new(data: *const u8, len: u32) -> Self {
        if data.is_null() || len == 0 {
            Self(Vec::new())
        } else {
            Self(unsafe { from_raw_parts(data, len as usize) }.to_vec())
        }
    }
}

impl<'a> CBytes<'a> {
    /// Slices into the raw bytes.
    ///
    /// # Safety
    /// The behavior is undefined if:
    /// - A non-null `data` doesn't point to an aligned, contiguous area of memory with at least
    /// `len` many [`u8`]s.
    /// - A `len` is too large to address the memory of a non-null [`u8`] array.
    pub unsafe fn as_slice(&self) -> &'a [u8] {
        if self.data.is_null() || self.len == 0 {
            &[]
        } else {
            unsafe { from_raw_parts(self.data, self.len as usize) }
        }
    }

    /// See [`bytes_drop()`] for more.
    unsafe fn drop(bytes: *mut Self) {
        if !bytes.is_null() {
            let bytes = unsafe { Box::from_raw(bytes) };
            if !bytes.data.is_null() && bytes.len > 0 {
                unsafe {
                    Box::from_raw(from_raw_parts_mut(
                        bytes.data as *mut u32,
                        bytes.len as usize,
                    ))
                };
            }
        }
    }
}

/// Creates an owned bytes buffer from the data.
///
/// # Errors
/// Returns a null pointer if:
/// - An unexpected panic happened.
///
/// # Safety
/// The bahavior is undefined if:
/// - A non-null `data` doesn't point to an aligned, contiguous area of memory with at least `len`
/// many [`u8`]s.
/// - A `len` is too large to address the memory of a non-null [`u8`] array.
/// - A non-null `error` doesn't point to an aligned, contiguous area of memory with an
/// [`ExternError`].
#[no_mangle]
pub unsafe extern "C" fn bytes_new(
    data: *const u8,
    len: u32,
    error: *mut ExternError,
) -> *mut CBytes<'static> {
    let new = || Ok(unsafe { Bytes::new(data, len) });
    let clean = || {};
    let error = unsafe { error.as_mut() };

    call_with_result(new, clean, error)
}

/// Frees the memory of the bytes buffer.
///
/// # Safety
/// The behavior is undefined if:
/// - A non-null `bytes` buffer doesn't point to memory allocated by [`bytes_new()`] or
/// [`xaynai_serialize()`].
/// - A non-null `bytes` buffer is freed more than once.
/// - A non-null `bytes` buffer is accessed after being freed.
///
/// [`xaynai_serialize()`]: crate::reranker::ai::xaynai_serialize
#[no_mangle]
pub unsafe extern "C" fn bytes_drop(bytes: *mut CBytes) {
    let drop = || {
        unsafe { CBytes::drop(bytes) };
        Ok(())
    };
    let clean = || {};
    let error = None;

    call_with_result(drop, clean, error);
}

#[cfg(test)]
mod tests {
    use std::pin::Pin;

    use super::*;
    use crate::{result::error::CCode, utils::tests::AsPtr};

    impl AsPtr for CBytes<'_> {}

    impl From<Pin<&[u8]>> for CBytes<'_> {
        fn from(bytes: Pin<&[u8]>) -> Self {
            let len = bytes.len() as u32;
            let data = if bytes.is_empty() {
                null()
            } else {
                bytes.as_ptr()
            };

            Self {
                data,
                len,
                _lifetime: PhantomData,
            }
        }
    }

    pub struct TestBytes<'a> {
        vec: Pin<Vec<u8>>,
        bytes: CBytes<'a>,
    }

    impl Drop for TestBytes<'_> {
        fn drop(&mut self) {}
    }

    impl<'a> AsPtr<CBytes<'a>> for TestBytes<'a> {
        fn as_ptr(&self) -> *const CBytes<'a> {
            self.bytes.as_ptr()
        }

        fn as_mut_ptr(&mut self) -> *mut CBytes<'a> {
            self.bytes.as_mut_ptr()
        }
    }

    impl Default for TestBytes<'_> {
        fn default() -> Self {
            let vec = Pin::new((0..10).collect::<Vec<_>>());
            let bytes = vec.as_ref().into();

            Self { vec, bytes }
        }
    }

    #[test]
    fn test_into_raw() {
        let buffer = TestBytes::default();
        let bytes = Bytes(buffer.vec.to_vec()).into_ffi_value();

        assert!(!bytes.is_null());
        assert!(!unsafe { &*bytes }.data.is_null());
        assert_eq!(unsafe { &*bytes }.len as usize, buffer.vec.len());
        assert_eq!(
            unsafe { (&*bytes).as_slice() },
            buffer.vec.as_ref().get_ref(),
        );

        unsafe { bytes_drop(bytes) };
    }

    #[test]
    fn test_into_empty() {
        let bytes = Bytes(Vec::new()).into_ffi_value();

        assert!(!bytes.is_null());
        assert!(unsafe { &*bytes }.data.is_null());
        assert_eq!(unsafe { &*bytes }.len, 0);
        assert!(unsafe { (&*bytes).as_slice() }.is_empty());

        unsafe { bytes_drop(bytes) };
    }

    #[test]
    fn test_new() {
        let buffer = TestBytes::default();
        let mut error = ExternError::default();

        let bytes = unsafe { bytes_new(buffer.bytes.data, buffer.bytes.len, error.as_mut_ptr()) };
        assert!(!bytes.is_null());
        assert_eq!(error.get_code(), CCode::Success);
        assert_eq!(
            unsafe { (&*bytes).as_slice() },
            buffer.vec.as_ref().get_ref(),
        );

        unsafe { bytes_drop(bytes) };
    }

    #[test]
    fn test_empty() {
        let mut error = ExternError::default();

        let bytes = unsafe { bytes_new(null(), 0, error.as_mut_ptr()) };
        assert!(!bytes.is_null());
        assert_eq!(error.get_code(), CCode::Success);
        assert!(unsafe { (&*bytes).as_slice() }.is_empty());

        unsafe { bytes_drop(bytes) };
    }
}
