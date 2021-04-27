use std::{
    panic::AssertUnwindSafe,
    slice::{from_raw_parts, from_raw_parts_mut},
};

use crate::{
    result::{call_with_result, error::CError},
    utils::IntoRaw,
};

/// A bytes buffer.
pub struct Bytes(pub(crate) Vec<u8>);

/// A raw slice of bytes.
#[repr(C)]
pub struct CBytes<'a> {
    /// The raw pointer to the bytes.
    pub data: Option<&'a u8>,
    /// The number of bytes.
    pub len: u32,
}

unsafe impl IntoRaw for Bytes {
    type Value = Option<&'static mut CBytes<'static>>;

    #[inline]
    fn into_raw(self) -> Self::Value {
        let len = self.0.len() as u32;
        let data = if self.0.is_empty() {
            None
        } else {
            self.0.leak().first()
        };

        Some(Box::leak(Box::new(CBytes { data, len })))
    }
}

impl Bytes {
    /// See [`bytes_new()`] for more.
    fn new(len: u32) -> Self {
        Self(vec![0; len as usize])
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
        match (self.data, self.len) {
            (None, _) | (_, 0) => &[],
            (Some(data), len) => unsafe { from_raw_parts(data, len as usize) },
        }
    }

    /// See [`bytes_drop()`] for more.
    unsafe fn drop(bytes: Option<&mut Self>) {
        if let Some(bytes) = bytes {
            let bytes = unsafe { Box::from_raw(bytes) };
            if let Some(data) = bytes.data {
                if bytes.len > 0 {
                    unsafe {
                        Box::from_raw(from_raw_parts_mut(
                            data as *const u8 as *mut u8,
                            bytes.len as usize,
                        ))
                    };
                }
            }
        }
    }
}

/// Creates a zeroized, owned bytes buffer.
///
/// # Errors
/// Returns a null pointer if:
/// - An unexpected panic happened.
///
/// # Safety
/// The behavior is undefined if:
/// - A `len` is too large to address the memory of a non-null [`u8`] array.
/// - A non-null `error` doesn't point to an aligned, contiguous area of memory with a [`CError`].
#[no_mangle]
pub unsafe extern "C" fn bytes_new(
    len: u32,
    error: Option<&mut CError>,
) -> Option<&'static mut CBytes<'static>> {
    let new = || Ok(Bytes::new(len));

    call_with_result(new, error)
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
pub unsafe extern "C" fn bytes_drop(bytes: Option<&mut CBytes>) {
    let drop = AssertUnwindSafe(
        // Safety: The memory is dropped anyways.
        || {
            unsafe { CBytes::drop(bytes) };
            Ok(())
        },
    );
    let error = None;

    call_with_result(drop, error);
}

#[cfg(test)]
mod tests {
    use std::pin::Pin;

    use super::*;
    use crate::result::error::CCode;

    pub struct TestBytes<'a> {
        vec: Pin<Vec<u8>>,
        bytes: CBytes<'a>,
    }

    impl Drop for TestBytes<'_> {
        fn drop(&mut self) {}
    }

    impl Default for TestBytes<'_> {
        fn default() -> Self {
            let vec = Pin::new(vec![0; 10]);
            let bytes = CBytes {
                data: unsafe { vec.as_ptr().as_ref() },
                len: vec.len() as u32,
            };

            Self { vec, bytes }
        }
    }

    #[test]
    fn test_into_raw() {
        let buffer = TestBytes::default();
        let bytes = Bytes(buffer.vec.to_vec()).into_raw().unwrap();

        assert!(bytes.data.is_some());
        assert_eq!(bytes.len as usize, buffer.vec.len());
        assert_eq!(unsafe { bytes.as_slice() }, buffer.vec.as_ref().get_ref());

        unsafe { bytes_drop(Some(bytes)) };
    }

    #[test]
    fn test_into_empty() {
        let bytes = Bytes(Vec::new()).into_raw().unwrap();

        assert!(bytes.data.is_none());
        assert_eq!(bytes.len, 0);
        assert!(unsafe { bytes.as_slice() }.is_empty());

        unsafe { bytes_drop(Some(bytes)) };
    }

    #[test]
    fn test_new() {
        let buffer = TestBytes::default();
        let mut error = CError::default();

        let bytes = unsafe { bytes_new(buffer.bytes.len, Some(&mut error)) }.unwrap();
        assert_eq!(error.code, CCode::Success);
        assert_eq!(unsafe { bytes.as_slice() }, buffer.vec.as_ref().get_ref());

        unsafe { bytes_drop(Some(bytes)) };
    }

    #[test]
    fn test_empty() {
        let mut error = CError::default();

        let bytes = unsafe { bytes_new(0, Some(&mut error)) }.unwrap();
        assert_eq!(error.code, CCode::Success);
        assert!(unsafe { bytes.as_slice() }.is_empty());

        unsafe { bytes_drop(Some(bytes)) };
    }
}
