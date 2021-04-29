use std::{convert::Infallible, panic::AssertUnwindSafe, slice};

use crate::{
    result::{call_with_result, error::CError},
    utils::IntoRaw,
};

/// A bytes buffer.
pub struct Bytes(pub(crate) Vec<u8>);

/// A raw slice of bytes.
#[repr(C)]
pub struct CBytes {
    /// The raw pointer to the bytes.
    pub data: Option<Box<u8>>,
    /// The number of bytes.
    pub len: u32,
}

unsafe impl IntoRaw for Bytes
where
    CBytes: Sized,
{
    // Safety:
    // CBytes is sized, hence Box<CBytes> is representable as a *mut CBytes and Option<Box<CBytes>>
    // is applicable for the nullable pointer optimization.
    type Value = Option<Box<CBytes>>;

    #[inline]
    fn into_raw(self) -> Self::Value {
        let len = self.0.len() as u32;
        let data = if self.0.is_empty() {
            None
        } else {
            // Safety:
            // Casting a Box<[u8]> to a Box<u8> is sound, but it leaks all values except the very
            // first one. Hence we store the length of the slice next to the pointer to be able to
            // reclaim the memory.
            Some(unsafe { Box::from_raw(self.0.leak().as_mut_ptr()) })
        };

        Some(Box::new(CBytes { data, len }))
    }
}

impl Bytes {
    /// See [`bytes_new()`] for more.
    #[allow(clippy::unnecessary_wraps)]
    fn new(len: u32) -> Result<Self, Infallible> {
        Ok(Self(vec![0; len as usize]))
    }
}

impl CBytes {
    /// Slices into the raw bytes.
    ///
    /// # Safety
    /// The behavior is undefined if:
    /// - A non-null `data` doesn't point to an aligned, contiguous area of memory with at least
    /// `len` many [`u8`]s.
    /// - A `len` is too large to address the memory of a non-null [`u8`] array.
    pub unsafe fn as_slice(&self) -> &[u8] {
        match (self.data.as_ref(), self.len) {
            (None, _) | (_, 0) => &[],
            (Some(data), len) => unsafe { slice::from_raw_parts(data.as_ref(), len as usize) },
        }
    }

    /// See [`bytes_drop()`] for more.
    #[allow(clippy::unnecessary_wraps)]
    unsafe fn drop(bytes: Option<Box<Self>>) -> Result<(), Infallible> {
        if let Some(bytes) = bytes {
            if let Some(data) = bytes.data {
                if bytes.len > 0 {
                    // Safety:
                    // Casting a Box<u8> to a Box<[u8]> is sound, if it originated from a boxed
                    // slice with corresponding length.
                    unsafe {
                        Box::from_raw(slice::from_raw_parts_mut(
                            Box::into_raw(data),
                            bytes.len as usize,
                        ))
                    };
                }
            }
        }

        Ok(())
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
pub unsafe extern "C" fn bytes_new(len: u32, error: Option<&mut CError>) -> Option<Box<CBytes>> {
    let new = || Bytes::new(len);

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
pub unsafe extern "C" fn bytes_drop(bytes: Option<Box<CBytes>>) {
    let drop = AssertUnwindSafe(
        // Safety: The memory is dropped anyways.
        || unsafe { CBytes::drop(bytes) },
    );
    let error = None;

    call_with_result(drop, error);
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::{result::error::CCode, utils::tests::AsPtr};

    impl AsPtr for CBytes {}

    struct TestBytes(Vec<u8>);

    impl Default for TestBytes {
        fn default() -> Self {
            Self(vec![0; 10])
        }
    }

    #[test]
    fn test_into_raw() {
        let buffer = TestBytes::default();
        let bytes = Bytes(buffer.0.clone()).into_raw().unwrap();

        assert!(bytes.data.is_some());
        assert_eq!(bytes.len as usize, buffer.0.len());
        assert_eq!(unsafe { bytes.as_slice() }, buffer.0);

        unsafe { bytes_drop(bytes.into_ptr()) };
    }

    #[test]
    fn test_into_empty() {
        let bytes = Bytes(Vec::new()).into_raw().unwrap();

        assert!(bytes.data.is_none());
        assert_eq!(bytes.len, 0);
        assert!(unsafe { bytes.as_slice() }.is_empty());

        unsafe { bytes_drop(bytes.into_ptr()) };
    }

    #[test]
    fn test_new() {
        let buffer = TestBytes::default();
        let mut error = CError::default();

        let bytes = unsafe { bytes_new(buffer.0.len() as u32, error.as_mut_ptr()) }.unwrap();
        assert_eq!(error.code, CCode::Success);
        assert_eq!(unsafe { bytes.as_slice() }, buffer.0);

        unsafe { bytes_drop(bytes.into_ptr()) };
    }

    #[test]
    fn test_empty() {
        let mut error = CError::default();

        let bytes = unsafe { bytes_new(0, error.as_mut_ptr()) }.unwrap();
        assert_eq!(error.code, CCode::Success);
        assert!(unsafe { bytes.as_slice() }.is_empty());

        unsafe { bytes_drop(bytes.into_ptr()) };
    }
}
