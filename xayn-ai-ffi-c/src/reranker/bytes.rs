use std::convert::Infallible;

use crate::{
    result::{call_with_result, error::CError},
    slice::CBoxedSlice,
    utils::IntoRaw,
};

/// A bytes buffer.
pub struct Bytes(pub(crate) Vec<u8>);

/// A raw slice of bytes.
pub type CBytes = CBoxedSlice<u8>;

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
        Some(Box::new(self.0.into_boxed_slice().into()))
    }
}

impl Bytes {
    /// See [`bytes_new()`] for more.
    #[allow(clippy::unnecessary_wraps)]
    fn new(len: u64) -> Result<Self, Infallible> {
        Ok(Self(vec![0; len as usize]))
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
pub unsafe extern "C" fn bytes_new(len: u64, error: Option<&mut CError>) -> Option<Box<CBytes>> {
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
pub unsafe extern "C" fn bytes_drop(_bytes: Option<Box<CBytes>>) {}

#[cfg(test)]
mod tests {
    use xayn_ai_ffi::CCode;

    use super::*;
    use crate::utils::tests::AsPtr;

    const fn test_bytes<const N: usize>() -> [u8; N] {
        [0; N]
    }

    #[test]
    fn test_into_raw() {
        let buffer = test_bytes::<10>();
        let bytes = Bytes(buffer.to_vec()).into_raw().unwrap();
        assert_eq!(bytes.as_slice(), buffer);
    }

    #[test]
    fn test_into_empty() {
        let bytes = Bytes(test_bytes::<0>().to_vec()).into_raw().unwrap();
        assert!(bytes.is_empty());
    }

    #[test]
    fn test_new() {
        let buffer = test_bytes::<10>();
        let mut error = CError::default();
        let bytes = unsafe { bytes_new(buffer.len() as u64, error.as_mut_ptr()) }.unwrap();
        assert_eq!(error.code, CCode::None);
        assert_eq!(bytes.as_slice(), buffer);
    }

    #[test]
    fn test_empty() {
        let mut error = CError::default();
        let bytes = unsafe { bytes_new(0, error.as_mut_ptr()) }.unwrap();
        assert_eq!(error.code, CCode::None);
        assert!(bytes.is_empty());
    }
}
