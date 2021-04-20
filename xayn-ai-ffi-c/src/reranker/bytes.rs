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
    // covariant in lifetime and type, only relevant for borrowed data
    pub(crate) _lifetime: PhantomData<&'a [u8]>,
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

/// Frees the memory of the bytes buffer.
///
/// # Safety
/// The behavior is undefined if:
/// - A non-null `bytes` buffer doesn't point to memory allocated by [`xaynai_serialize()`].
/// - A non-null `bytes` buffer is freed more than once.
/// - A non-null `bytes` buffer is accessed after being freed.
///
/// [`xaynai_serialize()`]: crate::reranker::ai::xaynai_serialize
#[no_mangle]
pub unsafe extern "C" fn bytes_drop(buffer: *mut CBytes) {
    let drop = || {
        unsafe { CBytes::drop(buffer) };
        Result::<_, ExternError>::Ok(())
    };
    let clean = || {};
    let error = None;

    call_with_result(drop, clean, error);
}

#[cfg(test)]
pub(crate) mod tests {
    use std::slice::from_raw_parts;

    use super::*;
    use crate::utils::tests::AsPtr;

    impl AsPtr for CBytes<'_> {}

    #[test]
    fn test_into_raw() {
        let buffer = (0..10).collect::<Vec<_>>();
        let bytes = Bytes(buffer.clone()).into_ffi_value();

        assert!(!bytes.is_null());
        let data = unsafe { &*bytes }.data;
        let len = unsafe { &*bytes }.len as usize;
        assert!(!data.is_null());
        assert_eq!(len, buffer.len());
        assert_eq!(unsafe { from_raw_parts(data, len) }, buffer);

        unsafe { bytes_drop(bytes) };
    }

    #[test]
    fn test_into_empty() {
        let bytes = Bytes(Vec::new()).into_ffi_value();

        assert!(!bytes.is_null());
        assert!(unsafe { &*bytes }.data.is_null());
        assert_eq!(unsafe { &*bytes }.len, 0);

        unsafe { bytes_drop(bytes) };
    }
}
