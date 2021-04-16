use std::{ptr::null_mut, slice::from_raw_parts_mut};

use ffi_support::{ExternError, IntoFfi};

use crate::result::call_with_result;

#[repr(C)]
pub struct CBytes {
    /// pointer to the data
    pub ptr: *const u8,
    /// number of bytes in the array
    pub len: u32,
}

unsafe impl IntoFfi for CBytes {
    type Value = *mut CBytes;

    #[inline]
    fn ffi_default() -> Self::Value {
        null_mut()
    }

    #[inline]
    fn into_ffi_value(self) -> Self::Value {
        Box::into_raw(Box::new(self))
    }
}

impl CBytes {
    pub fn from_vec(bytes: Vec<u8>) -> Self {
        if bytes.is_empty() {
            Self {
                ptr: null_mut(),
                len: 0,
            }
        } else {
            let len = bytes.len() as u32;
            let ptr = bytes.leak().as_mut_ptr();

            Self { ptr, len }
        }
    }

    fn drop(array: *mut CBytes) {
        if let Some(a) = unsafe { array.as_ref() } {
            if !a.ptr.is_null() && a.len > 0 {
                unsafe { Box::from_raw(from_raw_parts_mut(a.ptr as *mut u8, a.len as usize)) };
            }
            // Safety: we do not access `a` after we freed it
            unsafe { Box::from_raw(array) };
        }
    }
}

/// Frees the memory of a byte buffer.
///
/// # Safety
/// The behavior is undefined if:
/// - A non-null buffer doesn't point to memory allocated by [`xaynai_serialize()`].
/// - A non-null buffer is freed more than once.
/// - A non-null buffer is accessed after being freed.
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
