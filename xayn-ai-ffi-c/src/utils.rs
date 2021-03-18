use std::ffi::CStr;

/// This function does nothing.
///
/// Calling this prevents Swift to drop the library.
#[no_mangle]
pub extern "C" fn dummy_function() {}

/// Converts a raw C-string pointer to a string.
pub unsafe fn cstr_to_string(cstr: *const u8) -> Option<String> {
    if cstr.is_null() {
        return None;
    }

    unsafe { CStr::from_ptr(cstr as *const _) }
        .to_str()
        .map(|s| s.to_string())
        .ok()
}

#[cfg(test)]
pub mod tests {
    use std::{ffi::CString, ptr::null};

    use super::*;

    #[test]
    fn test_cstr_to_string() {
        assert!(unsafe { cstr_to_string(null()) }.is_none());
        let cstring = CString::new("test string").unwrap();
        assert_eq!(
            unsafe { cstr_to_string(cstring.as_ptr() as *const u8) }.unwrap(),
            "test string",
        );
    }
}
