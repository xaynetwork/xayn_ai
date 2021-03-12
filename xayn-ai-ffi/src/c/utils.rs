use std::{ffi::CStr, slice};

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

/// A helper to pass error messages accross the ffi boundary.
pub struct ErrorMsg<'a> {
    pub(crate) buffer: Option<&'a mut [u8]>,
}

impl<'a> ErrorMsg<'a> {
    /// Creates an error message handler.
    pub unsafe fn new(error: *mut u8, size: u32) -> Self {
        if error.is_null() || size == 0 {
            ErrorMsg { buffer: None }
        } else {
            ErrorMsg {
                buffer: Some(unsafe { slice::from_raw_parts_mut(error, size as usize) }),
            }
        }
    }

    /// Sets an error message.
    pub fn set(&mut self, msg: impl AsRef<str>) {
        if let Some(ref mut error) = self.buffer {
            // ensure that we have space for a null byte at the end
            let msg_bytes_max = error.len() - 1;

            let mut msg_bytes_count = 0;
            for c in msg.as_ref().chars() {
                // we will include the char only if it will fit entirely in error
                let char_byte_len = c.len_utf8();
                if char_byte_len + msg_bytes_count > msg_bytes_max {
                    break;
                }
                msg_bytes_count += char_byte_len;
            }

            error[..msg_bytes_count].copy_from_slice(msg.as_ref()[..msg_bytes_count].as_bytes());
            error[msg_bytes_count] = 0;
        }
    }
}

#[cfg(test)]
pub mod tests {
    use std::{
        ffi::CString,
        ptr::{null, null_mut},
        str::from_utf8,
    };

    use super::*;

    impl<'a> From<&'a mut [u8]> for ErrorMsg<'a> {
        /// Creates an error message handler without unsafety.
        fn from(error: &'a mut [u8]) -> Self {
            if error.len() == 0 {
                ErrorMsg { buffer: None }
            } else {
                ErrorMsg {
                    buffer: Some(error),
                }
            }
        }
    }

    impl<'a> ErrorMsg<'a> {
        /// Creates a string from the error message handler.
        ///
        /// Stops at the first encountered null byte. Clears the buffer with a leading null byte.
        #[allow(clippy::wrong_self_convention, clippy::inherent_to_string)]
        pub fn to_string(&mut self) -> String {
            self.buffer
                .as_mut()
                .map(|buffer| {
                    let msg = buffer
                        .splitn(2, |b| *b == 0)
                        .next()
                        .map(|msg| from_utf8(msg).unwrap_or_default().to_string())
                        .unwrap_or_default();
                    buffer[0] = 0;
                    msg
                })
                .unwrap_or_default()
        }
    }

    #[test]
    fn test_cstr_to_string() {
        assert!(unsafe { cstr_to_string(null()) }.is_none());
        let cstring = CString::new("test string").unwrap();
        assert_eq!(
            unsafe { cstr_to_string(cstring.as_ptr() as *const u8) }.unwrap(),
            "test string",
        );
    }

    #[test]
    fn test_errormsg() {
        let msg = "안녕하세요السلام عليكم.";
        let assert_error_msg = |min: usize, max: usize, expected: &str| {
            // check that the error message is as expected when the error size is in [min, max)
            for error_size in min..max {
                let mut error_msg = vec![0; error_size];
                let error = if error_msg.is_empty() {
                    null_mut()
                } else {
                    error_msg.as_mut_ptr()
                };
                unsafe { ErrorMsg::new(error, error_size as u32) }.set(msg);

                assert!(from_utf8(&error_msg).is_ok());
                assert_eq!(
                    ErrorMsg::from(error_msg.as_mut_slice()).to_string(),
                    expected,
                );
            }
        };

        assert_error_msg(0, 4, "");
        assert_error_msg(4, 7, "안");
        assert_error_msg(7, 10, "안녕");
        assert_error_msg(10, 13, "안녕하");
        assert_error_msg(13, 16, "안녕하세");
        assert_error_msg(16, 18, "안녕하세요");
        assert_error_msg(18, 20, "안녕하세요ا");
        assert_error_msg(20, 22, "안녕하세요ال");
        assert_error_msg(22, 24, "안녕하세요الس");
        assert_error_msg(24, 26, "안녕하세요السل");
        assert_error_msg(26, 28, "안녕하세요السلا");
        assert_error_msg(28, 29, "안녕하세요السلام");
        assert_error_msg(29, 31, "안녕하세요السلام ");
        assert_error_msg(31, 33, "안녕하세요السلام ع");
        assert_error_msg(33, 35, "안녕하세요السلام عل");
        assert_error_msg(35, 37, "안녕하세요السلام علي");
        assert_error_msg(37, 39, "안녕하세요السلام عليك");
        assert_error_msg(39, 40, "안녕하세요السلام عليكم");
        assert_error_msg(40, 50, msg);
    }
}
