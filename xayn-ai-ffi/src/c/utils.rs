use std::{ffi::CStr, slice, str::from_utf8};

/// This function does nothing.
///
/// It is only needed to make a dummy call to the library in the swift code to prevent dropping it.
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

    /// Creates a string from the error message handler.
    ///
    /// Stops at the first encountered null byte. Clears the buffer with a leading null byte.
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
