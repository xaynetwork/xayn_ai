use std::{panic::AssertUnwindSafe, slice::from_raw_parts_mut};

use ffi_support::{destroy_c_string, ExternError, IntoFfi};
use xayn_ai::Error;

use crate::result::{call_with_result, error::CCode};

/// The Xayn Ai faults.
pub struct Faults(Vec<String>);

/// A raw slice of faults.
#[repr(C)]
pub struct CFaults<'a> {
    /// The raw pointer to the faults.
    pub data: Option<&'a ExternError>,
    /// The number of faults.
    pub len: u32,
}

impl From<&[Error]> for Faults {
    fn from(faults: &[Error]) -> Self {
        Self(faults.iter().map(ToString::to_string).collect())
    }
}

unsafe impl IntoFfi for Faults {
    type Value = Option<&'static mut CFaults<'static>>;

    #[inline]
    fn ffi_default() -> Self::Value {
        None
    }

    #[inline]
    fn into_ffi_value(self) -> Self::Value {
        let len = self.0.len() as u32;
        let data = if self.0.is_empty() {
            None
        } else {
            self.0
                .into_iter()
                .map(|message| CCode::Fault.with_context(message))
                .collect::<Vec<_>>()
                .leak()
                .first()
        };

        Some(Box::leak(Box::new(CFaults { data, len })))
    }
}

impl CFaults<'_> {
    /// See [`faults_drop()`] for more.
    unsafe fn drop(faults: Option<&mut Self>) {
        if let Some(faults) = faults {
            let faults = unsafe { Box::from_raw(faults) };
            if let Some(data) = faults.data {
                if faults.len > 0 {
                    let faults = unsafe {
                        Box::from_raw(from_raw_parts_mut(
                            data as *const ExternError as *mut ExternError,
                            faults.len as usize,
                        ))
                    };
                    for fault in faults.iter() {
                        unsafe { destroy_c_string(fault.get_raw_message() as *mut _) }
                    }
                }
            }
        }
    }
}

/// Frees the memory of the faults.
///
/// # Safety
/// The behavior is undefined if:
/// - A non-null `faults` doesn't point to memory allocated by [`xaynai_faults()`].
/// - A non-null `faults` is freed more than once.
/// - A non-null `faults` is accessed after being freed.
///
/// [`xaynai_faults()`]: crate::reranker::ai::xaynai_faults
#[no_mangle]
pub unsafe extern "C" fn faults_drop(faults: Option<&mut CFaults>) {
    let drop = AssertUnwindSafe(|| {
        unsafe { CFaults::drop(faults) };
        Ok(())
    });
    let clean = || {};
    let error = None;

    call_with_result(drop, clean, error);
}

#[cfg(test)]
mod tests {
    use std::slice::from_raw_parts;

    use itertools::izip;

    use super::*;

    struct TestFaults(Vec<Error>);

    impl Default for TestFaults {
        fn default() -> Self {
            Self(
                (0..10)
                    .map(|idx| Error::msg(format!("fault {}", idx)))
                    .collect(),
            )
        }
    }

    #[test]
    fn test_from_faults() {
        let buffer = TestFaults::default().0;
        let faults = Faults::from(buffer.as_slice());
        assert_eq!(faults.0.len(), buffer.len());
        for (fault, error) in izip!(faults.0, buffer) {
            assert_eq!(fault, error.to_string());
        }
    }

    #[test]
    fn test_from_empty() {
        let faults = Faults::from(Vec::new().as_slice());
        assert!(faults.0.is_empty());
    }

    #[test]
    fn test_into_raw() {
        let buffer = TestFaults::default().0;
        let faults = Faults::from(buffer.as_slice()).into_ffi_value().unwrap();

        assert!(faults.data.is_some());
        assert_eq!(faults.len as usize, buffer.len());
        for (fault, error) in izip!(
            unsafe { from_raw_parts(faults.data.unwrap(), faults.len as usize) },
            buffer,
        ) {
            assert_eq!(fault.get_code(), CCode::Fault);
            assert_eq!(fault.get_message(), error.to_string().as_str());
        }

        unsafe { faults_drop(Some(faults)) };
    }

    #[test]
    fn test_into_empty() {
        let faults = Faults(Vec::new()).into_ffi_value().unwrap();

        assert!(faults.data.is_none());
        assert_eq!(faults.len, 0);

        unsafe { faults_drop(Some(faults)) };
    }
}
