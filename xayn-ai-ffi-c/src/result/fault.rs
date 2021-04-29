use std::{convert::Infallible, panic::AssertUnwindSafe, slice};

use xayn_ai::Error;

use crate::{
    result::{
        call_with_result,
        error::{CCode, CError},
    },
    utils::IntoRaw,
};

/// The Xayn Ai faults.
pub struct Faults(Vec<String>);

/// A raw slice of faults.
#[repr(C)]
pub struct CFaults {
    /// The raw pointer to the faults.
    pub data: Option<Box<CError>>,
    /// The number of faults.
    pub len: u32,
}

impl From<&[Error]> for Faults {
    fn from(faults: &[Error]) -> Self {
        Self(faults.iter().map(ToString::to_string).collect())
    }
}

unsafe impl IntoRaw for Faults
where
    CFaults: Sized,
{
    // Safety:
    // CFaults is sized, hence Box<CFaults> is representable as a *mut CFaults and
    // Option<Box<CFaults>> is applicable for the nullable pointer optimization.
    type Value = Option<Box<CFaults>>;

    #[inline]
    fn into_raw(self) -> Self::Value {
        let len = self.0.len() as u32;
        let data = if self.0.is_empty() {
            None
        } else {
            // Safety:
            // Casting a Box<[u8]> to a Box<u8> is sound, but it leaks all values except the very
            // first one. Since all slices are terminated with a single null byte each, we are able
            // to recover the lengths and reclaim the memory.
            let data = self
                .0
                .into_iter()
                .map(|message| CCode::Fault.with_context(message).into_raw())
                .collect::<Vec<_>>();
            // Safety:
            // Casting a Box<[CError]> to a Box<CError> is sound, but it leaks all values except the
            // very first one. Hence we store the length of the slice next to the pointer to be able
            // to reclaim the memory.
            Some(unsafe { Box::from_raw(data.leak().as_mut_ptr()) })
        };

        Some(Box::new(CFaults { data, len }))
    }
}

impl CFaults {
    /// See [`faults_drop()`] for more.
    #[allow(clippy::unnecessary_wraps)]
    unsafe fn drop(faults: Option<Box<Self>>) -> Result<(), Infallible> {
        if let Some(faults) = faults {
            if let Some(data) = faults.data {
                if faults.len > 0 {
                    // Safety:
                    // Casting a Box<CFaults> to a Box<[CFaults]> is sound, if it originated from a
                    // boxed slice with corresponding length.
                    let mut faults = unsafe {
                        Box::from_raw(slice::from_raw_parts_mut(
                            Box::into_raw(data),
                            faults.len as usize,
                        ))
                    };
                    // Safety:
                    // Casting a Box<u8> to a CString is sound, if it originated from boxed slice
                    // with a terminating null byte.
                    for fault in faults.iter_mut() {
                        let _ = unsafe { CError::drop_message(Some(fault)) };
                    }
                }
            }
        }

        Ok(())
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
pub unsafe extern "C" fn faults_drop(faults: Option<Box<CFaults>>) {
    let drop = AssertUnwindSafe(
        // Safety: The memory is dropped anyways.
        || unsafe { CFaults::drop(faults) },
    );
    let error = None;

    call_with_result(drop, error);
}

#[cfg(test)]
mod tests {
    use itertools::izip;

    use super::*;
    use crate::utils::tests::{as_str_unchecked, AsPtr};

    impl AsPtr for CFaults {}

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
        let faults = Faults::from(buffer.as_slice()).into_raw().unwrap();

        let data = faults.data.as_ref().unwrap().as_ref();
        let len = faults.len as usize;
        assert_eq!(len, buffer.len());
        for (fault, error) in izip!(unsafe { slice::from_raw_parts(data, len) }, buffer) {
            assert_eq!(fault.code, CCode::Fault);
            assert_eq!(
                as_str_unchecked(fault.message.as_ref().map(AsRef::as_ref)),
                error.to_string(),
            );
        }

        unsafe { faults_drop(faults.into_ptr()) };
    }

    #[test]
    fn test_into_empty() {
        let faults = Faults(Vec::new()).into_raw().unwrap();

        assert!(faults.data.is_none());
        assert_eq!(faults.len, 0);

        unsafe { faults_drop(faults.into_ptr()) };
    }
}
