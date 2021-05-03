use xayn_ai::Error;

use crate::{
    result::error::{CCode, CError},
    slice::CBoxedSlice,
    utils::IntoRaw,
};

/// The Xayn Ai faults.
pub struct Faults(Vec<String>);

/// A raw slice of faults.
pub type CFaults = CBoxedSlice<CError>;

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
        let faults = self
            .0
            .into_iter()
            .map(|message| CCode::Fault.with_context(message).into_raw())
            .collect::<Vec<_>>();
        Some(Box::new(faults.into_boxed_slice().into()))
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
pub unsafe extern "C" fn faults_drop(_faults: Option<Box<CFaults>>) {}

#[cfg(test)]
mod tests {
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
        let buffer = TestFaults::default();
        let faults = Faults::from(buffer.0.as_slice()).into_raw().unwrap();

        for (fault, error) in izip!(faults.as_slice(), buffer.0) {
            assert_eq!(fault.code, CCode::Fault);
            assert_eq!(
                fault.message.as_ref().unwrap().as_str_unchecked(),
                error.to_string(),
            );
        }
    }

    #[test]
    fn test_into_empty() {
        let faults = Faults(Vec::new()).into_raw().unwrap();
        assert!(faults.as_slice().is_empty());
    }
}
