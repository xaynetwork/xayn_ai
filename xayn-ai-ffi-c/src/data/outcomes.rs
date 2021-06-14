use crate::{slice::CBoxedSlice, utils::IntoRaw};

/// C-FFI accessible version of [`RerankingOutcomes`].
///
/// [`RerankingOutcomes`]: xayn_ai::RerankingOutcomes
#[repr(C)]
pub struct CRerankingOutcomes {
    /// The final ranking.
    ///
    /// See [`RerankingOutcomes.final_ranking`].
    ///
    /// Should only be empty if the input document list was empty.
    ///
    /// [`RerankingOutcomes.final_ranking`]: xayn_ai::RerankingOutcomes::final_ranking
    pub final_ranking: CBoxedSlice<u16>,

    /// See [`RerankingOutcomes.qambert_similarities`].
    ///
    /// If it was `None` the `CBoxedSlice` instance will have a
    /// null-pointer and a length of 0.
    ///
    /// If it was `Some` but still empty it will have a dangling
    /// pointer and a length of 0.
    ///
    /// [`RerankingOutcomes.qambert_similarities`]: xayn_ai::RerankingOutcomes::qambert_similarities
    pub qambert_similarities: CBoxedSlice<f32>,

    /// See [`RerankingOutcomes.context_scores`].
    ///
    /// If it was `None` the `CBoxedSlice` instance will have a
    /// null-pointer and a length of 0.
    ///
    /// If it was `Some` but still empty it will have a dangling
    /// pointer and a length of 0.
    ///
    /// [`RerankingOutcomes.context_scores`]: xayn_ai::RerankingOutcomes::context_scores
    pub context_scores: CBoxedSlice<f32>,
}

/// Wrapper to implement `IntoRaw` for [`RerankingOutcomes`].
///
/// [`RerankingOutcomes`]: xayn_ai::RerankingOutcomes
pub struct RerankingOutcomes(xayn_ai::RerankingOutcomes);

unsafe impl IntoRaw for RerankingOutcomes
where
    CRerankingOutcomes: Sized,
{
    // Safety:
    // CRerankingOutcomes is sized, hence Box<CRerankingOutcomes> is representable as a
    // *mut CRerankingOutcomes and Option<Box<CRerankingOutcomes>> is eligible for the nullable
    // pointer optimization.
    type Value = Option<Box<CRerankingOutcomes>>;

    #[inline]
    fn into_raw(self) -> Self::Value {
        let RerankingOutcomes(xayn_ai::RerankingOutcomes {
            final_ranking,
            qambert_similarities,
            context_scores,
        }) = self;

        Some(Box::new(CRerankingOutcomes {
            final_ranking: final_ranking.into(),
            qambert_similarities: qambert_similarities.into(),
            context_scores: context_scores.into(),
        }))
    }
}

impl From<xayn_ai::RerankingOutcomes> for RerankingOutcomes {
    fn from(outcomes: xayn_ai::RerankingOutcomes) -> Self {
        Self(outcomes)
    }
}

/// Runs the destructor on a `Option<Box<CRerankingOutcomes>>`.
///
/// You can pass in a potential null pointer, but if the pointer
/// is not null it must be valid for usage in rust.
///
/// This **moves ownership** from C-FFI to rust, the pointer
/// (or any pointer derived from it/it was derived from)
/// MUST NOT be used after this function was called.
///
/// Doing so would implicitly violate rust's safety guarantees.
///
/// # Safety
///
/// This is safe (in rust terms) as long as the C-FFI does keep its
/// constraints and doesn't violate any rust constraints wrt. the
/// passed in data (which makes it always rust-safe).
///
/// Explicitly:
///
/// - Either the pointer must be null or
/// - it must have been allocated by rusts global allocator
/// - it must be aligned properly and non-dangling
///     - which always should be the case if it was properly allocated by rust
/// - and the data-structure behind the pointer must not have been changed in
///   any way which would break rust invariants (just don't change it at all!).
#[no_mangle]
pub unsafe extern "C" fn reranking_outcomes_drop(
    _reranking_outcomes: Option<Box<CRerankingOutcomes>>,
) {
}

#[cfg(test)]
mod tests {
    use super::*;
    use xayn_ai::assert_approx_eq;

    #[test]
    fn test_into_raw() {
        let value = xayn_ai::RerankingOutcomes {
            final_ranking: vec![0, 2, 3, 1],
            qambert_similarities: Some(vec![3.0, 2.125, 4.5, 21.25]),
            context_scores: Some(vec![2.0, 1.0, 0.2, 0.8]),
        };
        let raw = RerankingOutcomes(value).into_raw().unwrap();

        assert_eq!(raw.final_ranking, [0, 2, 3, 1]);
        assert_approx_eq!(
            f32,
            raw.qambert_similarities.as_slice(),
            [3.0, 2.125, 4.5, 21.25],
            ulps = 0
        );
        assert_approx_eq!(
            f32,
            raw.context_scores.as_slice(),
            [2.0, 1.0, 0.2, 0.8],
            ulps = 0
        );
    }

    #[test]
    fn test_into_raw_no_data() {
        let value = xayn_ai::RerankingOutcomes {
            final_ranking: vec![0, 2, 3, 1],
            qambert_similarities: None,
            context_scores: None,
        };
        let raw = RerankingOutcomes(value).into_raw().unwrap();

        assert!(raw.qambert_similarities.is_empty());
        assert!(raw.context_scores.is_empty());
    }

    #[test]
    fn test_into_empty() {
        assert!(<RerankingOutcomes as IntoRaw>::Value::default().is_none());
    }
}
