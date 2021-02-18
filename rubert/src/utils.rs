use crate::ndarray::{ArcArray, ArrayBase, Ix2, Ix3, IxDyn};

pub type ArrayBase2<S> = ArrayBase<S, Ix2>;
pub type ArcArray2<S> = ArcArray<S, Ix2>;
pub type ArcArray3<S> = ArcArray<S, Ix3>;
pub type ArcArrayD<S> = ArcArray<S, IxDyn>;

// temporary test utils for `xayn_ai`, bound to be changed anytime
mod test_utils {
    use crate::{pipeline::Embeddings, utils::ArcArray2};

    // 2D embeddings (like from first/average pooling) From impl for #[cfg(test)]
    impl From<Vec<f32>> for Embeddings {
        fn from(vec: Vec<f32>) -> Self {
            Embeddings(
                ArcArray2::from_shape_vec((1, vec.len()), vec)
                    .unwrap()
                    .into_dyn(),
            )
        }
    }

    // Debug impl for #[cfg_attr(test, derive(Debug))]
    impl std::fmt::Debug for Embeddings {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            self.0.fmt(f)
        }
    }

    // PartialEq impl for #[cfg_attr(test, derive(PartialEq))]
    impl PartialEq for Embeddings {
        fn eq(&self, other: &Self) -> bool {
            self.0 == other.0
        }
    }
}
