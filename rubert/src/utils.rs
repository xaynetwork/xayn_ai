use crate::ndarray::{ArcArray, Ix2, IxDyn};

pub type ArcArray2<S> = ArcArray<S, Ix2>;
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
}
