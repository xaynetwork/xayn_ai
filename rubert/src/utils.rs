use crate::ndarray::{ArcArray, ArrayBase, Ix2, Ix3, IxDyn};

pub type ArrayBase2<S> = ArrayBase<S, Ix2>;
pub type ArcArray2<S> = ArcArray<S, Ix2>;
pub type ArcArray3<S> = ArcArray<S, Ix3>;
pub type ArcArrayD<S> = ArcArray<S, IxDyn>;
