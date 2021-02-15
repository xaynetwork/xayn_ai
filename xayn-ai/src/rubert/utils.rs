use crate::rubert::ndarray::{ArcArray, ArrayBase, Ix1, Ix2, Ix3, Ix4, IxDyn};

pub type ArrayBase1<S> = ArrayBase<S, Ix1>;
pub type ArrayBase2<S> = ArrayBase<S, Ix2>;
pub type ArrayBase3<S> = ArrayBase<S, Ix3>;
pub type ArrayBase4<S> = ArrayBase<S, Ix4>;
pub type ArrayBaseD<S> = ArrayBase<S, IxDyn>;

pub type ArcArray1<S> = ArcArray<S, Ix1>;
pub type ArcArray2<S> = ArcArray<S, Ix2>;
pub type ArcArray3<S> = ArcArray<S, Ix3>;
pub type ArcArray4<S> = ArcArray<S, Ix4>;
pub type ArcArrayD<S> = ArcArray<S, IxDyn>;
