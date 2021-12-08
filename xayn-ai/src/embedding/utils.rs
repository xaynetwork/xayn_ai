use std::ops::Deref;

use ndarray::{ArrayBase, Data, Ix1};
use rubert::Embedding1;

pub(crate) type Embedding = Embedding1;

/// Computes the l2 norm (euclidean metric) of a vector.
///
/// # Panics
/// Panics if the vector doesn't consist solely of real values.
pub fn l2_norm<A, S>(a: &A) -> f32
where
    A: Deref<Target = ArrayBase<S, Ix1>>,
    S: Data<Elem = f32>,
{
    let norm = a.dot(a.deref()).sqrt();
    assert!(
        norm.is_finite(),
        "vector must consist of real values only, but got:\n{:?}",
        a.deref(),
    );

    norm
}

/// Computes the l2 norm (euclidean metric) of the difference of two vectors.
///
/// # Panics
/// Panics if the vectors don't consist solely of real values or their shapes don't match.
pub fn l2_distance<A, B, S>(a: &A, b: &B) -> f32
where
    A: Deref<Target = ArrayBase<S, Ix1>>,
    B: Deref<Target = ArrayBase<S, Ix1>>,
    S: Data<Elem = f32>,
{
    l2_norm(&Embedding::from(a.deref() - b.deref()))
}

/// Computes the arithmetic mean of two vectors.
///
/// # Panics
/// Panics if the vectors don't consist solely of real values or their shapes don't match.
pub fn mean<A, B, S>(a: &A, b: &B) -> Embedding
where
    A: Deref<Target = ArrayBase<S, Ix1>>,
    B: Deref<Target = ArrayBase<S, Ix1>>,
    S: Data<Elem = f32>,
{
    let mean = 0.5 * (a.deref() + b.deref());
    assert!(
        mean.iter().copied().all(f32::is_finite),
        "vectors must consist of real values only, but got\na: {:?}\nb: {:?}",
        a.deref(),
        b.deref(),
    );

    mean.into()
}

/// Computes the cosine similarity of two vectors.
///
/// # Panics
/// Panics if the vectors don't consist solely of real values or their shapes don't match.
pub fn cosine_similarity<A, B, S>(a: &A, b: &B) -> f32
where
    A: Deref<Target = ArrayBase<S, Ix1>>,
    B: Deref<Target = ArrayBase<S, Ix1>>,
    S: Data<Elem = f32>,
{
    let norm_a = l2_norm(a);
    let norm_b = l2_norm(b);

    (norm_a != 0. && norm_b != 0.)
        .then(|| a.dot(b.deref()) / norm_a / norm_b)
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use ndarray::arr1;

    use super::*;
    use test_utils::assert_approx_eq;

    #[test]
    fn test_l2_norm() {
        let a = Embedding::from(arr1(&[1., 2., 3.]));
        assert_approx_eq!(f32, l2_norm(&a), 3.7416575);
    }

    #[test]
    #[should_panic(expected = "vector must consist of real values only, but got")]
    fn test_l2_norm_nan() {
        let a = Embedding::from(arr1(&[1., f32::NAN, 3.]));
        l2_norm(&a);
    }

    #[test]
    #[should_panic(expected = "vector must consist of real values only, but got")]
    fn test_l2_norm_inf() {
        let a = Embedding::from(arr1(&[1., f32::INFINITY, 3.]));
        l2_norm(&a);
    }

    #[test]
    #[should_panic(expected = "vector must consist of real values only, but got")]
    fn test_l2_norm_neginf() {
        let a = Embedding::from(arr1(&[1., f32::NEG_INFINITY, 3.]));
        l2_norm(&a);
    }

    #[test]
    fn test_l2_distance() {
        let a = Embedding::from(arr1(&[1., 2., 3.]));
        let b = Embedding::from(arr1(&[4., 5., 6.]));
        assert_approx_eq!(f32, l2_distance(&a, &b), 5.196152);
    }

    #[test]
    fn test_mean() {
        let a: Embedding = arr1(&[1., 2., 3.]).into();
        let b: Embedding = arr1(&[4., 5., 6.]).into();
        let m = mean(&a, &b);
        let c = arr1(&[2.5, 3.5, 4.5]);
        assert_approx_eq!(f32, m.deref(), c);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = Embedding::from(arr1(&[1., 2., 3.]));
        let b = Embedding::from(arr1(&[4., 5., 6.]));
        assert_approx_eq!(f32, cosine_similarity(&a, &b), 0.97463185);
    }
}
