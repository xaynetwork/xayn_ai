use itertools::Itertools;
use ndarray::{Array1, Array2, ArrayBase, Data, Ix1, RawDataClone};
use rubert::Embedding1;

pub(crate) type Embedding = Embedding1;

/// Computes the l2 norm (euclidean metric) of a vector.
///
/// # Panics
/// Panics if the vector doesn't consist solely of real values.
pub fn l2_norm<S>(a: ArrayBase<S, Ix1>) -> f32
where
    S: Data<Elem = f32>,
{
    let norm = a.dot(&a).sqrt();
    assert!(
        norm.is_finite(),
        "vector must consist of real values only, but got:\n{:?}",
        a,
    );

    norm
}

/// Computes the l2 norm (euclidean metric) of the difference of two vectors.
///
/// # Panics
/// Panics if the vectors don't consist solely of real values or their shapes don't match.
pub fn l2_distance<S>(a: ArrayBase<S, Ix1>, b: ArrayBase<S, Ix1>) -> f32
where
    S: Data<Elem = f32>,
{
    l2_norm(&a - &b)
}

/// Computes the arithmetic mean of two vectors.
///
/// # Panics
/// Panics if the vectors don't consist solely of real values or their shapes don't match.
pub fn mean<S>(a: ArrayBase<S, Ix1>, b: ArrayBase<S, Ix1>) -> Array1<f32>
where
    S: Data<Elem = f32>,
{
    let mean = 0.5 * (&a + &b);
    assert!(
        mean.iter().copied().all(f32::is_finite),
        "vectors must consist of real values only, but got\na: {:?}\nb: {:?}",
        a,
        b,
    );

    mean
}

/// Computes the pairwise cosine similarities of vectors.
///
/// Zero vectors are chosen to be similar to all other vectors, i.e. a similarity of 1.
///
/// # Panics
/// Panics if the vectors don't consist solely of real values or their shapes don't match.
///
/// Only use this with [`TrustedLen`] iterators, otherwise indexing may panic. The bound can't be
/// named currently, because the trait is nightly-gated. [`ExactSizeIterator`] isn't a feasible and
/// trusted replacement, for example it isn't implemented for [`Chain`]ed iterators.
///
/// [`TrustedLen`]: std::iter::TrustedLen
/// [`ExactSizeIterator`]: std::iter::ExactSizeIterator
/// [`Chain`]: std::iter::Chain
pub fn pairwise_cosine_similarity<I, S>(iter: I) -> Array2<f32>
where
    I: IntoIterator<Item = ArrayBase<S, Ix1>>,
    I::IntoIter: Clone,
    S: Data<Elem = f32> + RawDataClone,
{
    let iter = iter.into_iter();
    let size = match iter.size_hint() {
        (lower, Some(upper)) if lower == upper => lower,
        (lower, None) if lower == usize::MAX => lower,
        _ => unimplemented!("I::IntoIter: TrustedLen"),
    };

    let norms = iter.clone().map(|a| l2_norm(a.view())).collect::<Vec<_>>();
    let mut similarities = Array2::ones((size, size));
    for ((i, a), (j, b)) in iter
        .clone()
        .enumerate()
        .cartesian_product(iter.enumerate())
        .filter(|((i, _), (j, _))| j > i && norms[*i] > 0. && norms[*j] > 0.)
    {
        similarities[[i, j]] = (a.dot(&b) / norms[i] / norms[j]).clamp(-1., 1.);
        similarities[[j, i]] = similarities[[i, j]];
    }

    similarities
}

/// Computes the cosine similarity of two vectors.
///
/// Zero vectors are chosen to be similar to all other vectors, i.e. a similarity of 1.
///
/// # Panics
/// Panics if the vectors don't consist solely of real values or their shapes don't match.
#[allow(dead_code)]
pub fn cosine_similarity<S>(a: ArrayBase<S, Ix1>, b: ArrayBase<S, Ix1>) -> f32
where
    S: Data<Elem = f32>,
{
    pairwise_cosine_similarity([a.view(), b.view()])[[0, 1]]
}

#[cfg(test)]
mod tests {
    use ndarray::{arr1, arr2};

    use super::*;
    use test_utils::assert_approx_eq;

    #[test]
    fn test_l2_norm() {
        assert_approx_eq!(f32, l2_norm(arr1(&[1., 2., 3.])), 3.7416575);
    }

    #[test]
    #[should_panic(expected = "vector must consist of real values only, but got")]
    fn test_l2_norm_nan() {
        l2_norm(arr1(&[1., f32::NAN, 3.]));
    }

    #[test]
    #[should_panic(expected = "vector must consist of real values only, but got")]
    fn test_l2_norm_inf() {
        l2_norm(arr1(&[1., f32::INFINITY, 3.]));
    }

    #[test]
    #[should_panic(expected = "vector must consist of real values only, but got")]
    fn test_l2_norm_neginf() {
        l2_norm(arr1(&[1., f32::NEG_INFINITY, 3.]));
    }

    #[test]
    fn test_l2_distance() {
        assert_approx_eq!(
            f32,
            l2_distance(arr1(&[1., 2., 3.]), arr1(&[4., 5., 6.])),
            5.196152,
        );
    }

    #[test]
    fn test_mean() {
        assert_approx_eq!(
            f32,
            mean(arr1(&[1., 2., 3.]), arr1(&[4., 5., 6.])),
            arr1(&[2.5, 3.5, 4.5]),
        );
    }

    #[test]
    fn test_cosine_similarity_empty() {
        assert_approx_eq!(
            f32,
            pairwise_cosine_similarity([] as [Array1<f32>; 0]),
            arr2(&[[]]),
        );
    }

    #[test]
    fn test_cosine_similarity_single() {
        assert_approx_eq!(
            f32,
            pairwise_cosine_similarity([arr1(&[1., 2., 3.])]),
            arr2(&[[1.]]),
        );
    }

    #[test]
    fn test_cosine_similarity_pair() {
        assert_approx_eq!(
            f32,
            pairwise_cosine_similarity([arr1(&[1., 2., 3.]), arr1(&[4., 5., 6.])]),
            arr2(&[[1., 0.97463185], [0.97463185, 1.]]),
        );
    }
}
