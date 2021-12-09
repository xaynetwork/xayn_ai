use ndarray::{Array1, Array2, ArrayBase, Data, Ix1};
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
/// # Panics
/// Panics if the vectors don't consist solely of real values or their shapes don't match.
#[allow(dead_code)]
pub fn pairwise_cosine_similarity<'a, I, S>(iter: I) -> Array2<f32>
where
    I: IntoIterator<Item = &'a ArrayBase<S, Ix1>>,
    I::IntoIter: Clone,
    S: Data<Elem = f32> + 'a,
{
    let iter = iter.into_iter();
    let norms = iter.clone().map(|a| l2_norm(a.view())).collect::<Vec<_>>();
    let size = iter.clone().count();
    let mut similarities = Array2::ones((size, size));
    iter.clone().enumerate().for_each(|(i, a)| {
        if norms[i] != 0. {
            iter.clone().enumerate().skip(i + 1).for_each(|(j, b)| {
                if norms[j] != 0. {
                    similarities[[i, j]] = a.dot(b) / norms[i] / norms[j];
                    similarities[[j, i]] = similarities[[i, j]];
                }
            });
        }
    });

    similarities
}

/// Computes the cosine similarity of two vectors.
///
/// # Panics
/// Panics if the vectors don't consist solely of real values or their shapes don't match.
#[allow(dead_code)]
pub fn cosine_similarity<S>(a: ArrayBase<S, Ix1>, b: ArrayBase<S, Ix1>) -> f32
where
    S: Data<Elem = f32>,
{
    pairwise_cosine_similarity(&[a, b])[[0, 1]]
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
            pairwise_cosine_similarity(&[] as &[Array1<f32>]),
            arr2(&[[]]),
        );
    }

    #[test]
    fn test_cosine_similarity_single() {
        assert_approx_eq!(
            f32,
            pairwise_cosine_similarity(&[arr1(&[1., 2., 3.])]),
            arr2(&[[1.]]),
        );
    }

    #[test]
    fn test_cosine_similarity_pair() {
        assert_approx_eq!(
            f32,
            pairwise_cosine_similarity(&[arr1(&[1., 2., 3.]), arr1(&[4., 5., 6.])]),
            arr2(&[[1., 0.97463185], [0.97463185, 1.]]),
        );
    }
}
