use std::ops::Deref;

use ndarray::{ArrayBase, Data, Ix1};

/// Computes the l2 norm (euclidean metric) of the difference of two vectors.
///
/// # Panics
/// Panics if the vectors didn't consist of all real values.
pub fn l2_distance<A, B, S>(a: &A, b: &B) -> f32
where
    A: Deref<Target = ArrayBase<S, Ix1>>,
    B: Deref<Target = ArrayBase<S, Ix1>>,
    S: Data<Elem = f32>,
{
    let difference = a.deref() - b.deref();
    let distance = difference.dot(&difference).sqrt();

    if distance.is_nan() || distance.is_infinite() {
        panic!(
            "vectors must consist of real values only, but got\na: {:?}\nb: {:?}",
            a.deref(),
            b.deref(),
        );
    }

    distance
}

#[cfg(test)]
mod tests {
    use ndarray::arr1;

    use super::*;
    use crate::embedding::smbert::Embedding;

    #[test]
    fn test_l2_distance() {
        let a: Embedding = arr1(&[1., 2., 3.]).into();
        let b: Embedding = arr1(&[4., 5., 6.]).into();
        assert_eq!(l2_distance(&a, &b), 5.196152422706632);
    }

    #[test]
    #[should_panic(expected = "vectors must consist of real values only, but got")]
    fn test_l2_distance_nan() {
        let a: Embedding = arr1(&[1., 2., 3.]).into();
        let b: Embedding = arr1(&[4., f32::NAN, 6.]).into();
        l2_distance(&a, &b);
    }

    #[test]
    #[should_panic(expected = "vectors must consist of real values only, but got")]
    fn test_l2_distance_inf() {
        let a: Embedding = arr1(&[1., 2., 3.]).into();
        let b: Embedding = arr1(&[4., f32::INFINITY, 6.]).into();
        l2_distance(&a, &b);
    }

    #[test]
    #[should_panic(expected = "vectors must consist of real values only, but got")]
    fn test_l2_distance_neginf() {
        let a: Embedding = arr1(&[1., 2., 3.]).into();
        let b: Embedding = arr1(&[4., f32::NEG_INFINITY, 6.]).into();
        l2_distance(&a, &b);
    }
}
