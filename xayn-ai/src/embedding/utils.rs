use ndarray::Array1;

use std::ops::Deref;

fn l2_norm(array: Array1<f32>) -> f32 {
    let l2 = array.dot(&array).sqrt();
    if l2.is_normal() || l2 == 0.0 {
        l2
    } else {
        panic!("vectors must consist of real values only")
    }
}

/// Computes the l2 norm (euclidean metric) of the difference of two vectors.
///
/// # Panics
/// Panics if the vectors didn't consist of all real values.
pub fn l2_norm_distance<A, B>(a: &A, b: &B) -> f32
where
    A: Deref<Target = Array1<f32>>,
    B: Deref<Target = Array1<f32>>,
{
    l2_norm(a.deref() - b.deref())
}
