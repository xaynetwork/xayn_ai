use ndarray::Array1;

use std::ops::Deref;

fn l2_norm(array: Array1<f32>) -> f32 {
    array.dot(&array).sqrt()
}

pub fn l2_norm_distance<A, B>(a: &A, b: &B) -> f32
where
    A: Deref<Target = Array1<f32>>,
    B: Deref<Target = Array1<f32>>,
{
    l2_norm(a.deref() - b.deref())
}
