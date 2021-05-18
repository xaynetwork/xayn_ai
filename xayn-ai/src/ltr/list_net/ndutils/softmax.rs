use ndarray::{ArrayBase, Axis, DataMut, DataOwned, Dimension, NdFloat, RemoveAxis};

/// Computes softmax along specified axis.
///
/// Inspired by [autograd's softmax implementation], especially the trick to subtract the max value
/// to reduce the chance of an overflow.
///
/// [autograd softmax]: https://docs.rs/autograd/1.0.0/src/autograd/ops/activation_ops.rs.html#59
pub fn softmax<A, S, D>(array: ArrayBase<S, D>, axis: Axis) -> ArrayBase<S, D>
where
    A: NdFloat,
    S: DataOwned<Elem = A> + DataMut<Elem = A>,
    D: Dimension + RemoveAxis,
{
    // Subtract `max` to prevent overflow, this
    // doesn't affect the outcome of the softmax.
    let max = array
        .fold_axis(axis, A::min_value(), |state, val| A::max(*state, *val))
        .insert_axis(axis);
    let mut tmp = array - max;

    // Standard 3step softmax, 1) exp(x), 2) sum up, 3) divide through sum
    tmp.mapv_inplace(|v| v.exp());
    let sum = tmp.sum_axis(axis).insert_axis(axis);
    tmp / sum
}

#[cfg(test)]
mod tests {
    use ndarray::{arr1, arr2, arr3};

    use super::*;

    #[test]
    fn test_softmax_1d() {
        let arr = arr1(&[-1_f32, 0., 1., 2., 3.]);

        // axis 0
        let res = arr1(&[
            0.011656231_f32,
            0.03168492,
            0.08612854,
            0.23412167,
            0.6364086,
        ]);
        assert_ndarray_eq!(f32, softmax(arr, Axis(0)), res);
    }

    #[test]
    fn test_softmax_2d() {
        let arr = arr2(&[
            [-1_f32, 0., 1., 2., 3.],
            [9., 8., 7., 6., 5.],
            [1., -1., 1., -1., 1.],
        ])
        .into_shared();

        // axis 0
        let res = arr2(&[
            [
                0.000045382647_f32,
                0.00033530878,
                0.0024665247,
                0.017970119,
                0.11731042,
            ],
            [0.99961925, 0.9995414, 0.995067, 0.9811352, 0.8668133],
            [
                0.00033533492,
                0.00012335321,
                0.0024665247,
                0.0008946795,
                0.01587624,
            ],
        ]);
        assert_ndarray_eq!(f32, softmax(arr.clone(), Axis(0)), res);

        // axis 1
        let res = arr2(&[
            [
                0.011656231_f32,
                0.03168492,
                0.08612854,
                0.23412167,
                0.6364086,
            ],
            [0.6364086, 0.23412167, 0.08612854, 0.03168492, 0.011656231],
            [0.3057477, 0.04137845, 0.3057477, 0.04137845, 0.3057477],
        ]);
        assert_ndarray_eq!(f32, softmax(arr, Axis(1)), res);
    }

    #[test]
    fn test_softmax_3d() {
        let arr = arr3(&[
            [
                [-1_f32, 0., 1., 2., 3.],
                [9., 8., 7., 6., 5.],
                [1., -1., 1., -1., 1.],
            ],
            [
                [1., 1., 1., 1., 1.],
                [2., 2., 2., 2., 2.],
                [3., 3., 3., 3., 3.],
            ],
        ])
        .into_shared();

        // axis 0
        let res = arr3(&[
            [
                [0.11920292_f32, 0.26894143, 0.5, 0.7310586, 0.880797],
                [0.999089, 0.9975274, 0.9933072, 0.98201376, 0.95257413],
                [0.11920292, 0.01798621, 0.11920292, 0.01798621, 0.11920292],
            ],
            [
                [0.880797, 0.7310586, 0.5, 0.26894143, 0.11920292],
                [
                    0.00091105123,
                    0.0024726233,
                    0.006692851,
                    0.01798621,
                    0.047425874,
                ],
                [0.880797, 0.98201376, 0.880797, 0.98201376, 0.880797],
            ],
        ]);
        assert_ndarray_eq!(f32, softmax(arr.clone(), Axis(0)), res);

        // axis 1
        let res = arr3(&[
            [
                [
                    0.000045382647_f32,
                    0.00033530878,
                    0.0024665247,
                    0.017970119,
                    0.11731042,
                ],
                [0.99961925, 0.9995414, 0.995067, 0.9811352, 0.8668133],
                [
                    0.00033533492,
                    0.00012335321,
                    0.0024665247,
                    0.0008946795,
                    0.01587624,
                ],
            ],
            [
                [0.09003057, 0.09003057, 0.09003057, 0.09003057, 0.09003057],
                [0.24472848, 0.24472848, 0.24472848, 0.24472848, 0.24472848],
                [0.66524094, 0.66524094, 0.66524094, 0.66524094, 0.66524094],
            ],
        ]);
        assert_ndarray_eq!(f32, softmax(arr.clone(), Axis(1)), res);

        // axis 2
        let res = arr3(&[
            [
                [
                    0.011656232_f32,
                    0.03168492,
                    0.08612854,
                    0.23412167,
                    0.6364086,
                ],
                [0.6364086, 0.23412167, 0.08612854, 0.03168492, 0.011656231],
                [0.3057477, 0.04137845, 0.3057477, 0.04137845, 0.3057477],
            ],
            [
                [0.2, 0.2, 0.2, 0.2, 0.2],
                [0.2, 0.2, 0.2, 0.2, 0.2],
                [0.2, 0.2, 0.2, 0.2, 0.2],
            ],
        ]);
        assert_ndarray_eq!(f32, softmax(arr, Axis(2)), res);
    }

    #[test]
    fn test_softmax_edgecases() {
        // 2D axis 0
        let arr = arr2(&[[-1_f32, 0., 1., 2., 3.]]);
        let res = arr2(&[[1_f32, 1., 1., 1., 1.]]);
        assert_ndarray_eq!(f32, softmax(arr, Axis(0)), res);

        // 2D axis 1
        let arr = arr2(&[[-1_f32], [9.], [1.]]);
        let res = arr2(&[[1_f32], [1.], [1.]]);
        assert_ndarray_eq!(f32, softmax(arr, Axis(1)), res);

        // 3D axis 0
        let arr = arr3(&[[
            [-1_f32, 0., 1., 2., 3.],
            [9., 8., 7., 6., 5.],
            [1., -1., 1., -1., 1.],
        ]]);
        let res = arr3(&[[
            [1_f32, 1., 1., 1., 1.],
            [1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1.],
        ]]);
        assert_ndarray_eq!(f32, softmax(arr, Axis(0)), res);

        // 3D axis 1
        let arr = arr3(&[[[-1_f32, 0., 1., 2., 3.]], [[1., 1., 1., 1., 1.]]]);
        let res = arr3(&[[[1_f32, 1., 1., 1., 1.]], [[1., 1., 1., 1., 1.]]]);
        assert_ndarray_eq!(f32, softmax(arr, Axis(1)), res);

        // 3D axis 2
        let arr = arr3(&[[[-1_f32], [9.], [1.]], [[1.], [2.], [3.]]]);
        let res = arr3(&[[[1_f32], [1.], [1.]], [[1.], [1.], [1.]]]);
        assert_ndarray_eq!(f32, softmax(arr, Axis(2)), res);
    }
}
