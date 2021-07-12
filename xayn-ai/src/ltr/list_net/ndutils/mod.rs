use ndarray::{ArrayBase, ArrayView1, Axis, DataMut, DataOwned, Dimension, NdFloat, RemoveAxis};

pub mod io;

/// Computes softmax along specified axis.
///
/// Inspired by [autograd's softmax implementation], especially the trick to subtract the max value
/// to reduce the chance of an overflow.
///
/// [autograd's softmax implementation]: https://docs.rs/autograd/1.0.0/src/autograd/ops/activation_ops.rs.html#59
pub fn softmax<A, S, D>(mut array: ArrayBase<S, D>, axis: Axis) -> ArrayBase<S, D>
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
    array -= &max;

    // Standard 3step softmax, 1) exp(x), 2) sum up, 3) divide through sum
    array.mapv_inplace(|v| v.exp());
    let sum = array.sum_axis(axis).insert_axis(axis);
    array /= &sum;
    array
}

/// Compute the Kullback-Leibler Divergence between a "good" distribution and one we want to evaluate.
///
/// This returns a result based on `nats`, i.e. it uses `ln` (instead of `log2` which
/// would produce a result based on `bits`).
///
/// All values are clamped/cliped to the range `f32::EPSILON..=1.`.
///
/// For the eval distribution this makes sense as we should never predict `0` but at most
/// a value so close to it, that it ends up as 0 due to the limited precision of
/// `f32`.
///
/// For the good distribution we could argue similarly. An alternative choice
/// is to return 0 if the good distributions probability is 0.)
pub fn kl_divergence(good_dist: ArrayView1<f32>, eval_dist: ArrayView1<f32>) -> f32 {
    good_dist.into_iter().zip(eval_dist.into_iter()).fold(
        0f32,
        |acc, (good_dist_prob, eval_dist_prob)| {
            let good_dist_prob = good_dist_prob.clamp(f32::EPSILON, 1.);
            let eval_dist_prob = eval_dist_prob.clamp(f32::EPSILON, 1.);
            acc + good_dist_prob * (good_dist_prob / eval_dist_prob).ln()
        },
    )
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
        assert_approx_eq!(f32, softmax(arr, Axis(0)), res);
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
        assert_approx_eq!(f32, softmax(arr.clone(), Axis(0)), res);

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
        assert_approx_eq!(f32, softmax(arr, Axis(1)), res);
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
        assert_approx_eq!(f32, softmax(arr.clone(), Axis(0)), res);

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
        assert_approx_eq!(f32, softmax(arr.clone(), Axis(1)), res);

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
        assert_approx_eq!(f32, softmax(arr, Axis(2)), res);
    }

    #[test]
    fn test_softmax_edgecases() {
        // 2D axis 0
        let arr = arr2(&[[-1_f32, 0., 1., 2., 3.]]);
        let res = arr2(&[[1_f32, 1., 1., 1., 1.]]);
        assert_approx_eq!(f32, softmax(arr, Axis(0)), res);

        // 2D axis 1
        let arr = arr2(&[[-1_f32], [9.], [1.]]);
        let res = arr2(&[[1_f32], [1.], [1.]]);
        assert_approx_eq!(f32, softmax(arr, Axis(1)), res);

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
        assert_approx_eq!(f32, softmax(arr, Axis(0)), res);

        // 3D axis 1
        let arr = arr3(&[[[-1_f32, 0., 1., 2., 3.]], [[1., 1., 1., 1., 1.]]]);
        let res = arr3(&[[[1_f32, 1., 1., 1., 1.]], [[1., 1., 1., 1., 1.]]]);
        assert_approx_eq!(f32, softmax(arr, Axis(1)), res);

        // 3D axis 2
        let arr = arr3(&[[[-1_f32], [9.], [1.]], [[1.], [2.], [3.]]]);
        let res = arr3(&[[[1_f32], [1.], [1.]], [[1.], [1.], [1.]]]);
        assert_approx_eq!(f32, softmax(arr, Axis(2)), res);
    }

    #[test]
    fn test_kl_divergence_calculation() {
        let good_dist = arr1(&[0.5, 0.1, 0.025, 0.3, 0.075]);
        let eval_dist = arr1(&[0.3, 0.2, 0.15, 0.2, 0.15]);

        let cost = kl_divergence(good_dist.view(), eval_dist.view());

        assert_approx_eq!(f32, cost, 0.210_957_6);
    }

    #[test]
    fn test_kl_divergence_calculation_handles_zeros() {
        let good_dist = arr1(&[0.0, 0.1, 0.0, 0.3, 0.075]);
        let eval_dist = arr1(&[0.0, 0.2, 0.15, 0.0, 0.15]);

        let cost = kl_divergence(good_dist.view(), eval_dist.view());

        assert_approx_eq!(f32, cost, 4.300_221_4);
    }
}
