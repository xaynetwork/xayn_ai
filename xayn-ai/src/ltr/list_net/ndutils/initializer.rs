use std::f32::consts::SQRT_2;

use ndarray::{Array2, Dimension, IntoDimension, Ix, Ix2};
use rand::Rng;
use rand_distr::{Distribution, Normal, Uniform};

/// He-Normal Initializer
///
/// Weights for layer `j` are sampled from following normal distribution:
///
/// ```ascii
/// W_j ~ N(μ=0, σ²=2/n_j)
/// ```
///
/// Where `n_j` is the number of input units of this layer.
/// This means for us `n_j` is the the number of rows of `W_j`.
///
/// Furthermore as we want to avoid exceedingly large values
/// we truncate the normal distribution at 2σ.
///
/// Source:
///
/// - Website: https://www.cv-foundation.org/openaccess/content_iccv_2015/html/He_Delving_Deep_into_ICCV_2015_paper.html
/// - Pdf: https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf
pub fn he_normal_weights_init<R>(rng: &mut R, dim: impl IntoDimension<Dim = Ix2>) -> Array2<f32>
where
    R: Rng + ?Sized,
{
    let dim = dim.into_dimension();
    let nr_rows = dim[0];

    // Avoids problems with by-0 division.
    if nr_rows == 0 {
        return Array2::zeros(dim);
    }

    let std_dev = SQRT_2 / (nr_rows as f32).sqrt();
    let dist = Normal::new(0., std_dev).unwrap();
    let limit = 2. * std_dev;

    Array2::from_shape_simple_fn(dim, || loop {
        let res = dist.sample(rng);
        if -limit <= res && res <= limit {
            break res;
        }
    })
}

/// He-Uniform Initializer
///
/// Weights for layer `j` are sampled from a uniform distribution over
/// `[-√(6/n_j); √(6/n_j)]`.
///
/// Where `n_j` is the number of input units of this layer.
/// This means for us `n_j` is the the number of rows of `W_j`.
///
/// Source:
///
/// - Website: https://www.cv-foundation.org/openaccess/content_iccv_2015/html/He_Delving_Deep_into_ICCV_2015_paper.html
/// - Pdf: https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf
pub fn he_uniform_weights_init<R>(rng: &mut R, dim: impl IntoDimension<Dim = Ix2>) -> Array2<f32>
where
    R: Rng + ?Sized,
{
    let dim = dim.into_dimension();
    let nr_rows = dim[0];

    // Avoids problems with by-0 division.
    if nr_rows == 0 {
        return Array2::zeros(dim);
    }

    let limit = (6.0 / nr_rows as f32).sqrt();
    let dist = Uniform::new_inclusive(-limit, limit);

    Array2::from_shape_simple_fn(dim, || dist.sample(rng))
}

/// Glorot-Normal Initializer
///
/// Weights for layer `j` are sampled from following normal distribution:
///
/// ```ascii
/// W_j ~ N(μ=0, σ²=2/(n_j + o_j))
/// ```
///
/// Where `n_j` is the number of input units of this layer.
/// This means for us `n_j` is the the number of rows of `W_j`.
/// And where `o_j` is the number of output units of this layer.
/// This means for us `o_j` is the the number of columns of `W_j`.
///
/// Furthermore as we want to avoid exceedingly large values
/// we truncate the normal distribution at 2σ.
///
/// Source:
///
/// - Website: https://proceedings.mlr.press/v9/glorot10a.html
/// - Pdf: https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
pub fn glorot_normal_weights_init<R>(rng: &mut R, dim: impl IntoDimension<Dim = Ix2>) -> Array2<f32>
where
    R: Rng + ?Sized,
{
    let dim = dim.into_dimension();
    let in_plus_out: Ix = dim.slice().iter().sum();

    // Avoids problems with by-0 division.
    if in_plus_out == 0 {
        return Array2::zeros(dim);
    }

    let std_dev = SQRT_2 / (in_plus_out as f32).sqrt();
    let dist = Normal::new(0., std_dev).unwrap();
    let limit = 2. * std_dev;

    Array2::from_shape_simple_fn(dim, || loop {
        let res = dist.sample(rng);
        if -limit <= res && res <= limit {
            break res;
        }
    })
}

/// Glorot-Uniform Initializer
///
/// Weights for layer `j` are sampled from a uniform distribution over
/// `[-√(6/(n_j+o_j)); √(6/(n_j+o_j))]`.
///
/// Where `n_j` is the number of input units of this layer.
/// This means for us `n_j` is the the number of rows of `W_j`.
/// And where `o_j` is the number of output units of this layer.
/// This means for us `o_j` is the the number of columns of `W_j`.
///
/// Source:
///
/// - Website: https://proceedings.mlr.press/v9/glorot10a.html
/// - Pdf: https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
pub fn glorot_uniform_weights_init<R>(
    rng: &mut R,
    dim: impl IntoDimension<Dim = Ix2>,
) -> Array2<f32>
where
    R: Rng + ?Sized,
{
    let dim = dim.into_dimension();
    let in_plus_out: Ix = dim.slice().iter().sum();

    // Avoids problems with by-0 division.
    if in_plus_out == 0 {
        return Array2::zeros(dim);
    }

    let limit = (6.0 / in_plus_out as f32).sqrt();
    let dist = Uniform::new_inclusive(-limit, limit);

    Array2::from_shape_simple_fn(dim, || dist.sample(rng))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_he_normal_weight_init_zero_dimensions() {
        let mut rng = rand::thread_rng();
        assert_eq!(
            he_normal_weights_init(&mut rng, (0, 200)).shape(),
            &[0, 200]
        );
        assert_eq!(
            he_normal_weights_init(&mut rng, (300, 0)).shape(),
            &[300, 0]
        );
        assert_eq!(he_normal_weights_init(&mut rng, (0, 0)).shape(), &[0, 0]);
    }

    #[test]
    fn test_he_normal_weight_init() {
        let mut rng = rand::thread_rng();
        let weights = he_normal_weights_init(&mut rng, (300, 200));

        assert_eq!(weights.shape(), &[300, 200]);

        let std = SQRT_2 / 300f32.sqrt();
        let limit = 2. * std;
        let mut c_1std = 0;
        let mut c_2std = 0;
        for &w in weights.iter() {
            assert!(
                -limit <= w && w <= limit,
                "out of bound weight: {} <= {} <= {}",
                -limit,
                w,
                limit
            );
            if -std <= w && w <= std {
                c_1std += 1;
            } else {
                c_2std += 1;
            }
        }

        let nr_weights = (300 * 200) as f32;
        let prob_1std = c_1std as f32 / nr_weights;
        let prob_2std = c_2std as f32 / nr_weights;

        // Probabilities of a weight being in +-1std or +-2std corrected
        // wrt. the truncating of values outside of +-2std. We accept this
        // to be true if the found percentage is around +-5% of the expected
        // percentage.
        assert_approx_eq!(f32, prob_1std, 0.715_232_8, epsilon = 0.05);
        assert_approx_eq!(f32, prob_2std, 0.284_767_2, epsilon = 0.05);
    }

    #[test]
    fn test_he_uniform_weight_init_zero_dimensions() {
        let mut rng = rand::thread_rng();
        assert_eq!(
            he_uniform_weights_init(&mut rng, (0, 200)).shape(),
            &[0, 200]
        );
        assert_eq!(
            he_uniform_weights_init(&mut rng, (300, 0)).shape(),
            &[300, 0]
        );
        assert_eq!(he_uniform_weights_init(&mut rng, (0, 0)).shape(), &[0, 0]);
    }

    #[test]
    fn test_he_uniform_weight_init() {
        let mut rng = rand::thread_rng();
        let weights = he_uniform_weights_init(&mut rng, (300, 200));

        assert_eq!(weights.shape(), &[300, 200]);
        assert_approx_eq!(f32, weights.mean().unwrap(), 0.0, epsilon = 0.0016);

        let limit = (6.0f32 / 300.0).sqrt();
        let half_limit = limit / 2.0;
        let buckets = &mut [0, 0, 0, 0];

        for &w in weights.iter() {
            assert!(w <= limit);
            assert!(w >= -limit);
            let bucket_idx = if w.is_sign_positive() {
                if w <= half_limit {
                    0
                } else {
                    1
                }
            } else if w >= -half_limit {
                2
            } else {
                3
            };
            buckets[bucket_idx] += 1;
        }
        let count = weights.len() as f32;
        assert_approx_eq!(f32, buckets[0] as f32 / count, 0.25, epsilon = 0.05);
        assert_approx_eq!(f32, buckets[1] as f32 / count, 0.25, epsilon = 0.05);
        assert_approx_eq!(f32, buckets[2] as f32 / count, 0.25, epsilon = 0.05);
        assert_approx_eq!(f32, buckets[3] as f32 / count, 0.25, epsilon = 0.05);
    }

    #[test]
    fn test_glorot_normal_weight_init_zero_dimensions() {
        let mut rng = rand::thread_rng();
        assert_eq!(
            glorot_normal_weights_init(&mut rng, (0, 200)).shape(),
            &[0, 200]
        );
        assert_eq!(
            glorot_normal_weights_init(&mut rng, (300, 0)).shape(),
            &[300, 0]
        );
        assert_eq!(
            glorot_normal_weights_init(&mut rng, (0, 0)).shape(),
            &[0, 0]
        );
    }

    #[test]
    fn test_glorot_normal_weight_init() {
        let mut rng = rand::thread_rng();
        let weights = glorot_normal_weights_init(&mut rng, (300, 200));

        assert_eq!(weights.shape(), &[300, 200]);

        let std = SQRT_2 / 500f32.sqrt();
        let limit = 2. * std;
        let mut c_1std = 0;
        let mut c_2std = 0;
        for &w in weights.iter() {
            assert!(
                -limit <= w && w <= limit,
                "out of bound weight: {} <= {} <= {}",
                -limit,
                w,
                limit
            );
            if -std <= w && w <= std {
                c_1std += 1;
            } else {
                c_2std += 1;
            }
        }

        let nr_weights = (300 * 200) as f32;
        let prob_1std = c_1std as f32 / nr_weights;
        let prob_2std = c_2std as f32 / nr_weights;

        // Probabilities of a weight being in +-1std or +-2std corrected
        // wrt. the truncating of values outside of +-2std. We accept this
        // to be true if the found percentage is around +-5% of the expected
        // percentage.
        assert_approx_eq!(f32, prob_1std, 0.715_232_8, epsilon = 0.05);
        assert_approx_eq!(f32, prob_2std, 0.284_767_2, epsilon = 0.05);
    }

    #[test]
    fn test_glorot_uniform_weight_init_zero_dimensions() {
        let mut rng = rand::thread_rng();
        assert_eq!(
            glorot_uniform_weights_init(&mut rng, (0, 200)).shape(),
            &[0, 200]
        );
        assert_eq!(
            glorot_uniform_weights_init(&mut rng, (300, 0)).shape(),
            &[300, 0]
        );
        assert_eq!(
            glorot_uniform_weights_init(&mut rng, (0, 0)).shape(),
            &[0, 0]
        );
    }

    #[test]
    fn test_glorot_uniform_weight_init() {
        let mut rng = rand::thread_rng();
        let weights = glorot_uniform_weights_init(&mut rng, (300, 200));

        assert_eq!(weights.shape(), &[300, 200]);
        assert_approx_eq!(f32, weights.mean().unwrap(), 0.0, epsilon = 0.0016);

        let limit = (6.0f32 / 500.0).sqrt();
        let half_limit = limit / 2.0;
        let buckets = &mut [0, 0, 0, 0];

        for &w in weights.iter() {
            assert!(w <= limit);
            assert!(w >= -limit);
            let bucket_idx = if w.is_sign_positive() {
                if w <= half_limit {
                    0
                } else {
                    1
                }
            } else if w >= -half_limit {
                2
            } else {
                3
            };
            buckets[bucket_idx] += 1;
        }
        let count = weights.len() as f32;
        assert_approx_eq!(f32, buckets[0] as f32 / count, 0.25, epsilon = 0.05);
        assert_approx_eq!(f32, buckets[1] as f32 / count, 0.25, epsilon = 0.05);
        assert_approx_eq!(f32, buckets[2] as f32 / count, 0.25, epsilon = 0.05);
        assert_approx_eq!(f32, buckets[3] as f32 / count, 0.25, epsilon = 0.05);
    }
}
