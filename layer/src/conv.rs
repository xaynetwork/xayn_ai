// TODO: refactor looped assignments as iterator collections

use ndarray::{s, Array1, Array2, Array3, ArrayBase, ArrayView2, Data, Ix3};

use crate::utils::IncompatibleMatrices;

pub struct Conv1D {
    // params
    weights: Array2<f32>,
    bias: Option<Array3<f32>>,

    // configs
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,

    // shapes
    channel_out_size: usize,
    channel_grouped_size: usize,
    kernel_size: usize,
    dilated_kernel_size: usize,
}

impl Conv1D {
    // https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/Convolution.cpp#L515
    pub fn new(
        weights: Array3<f32>,
        bias: Option<Array1<f32>>,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Self, IncompatibleMatrices> {
        assert!(!weights.is_empty());
        assert!(stride > 0, "non-positive stride is not supported");
        assert!(dilation > 0, "non-positive dilation is not supported");
        assert!(groups > 0, "non-positive groups is not supported");

        let weights_shape = weights.shape();
        let channel_out_size = weights_shape[0];
        let channel_grouped_size = weights_shape[1];
        let kernel_size = weights_shape[2];
        let dilated_kernel_size = dilation * (kernel_size - 1) + 1;

        assert!(
            groups <= channel_out_size,
            "Given groups={}, expected weight to be at least {} at dimension 0, but got weight of size {:?} instead",
            groups,
            groups,
            weights_shape,
        );
        assert_eq!(
            channel_out_size % groups,
            0,
            "Given groups={}, expected weight to be divisible by {} at dimension 0, but got weight of size {:?} instead",
            groups,
            groups,
            weights_shape,
        );
        let bias = bias.map(|bias| {
            assert!(!bias.is_empty());
            let bias_size = bias.len();
            assert_eq!(
                bias_size,
                channel_out_size,
                "Given weight of size {:?}, expected bias to be 1-dimensional with {} elements, but got bias of size {:?} instead",
                weights_shape,
                channel_out_size,
                bias_size,
            );
            bias.into_shape((1, bias_size, 1)).unwrap()
        });
        let weights = weights
            .into_shape((channel_out_size, channel_grouped_size * kernel_size))
            .unwrap();

        Ok(Self {
            weights,
            bias,

            stride,
            padding,
            dilation,
            groups,

            channel_out_size,
            channel_grouped_size,
            kernel_size,
            dilated_kernel_size,
        })
    }

    // https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/Convolution.cpp#L820
    // https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/Convolution.cpp#L1070
    pub fn run<S>(&self, input: ArrayBase<S, Ix3>) -> Array3<f32>
    where
        S: Data<Elem = f32>,
    {
        let input_shape = input.shape();
        let batch_size = input_shape[0];
        let channel_in_size = input_shape[1];
        let input_size = input_shape[2];

        assert_eq!(
            channel_in_size,
            self.channel_grouped_size * self.groups,
            "Given groups={}, weight of size {:?}, expected input {:?} to have {} channels, but got {} channels instead",
            self.groups,
            [self.channel_out_size, self.channel_grouped_size, self.kernel_size],
            input_shape,
            self.channel_grouped_size * self.groups,
            channel_in_size,
        );
        let padded_input_size = input_size + 2 * self.padding;
        assert!(
            padded_input_size >= self.dilated_kernel_size,
            "Calculated padded input size per channel: {}. Kernel size: {}. Kernel size can't be greater than actual input size.",
            padded_input_size,
            self.dilated_kernel_size,
        );

        let output_size = (padded_input_size - self.dilated_kernel_size) / self.stride + 1;
        if input.is_empty() {
            return Array3::zeros([0, self.channel_out_size, output_size]);
        }

        if self.groups > 1 {
            unimplemented!("https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/Convolution.cpp#L1032-L1041");
        }
        if self.dilation > 1 {
            unimplemented!("https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/NaiveDilatedConvolution.cpp#L434");
        }

        // https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/ConvolutionMM2d.cpp#L140
        let mut output = Array3::zeros([batch_size, self.channel_out_size, output_size]);
        for i in 0..batch_size {
            let input = input.slice(s![i, .., ..]);
            let mut output = output.slice_mut(s![i, .., ..]);
            if self.kernel_size == 1 && self.stride == 1 && self.padding == 0 {
                output.assign(&self.weights.dot(&input));
            } else {
                let input = self.unfold(input, output_size);
                output.assign(&self.weights.dot(&input));
            }
        }

        if let Some(ref bias) = self.bias {
            output + bias
        } else {
            output
        }
    }

    // https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cpu/Unfold2d.cpp#L160
    fn unfold(&self, input: ArrayView2<f32>, output_size: usize) -> Array2<f32> {
        if self.padding > 0 {
            unimplemented!("https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cpu/Unfold2d.cpp#L191-L243");
        }

        let channel_in_size = input.shape()[0];
        let input_size = input.shape()[1];
        let input = input.as_slice().unwrap();

        let mut unfolded = Array2::zeros((channel_in_size * self.kernel_size, output_size));
        {
            let unfolded = unfolded.as_slice_mut().unwrap();
            for k in 0..channel_in_size * self.kernel_size {
                let nip = k / self.kernel_size;
                let rest = k % self.kernel_size;
                let kh = rest / self.kernel_size;
                let kw = rest % self.kernel_size;

                let src = nip * input_size + kh * input_size + kw;
                let dst = nip * self.kernel_size * output_size
                    + kh * self.kernel_size * output_size
                    + kw * output_size;

                if self.stride > 1 {
                    for j in 0..output_size {
                        unfolded[dst + j] = input[src + j * self.stride];
                    }
                } else {
                    unfolded[dst..dst + output_size]
                        .copy_from_slice(&input[src..src + output_size]);
                }
            }
        }

        unfolded
    }
}

#[cfg(test)]
mod tests {
    use ndarray::arr3;

    use super::*;
    use test_utils::assert_approx_eq;

    // https://github.com/pytorch/pytorch/blob/master/test/cpp/api/functional.cpp#L13-L26
    #[test]
    fn test_conv1d_wide_kernel() {
        let weights = Array1::range(0., 18., 1.).into_shape((2, 3, 3)).unwrap();
        let input = Array1::range(0., 30., 1.).into_shape((2, 3, 5)).unwrap();
        let output = Conv1D::new(weights, None, 1, 0, 1, 1).unwrap().run(input);
        let expected = arr3(&[
            [[312., 348., 384.], [798., 915., 1032.]],
            [[852., 888., 924.], [2553., 2670., 2787.]],
        ]);
        assert_approx_eq!(f32, output, expected);
    }

    #[test]
    fn test_conv1d_single_kernel() {
        let weights = Array1::range(0., 6., 1.).into_shape((2, 3, 1)).unwrap();
        let input = Array1::range(0., 30., 1.).into_shape((2, 3, 5)).unwrap();
        let output = Conv1D::new(weights, None, 1, 0, 1, 1).unwrap().run(input);
        let expected = arr3(&[
            [[25., 28., 31., 34., 37.], [70., 82., 94., 106., 118.]],
            [[70., 73., 76., 79., 82.], [250., 262., 274., 286., 298.]],
        ]);
        assert_approx_eq!(f32, output, expected);
    }

    #[test]
    fn test_conv1d_stride() {
        let weights = Array1::range(0., 18., 1.).into_shape((2, 3, 3)).unwrap();
        let input = Array1::range(0., 30., 1.).into_shape((2, 3, 5)).unwrap();
        let output = Conv1D::new(weights, None, 2, 0, 1, 1).unwrap().run(input);
        let expected = arr3(&[
            [[312., 384.], [798., 1032.]],
            [[852., 924.], [2553., 2787.]],
        ]);
        assert_approx_eq!(f32, dbg!(output), expected);
    }

    #[test]
    #[should_panic(expected = "not implemented: https://github.com/pytorch")]
    fn test_conv1d_padding() {
        let weights = Array1::range(0., 18., 1.).into_shape((2, 3, 3)).unwrap();
        let input = Array1::range(0., 30., 1.).into_shape((2, 3, 5)).unwrap();
        let output = Conv1D::new(weights, None, 1, 1, 1, 1).unwrap().run(input);
        let expected = arr3(&[
            [
                [210., 312., 348., 384., 240.],
                [507., 798., 915., 1032., 699.],
            ],
            [
                [615., 852., 888., 924., 555.],
                [1722., 2553., 2670., 2787., 1824.],
            ],
        ]);
        assert_approx_eq!(f32, output, expected);
    }

    #[test]
    #[should_panic(expected = "not implemented: https://github.com/pytorch")]
    fn test_conv1d_dilation() {
        let weights = Array1::range(0., 18., 1.).into_shape((2, 3, 3)).unwrap();
        let input = Array1::range(0., 30., 1.).into_shape((2, 3, 5)).unwrap();
        let output = Conv1D::new(weights, None, 1, 0, 2, 1).unwrap().run(input);
        let expected = arr3(&[[[354.], [921.]], [[894.], [2676.]]]);
        assert_approx_eq!(f32, output, expected);
    }

    #[test]
    #[should_panic(expected = "not implemented: https://github.com/pytorch")]
    fn test_conv1d_groups() {
        let weights = Array1::range(0., 24., 1.).into_shape((2, 4, 3)).unwrap();
        let input = Array1::range(0., 80., 1.).into_shape((2, 8, 5)).unwrap();
        let output = Conv1D::new(weights, None, 1, 0, 1, 2).unwrap().run(input);
        let expected = arr3(&[
            [[794., 860., 926.], [6218., 6428., 6638.]],
            [[3434., 3500., 3566.], [14618., 14828., 15038.]],
        ]);
        assert_approx_eq!(f32, output, expected);
    }

    #[test]
    fn test_conv1d_bias() {
        let weights = Array1::range(0., 18., 1.).into_shape((2, 3, 3)).unwrap();
        let bias = Array1::range(0., 2., 1.);
        let input = Array1::range(0., 30., 1.).into_shape((2, 3, 5)).unwrap();
        let output = Conv1D::new(weights, Some(bias), 1, 0, 1, 1)
            .unwrap()
            .run(input);
        let expected = arr3(&[
            [[312., 348., 384.], [799., 916., 1033.]],
            [[852., 888., 924.], [2554., 2671., 2788.]],
        ]);
        assert_approx_eq!(f32, output, expected);
    }
}
