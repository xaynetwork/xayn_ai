use displaydoc::Display;
use ndarray::{
    azip,
    Array1,
    Array2,
    Array3,
    ArrayBase,
    ArrayView2,
    ArrayView3,
    Data,
    ErrorKind,
    Ix2,
    Ix3,
    ShapeError,
};
use thiserror::Error;

use crate::{
    activation::ActivationFunction,
    io::{BinParamsWithScope, LoadingLayerFailed},
};

/// A 1-dimensional convolutional layer.
///
/// The layer can be configured via:
/// - stride: Strides of the convolving kernel, must be positive.
/// - padding: Zero-padding of the input (unimplemented for `padding != 0`).
/// - dilation: Spacing between the convolving kernel elements, must be positive (unimplemented
/// for `dilation != 1`).
/// - groups: Grouping of the input, must be positive and must divide the `channel_in_size`
/// (unimplemented for `groups != 1`).
#[derive(Debug)]
pub struct Conv1D<AF>
where
    AF: ActivationFunction<f32>,
{
    // params
    weights: Array2<f32>,
    bias: Array3<f32>,
    activation: AF,

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

/// Convolutional layer errors.
#[derive(Debug, Display, Error)]
pub enum ConvError {
    /// Invalid shapes.
    #[displaydoc("{0}")]
    Shape(#[from] ShapeError),

    /// Stride must be positive
    Stride,

    /// Dilation must be positive
    Dilation,

    /// Groups must be positive
    Groups,

    /// Channel out size must be divisible by groups
    ChannelOutSize,

    /// Loading failed.
    #[displaydoc("{0}")]
    Load(#[from] LoadingLayerFailed),
}

impl<AF> Conv1D<AF>
where
    AF: ActivationFunction<f32>,
{
    /// Creates a 1-dimensional convolutional layer.
    ///
    /// The weights are of shape `(channel_out_size, channel_in_size/groups, kernel_size)` and the
    /// bias is of shape `(channel_out_size,)`.
    pub fn new(
        weights: Array3<f32>,
        bias: Array1<f32>,
        activation: AF,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Self, ConvError> {
        if weights.is_empty() {
            return Err(ShapeError::from_kind(ErrorKind::IncompatibleShape).into());
        }
        if weights.iter().any(|w| w.is_nan() || w.is_infinite())
            || bias.iter().any(|b| b.is_nan() || b.is_infinite())
        {
            return Err(LoadingLayerFailed::InvalidParams.into());
        }

        if stride == 0 {
            return Err(ConvError::Stride);
        }
        if dilation == 0 {
            return Err(ConvError::Dilation);
        }
        if groups == 0 {
            return Err(ConvError::Groups);
        }

        let weights_shape = weights.shape();
        let channel_out_size = weights_shape[0];
        let channel_grouped_size = weights_shape[1];
        let kernel_size = weights_shape[2];
        let dilated_kernel_size = dilation * (kernel_size - 1) + 1;
        let bias_size = bias.len();

        if channel_out_size % groups != 0 {
            return Err(ConvError::ChannelOutSize);
        }
        if channel_out_size != bias_size {
            return Err(ShapeError::from_kind(ErrorKind::IncompatibleShape).into());
        }

        let weights = weights.into_shape((channel_out_size, channel_grouped_size * kernel_size))?;
        let bias = bias.into_shape((1, bias_size, 1))?;

        Ok(Self {
            weights,
            bias,
            activation,

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

    /// Loads a 1-dimensional convolutional layer from a binary.
    pub fn load(
        mut params: BinParamsWithScope,
        activation: AF,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Self, ConvError> {
        let weights = params
            .take("weights")
            .map_err::<LoadingLayerFailed, _>(Into::into)?;
        let bias = params
            .take("bias")
            .map_err::<LoadingLayerFailed, _>(Into::into)?;

        Self::new(weights, bias, activation, stride, padding, dilation, groups)
    }

    /// Gets the weights.
    ///
    /// The weights are of shape `(channel_out_size, channel_grouped_size * kernel_size)`.
    pub fn weights(&self) -> ArrayView2<f32> {
        self.weights.view()
    }

    /// Gets the bias.
    ///
    /// The bias is of shape `(1, bias_size, 1)`.
    pub fn bias(&self) -> ArrayView3<f32> {
        self.bias.view()
    }

    /// Gets the first dimension of the original weights.
    pub fn channel_out_size(&self) -> usize {
        self.channel_out_size
    }

    /// Gets the second dimension of the original weights.
    pub fn channel_grouped_size(&self) -> usize {
        self.channel_grouped_size
    }

    /// Gets the third dimension of the original weights.
    pub fn kernel_size(&self) -> usize {
        self.kernel_size
    }

    /// Computes the forward pass of the input through the layer.
    ///
    /// The input is of shape `(batch_size, channel_in_size, input_size)` and the output is of shape
    /// `(batch_size, channel_out_size, output_size)`.
    pub fn run(
        &self,
        input: ArrayBase<impl Data<Elem = f32>, Ix3>,
    ) -> Result<Array3<f32>, ConvError> {
        let input_shape = input.shape();
        let batch_size = input_shape[0];
        let channel_in_size = input_shape[1];
        let padded_input_size = input_shape[2] + 2 * self.padding;

        if channel_in_size != self.channel_grouped_size * self.groups {
            return Err(ShapeError::from_kind(ErrorKind::IncompatibleShape).into());
        }
        if padded_input_size < self.dilated_kernel_size {
            return Err(ShapeError::from_kind(ErrorKind::IncompatibleShape).into());
        }

        let output_size = (padded_input_size - self.dilated_kernel_size) / self.stride + 1;
        if input.is_empty() {
            return Ok(Array3::zeros([0, self.channel_out_size, output_size]));
        }

        if self.groups > 1 {
            unimplemented!("ATen/native/Convolution");
        }
        if self.dilation > 1 {
            unimplemented!("ATen/native/NaiveDilatedConvolution");
        }

        let mut output = Array3::zeros([batch_size, self.channel_out_size, output_size]);
        azip!((mut output in output.outer_iter_mut(), input in input.outer_iter()) {
            let input = self.unfold(input, output_size);
            output.assign(&self.weights.dot(&input));
        });
        let output = self.activation.apply_to(output + &self.bias);

        Ok(output)
    }

    /// Unfolds the input for the kernel.
    ///
    /// The input is a slice of a single batch sample. The unfolded input is of shape
    /// `(channel_in_size * kernel_size, output_size)`.
    fn unfold(
        &self,
        input: ArrayBase<impl Data<Elem = f32>, Ix2>,
        output_size: usize,
    ) -> Array2<f32> {
        if self.padding > 0 {
            unimplemented!("ATen/native/cpu/Unfold2d");
        }

        Array2::from_shape_fn(
            (input.shape()[0] * self.kernel_size, output_size),
            |(i, j)| input[[i / self.kernel_size, i % self.kernel_size + j * self.stride]],
        )
    }
}

#[cfg(test)]
mod tests {
    use ndarray::arr3;

    use super::*;
    use crate::activation::Linear;
    use test_utils::assert_approx_eq;

    #[test]
    fn test_conv1d_nonsingleton_kernel() {
        let weights = Array1::range(0., 18., 1.).into_shape((2, 3, 3)).unwrap();
        let bias = Array1::zeros(2);
        let input = Array1::range(0., 30., 1.).into_shape((2, 3, 5)).unwrap();
        let output = Conv1D::new(weights, bias, Linear, 1, 0, 1, 1)
            .unwrap()
            .run(input)
            .unwrap();
        let expected = arr3(&[
            [[312., 348., 384.], [798., 915., 1032.]],
            [[852., 888., 924.], [2553., 2670., 2787.]],
        ]);
        assert_approx_eq!(f32, output, expected);
    }

    #[test]
    fn test_conv1d_singleton_kernel() {
        let weights = Array1::range(0., 6., 1.).into_shape((2, 3, 1)).unwrap();
        let bias = Array1::zeros(2);
        let input = Array1::range(0., 30., 1.).into_shape((2, 3, 5)).unwrap();
        let output = Conv1D::new(weights, bias, Linear, 1, 0, 1, 1)
            .unwrap()
            .run(input)
            .unwrap();
        let expected = arr3(&[
            [[25., 28., 31., 34., 37.], [70., 82., 94., 106., 118.]],
            [[70., 73., 76., 79., 82.], [250., 262., 274., 286., 298.]],
        ]);
        assert_approx_eq!(f32, output, expected);
    }

    #[test]
    fn test_conv1d_bias() {
        let weights = Array1::range(0., 18., 1.).into_shape((2, 3, 3)).unwrap();
        let bias = Array1::range(0., 2., 1.);
        let input = Array1::range(0., 30., 1.).into_shape((2, 3, 5)).unwrap();
        let output = Conv1D::new(weights, bias, Linear, 1, 0, 1, 1)
            .unwrap()
            .run(input)
            .unwrap();
        let expected = arr3(&[
            [[312., 348., 384.], [799., 916., 1033.]],
            [[852., 888., 924.], [2554., 2671., 2788.]],
        ]);
        assert_approx_eq!(f32, output, expected);
    }

    #[test]
    fn test_conv1d_stride_with_nonsingleton_kernel() {
        let weights = Array1::range(0., 18., 1.).into_shape((2, 3, 3)).unwrap();
        let bias = Array1::zeros(2);
        let input = Array1::range(0., 30., 1.).into_shape((2, 3, 5)).unwrap();
        let output = Conv1D::new(weights, bias, Linear, 2, 0, 1, 1)
            .unwrap()
            .run(input)
            .unwrap();
        let expected = arr3(&[
            [[312., 384.], [798., 1032.]],
            [[852., 924.], [2553., 2787.]],
        ]);
        assert_approx_eq!(f32, output, expected);
    }

    #[test]
    fn test_conv1d_stride_with_singleton_kernel() {
        let weights = Array1::range(0., 6., 1.).into_shape((2, 3, 1)).unwrap();
        let bias = Array1::zeros(2);
        let input = Array1::range(0., 30., 1.).into_shape((2, 3, 5)).unwrap();
        let output = Conv1D::new(weights, bias, Linear, 2, 0, 1, 1)
            .unwrap()
            .run(input)
            .unwrap();
        let expected = arr3(&[
            [[25., 31., 37.], [70., 94., 118.]],
            [[70., 76., 82.], [250., 274., 298.]],
        ]);
        assert_approx_eq!(f32, output, expected);
    }

    #[test]
    #[should_panic(expected = "not implemented: ATen/native/cpu/Unfold2d")]
    fn test_conv1d_padding_with_nonsingleton_kernel() {
        let weights = Array1::range(0., 18., 1.).into_shape((2, 3, 3)).unwrap();
        let bias = Array1::zeros(2);
        let input = Array1::range(0., 30., 1.).into_shape((2, 3, 5)).unwrap();
        let output = Conv1D::new(weights, bias, Linear, 1, 1, 1, 1)
            .unwrap()
            .run(input)
            .unwrap();
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
    #[should_panic(expected = "not implemented: ATen/native/cpu/Unfold2d")]
    fn test_conv1d_padding_with_singleton_kernel() {
        let weights = Array1::range(0., 6., 1.).into_shape((2, 3, 1)).unwrap();
        let bias = Array1::zeros(2);
        let input = Array1::range(0., 30., 1.).into_shape((2, 3, 5)).unwrap();
        let output = Conv1D::new(weights, bias, Linear, 1, 1, 1, 1)
            .unwrap()
            .run(input)
            .unwrap();
        let expected = arr3(&[
            [
                [0., 25., 28., 31., 34., 37., 0.],
                [0., 70., 82., 94., 106., 118., 0.],
            ],
            [
                [0., 70., 73., 76., 79., 82., 0.],
                [0., 250., 262., 274., 286., 298., 0.],
            ],
        ]);
        assert_approx_eq!(f32, output, expected);
    }

    #[test]
    #[should_panic(expected = "not implemented: ATen/native/NaiveDilatedConvolution")]
    fn test_conv1d_dilation() {
        let weights = Array1::range(0., 18., 1.).into_shape((2, 3, 3)).unwrap();
        let bias = Array1::zeros(2);
        let input = Array1::range(0., 30., 1.).into_shape((2, 3, 5)).unwrap();
        let output = Conv1D::new(weights, bias, Linear, 1, 0, 2, 1)
            .unwrap()
            .run(input)
            .unwrap();
        let expected = arr3(&[[[354.], [921.]], [[894.], [2676.]]]);
        assert_approx_eq!(f32, output, expected);
    }

    #[test]
    #[should_panic(expected = "not implemented: ATen/native/Convolution")]
    fn test_conv1d_groups() {
        let weights = Array1::range(0., 24., 1.).into_shape((2, 4, 3)).unwrap();
        let bias = Array1::zeros(2);
        let input = Array1::range(0., 80., 1.).into_shape((2, 8, 5)).unwrap();
        let output = Conv1D::new(weights, bias, Linear, 1, 0, 1, 2)
            .unwrap()
            .run(input)
            .unwrap();
        let expected = arr3(&[
            [[794., 860., 926.], [6218., 6428., 6638.]],
            [[3434., 3500., 3566.], [14618., 14828., 15038.]],
        ]);
        assert_approx_eq!(f32, output, expected);
    }
}
