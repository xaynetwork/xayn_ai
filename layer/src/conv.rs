use ndarray::{Array1, Array3, ArrayBase, Data, Ix3, s};

use crate::utils::IncompatibleMatrices;

// not transposed, zero output padding
pub struct Conv1D {
    weights: Array3<f32>,
    bias: Option<Array1<f32>>,

    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
}

impl Conv1D {
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
        assert!(
            groups <= weights_shape[0],
            "Given groups={}, expected weight to be at least {} at dimension 0, but got weight of size {:?} instead",
            groups,
            groups,
            weights_shape,
        );
        assert_eq!(
            weights_shape[0] % groups,
            0,
            "Given groups={}, expected weight to be divisible by {} at dimension 0, but got weight of size {:?} instead",
            groups,
            groups, 
            weights_shape,
        );
        if let Some(ref bias) = bias {
            assert!(!bias.is_empty());
            let bias_shape = bias.shape();
            assert_eq!(
                bias_shape[0],
                weights_shape[0],
                "Given weight of size {:?}, expected bias to be 1-dimensional with {} elements, but got bias of size {:?} instead",
                weights_shape,
                weights_shape[0],
                bias_shape,
            );
        }

        Ok(Self {
            weights,
            bias,
            stride,
            padding,
            dilation,
            groups,
        })
    }

    // https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/Convolution.cpp#L820
    pub fn run<S>(&self, input: ArrayBase<S, Ix3>) -> Array3<f32>
    where
        S: Data<Elem = f32>,
    {
        let input_shape = input.shape();
        let weights_shape = self.weights.shape();
        assert_eq!(
            input_shape[1],
            weights_shape[1] * self.groups,
            "Given groups={}, weight of size {:?}, expected input {:?} to have {} channels, but got {} channels instead",
            self.groups,
            weights_shape,
            input_shape, 
            weights_shape[1] * self.groups,
            input_shape[1],
        );
        let input_size = input_shape[2] + 2 * self.padding;
        let kernel_size = self.dilation * (weights_shape[2] - 1) + 1;
        // d * (s2 - 1) + 1 - s2 = (d - 1) * s2 - d + 1 = (d - 1) * s2 - (d - 1) = (d - 1) * (s2 - 1) >= 0
        // => input_size >= kernel_size >= weights_shape[2]
        assert!(
            input_size >= kernel_size,
            "Calculated padded input size per channel: {}. Kernel size: {}. Kernel size can't be greater than actual input size.",
            input_size,
            kernel_size,
        );

        let output_size = (input_size - kernel_size) / self.stride + 1;
        if input.is_empty() {
            return Array3::zeros([0, self.weights.shape()[0], output_size]);
        }

        // view weights and input as 4D via reshape(shape0, shape1, 1, shape2)

        if self.groups > 1 {
            unimplemented!("https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/Convolution.cpp#L1032-L1041");
        }

        // https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/Convolution.cpp#L1070
        if self.dilation > 1 {
            unimplemented!("https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/NaiveDilatedConvolution.cpp#L434");
        } else {
            let _kernel_height = 1;
            let kernel_width = weights_shape[2];
            let _pad_height = 0;
            let pad_width = self.padding;
            let _stride_height = 1;
            let stride_width = self.stride;

            let n_input_plane = input_shape[1];
            let _input_height = 1;
            let input_width = input_shape[2];
            let n_output_plane = weights_shape[0];
            let _output_height = 1;
            let output_width = output_size;
            let batch_size = input_shape[0];

            // TODO: reshape only once in `new`
            let weights = self.weights.view().into_shape((weights_shape[0], weights_shape[1] * weights_shape[2])).unwrap();

            // TODO: move this into batch_size loop
            let mut finput = if kernel_width == 1 && stride_width == 1 && pad_width == 0 {
                // output_width == input_shape[2]
                input.to_owned().into_shape((input_shape[0], input_shape[1], output_width)).unwrap()
            } else {
                Array3::zeros((input_shape[0], input_shape[1] * kernel_width, output_width))
            };
            
            // https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/ConvolutionMM2d.cpp#L140
            let mut output = Array3::<f32>::zeros([
                batch_size,
                n_output_plane, /* ws0 */
                output_width,   /* is2 - ws2 + 1 */
            ]);
            for i in 0..batch_size {
                let input = input.slice(s![i, .., ..]);
                let mut finput = finput.slice_mut(s![i, .., ..]);
                let mut output = output.slice_mut(s![i, .., ..]);
                if kernel_width != 1 || stride_width != 1 || pad_width != 0 {
                    let input = input.as_slice().unwrap();
                    let finput = finput.as_slice_mut().unwrap();

                    // https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cpu/Unfold2d.cpp#L160
                    for k in 0..n_input_plane * kernel_width {
                        let nip = k / kernel_width;
                        let rest = k % kernel_width;
                        let kh = rest / kernel_width;
                        let kw = rest % kernel_width;
    
                        if pad_width > 0 {
                            unimplemented!("https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cpu/Unfold2d.cpp#L160");
                        } else if stride_width == 1 {
                            let src = nip*input_width + kh*input_width + kw;
                            let dst = nip * kernel_width * output_width + kh * kernel_width * output_width + kw * output_width;
                            finput[dst..dst+output_width].copy_from_slice(&input[src..src+output_width]);
                        } else {
                            unimplemented!("https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cpu/Unfold2d.cpp#L160");
                        }
                    }
                }
                // multiply weights with input[i, ..] and assign to output[i, ..]
                // weights: (ws0, ws1 * ws2), input: (is0, is1 * ws2, is2 - ws2 + 1) => ws1 == is1 is guaranteed above
                output.assign(&weights.dot(&finput));
            }

            if let Some(ref bias) = self.bias {
                // view 1D bias as 3D via reshape(1, shape0, 1)
                output = output + bias.view().into_shape((1, bias.len(), 1)).unwrap();
            }

            output
        }


    }
}

#[cfg(test)]
mod tests {
    use ndarray::arr3;

    use test_utils::assert_approx_eq;
    use super::*;

    #[test]
    fn test_conv1d() {
        let weights = (0..18).map(|e| e as f32).collect::<Array1<f32>>().into_shape((2, 3, 3)).unwrap();
        let input = (0..30).map(|e| e as f32).collect::<Array1<f32>>().into_shape((2, 3, 5)).unwrap();
        let output = Conv1D::new(weights, None, 1, 0, 1, 1).unwrap().run(input);
        let expected = arr3(&[[[ 312.,  348.,  384.],
            [ 798.,  915., 1032.]],
           [[ 852.,  888.,  924.],
            [2553., 2670., 2787.]]]);
        assert_approx_eq!(f32, output, expected);
    }
}
