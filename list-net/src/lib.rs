//! ListNet implementation using NdArray.
//!
//!
//!
//!
//! # Architecture
//!
//! ListNet has a comparable simple architecture, having following
//! stack of layers:
//!
//! 0. Kind Input: Type f32; Shape (nr_docs, nr_of_features);
//! 1. Kind Dense: reLu; 48units; with bias  [nr_of_features => 48]
//! 2. Kind Dense: reLu;  8units; with bias  [48 => 8]
//! 3. Kind Dense: linear; 1units; with bias [8 => 1]
//! 4. Kind Flatten: -- [(nr_docs, 1) => (nr_docs,)]
//! 5. Kind SoftMax: nr_docs units [nr_docs => nr_docs, but sum == 1]

use std::{io::Read, path::Path};

use ndutils::io::LoadingBinParamsFailed;
use thiserror::Error;

use ndarray::{Array1, Array2, Dimension, IntoDimension, Ix, Ix2};
use ndlayers::{
    activation::{Linear, Relu, Softmax},
    Dense,
    IncompatibleMatrices,
    LoadingDenseFailed,
};

use crate::ndutils::io::BinParams;

#[macro_use]
mod utils;

mod ndlayers;
mod ndutils;

/// ListNet implementations.
///
/// This chains following feed forward layers:
///
/// The input is currently fixed to 10 document with 50 features each
/// in the shape `(10, 50)`
///
/// 1. Dense with 48 units, bias and reLu activation function (out shape: `(10, 48)`)
/// 2. Dense with 8 units, bias and reLu activation function (out shape: `(10, 8)`)
/// 3. Dense with 1 unit, bias and linear activation function (out shape: `(10, 1)`)
/// 4. Flattening to the number of input documents. (out shape `(10,)`)
/// 5. Dense with `nr_of_input_documents` (10) units, bias and softmax activation function (out shape `(10,)`)
pub struct ListNet {
    dense_1: Dense<Relu>,
    dense_2: Dense<Relu>,
    scores: Dense<Linear>,
    scores_prop_dist: Dense<Softmax>,
}

impl ListNet {
    /// Currently input is fixed to (10, 50)
    const INPUT_SHAPE: [Ix; 2] = [10, 50];

    /// Load list net from file at given path.
    pub fn load_from_file(path: impl AsRef<Path>) -> Result<Self, LoadingListNetFailed> {
        let params = BinParams::load_from_file(path)?;
        Self::load(params)
    }

    /// Load list net from byte reader.
    pub fn load_from_source(params_source: impl Read) -> Result<Self, LoadingListNetFailed> {
        let params = BinParams::load(params_source)?;
        Self::load(params)
    }

    /// Load list net from `BinParams`.
    pub(crate) fn load(mut params: BinParams) -> Result<Self, LoadingListNetFailed> {
        let mut params = params.with_scope("ltr/v1");

        let dense_1 = Dense::load(params.with_scope("dense_1"), Relu::default())?;
        let dense_1_out_shape = dense_1.check_in_out_shapes(Self::INPUT_SHAPE.into_dimension())?;

        let dense_2 = Dense::load(params.with_scope("dense_2"), Relu::default())?;
        let dense_2_out_shape = dense_2.check_in_out_shapes(dense_1_out_shape)?;

        let scores = Dense::load(params.with_scope("scores"), Linear::default())?;
        let scores_out_shape = scores.check_in_out_shapes(dense_2_out_shape)?;
        let flattened_shape = [scores_out_shape.size()].into_dimension();

        let scores_prop_dist =
            Dense::load(params.with_scope("scores_prop_dist"), Softmax::default())?;
        let scores_prop_dist_out_shape = scores_prop_dist.check_in_out_shapes(flattened_shape)?;

        if scores_prop_dist_out_shape.clone().into_pattern() == 10 {
            Ok(Self {
                dense_1,
                dense_2,
                scores,
                scores_prop_dist,
            })
        } else {
            Err(IncompatibleMatrices {
                name_left: "scores_prop_dist/output",
                shape_left: scores_prop_dist_out_shape.into_dyn(),
                name_right: "list_net/output",
                shape_right: (10,).into_dimension().into_dyn(),
                hint: "expected scores_prop_dist output shape to be equal to (10,)",
            }
            .into())
        }
    }

    /// Runs List net on the input.
    ///
    /// The input is a 2 dimensional array
    /// with the shape `(number_of_documents, number_of_feature_per_document)`.
    pub fn run(&self, inputs: Array2<f32>) -> Array1<f32> {
        let dense1_out = self.dense_1.run(inputs);
        let dense2_out = self.dense_2.run(dense1_out);
        let scores = self.scores.run(dense2_out);
        let shape: Ix2 = scores.raw_dim();
        debug_assert_eq!(shape[1], 1);
        let scores: Array1<f32> = scores.into_shape((shape[0],)).unwrap();
        self.scores_prop_dist.run(scores)
    }
}

/// Loading list net failed.
#[derive(Debug, Error)]
pub enum LoadingListNetFailed {
    /// Failed to load bin params.
    #[error(transparent)]
    BinParams(#[from] LoadingBinParamsFailed),

    /// Failed to create instance of `Dense`.
    #[error(transparent)]
    Dense(#[from] LoadingDenseFailed),

    /// Parameter configuration error
    #[error(transparent)]
    IncompatibleMatrices(#[from] IncompatibleMatrices),
}

#[cfg(test)]
mod tests {

    use ndarray::arr1;

    use super::*;

    const LIST_NET_BIN_PARAMS_PATH: &str = "../data/ltr_v0000/ltr_v0000.binparams";

    /// A single List-Net Input, cast to shape (10, 50).
    ///
    /// The `arr2` helper is currently only implemented up
    /// to a N of 16. (We need 50.)
    #[rustfmt::skip]
    const SAMPLE_INPUTS: &[f32; 500] = &[
        1.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,0.0,0.283,0.283,0.283,0.283,0.0,0.0,0.283,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.283,0.0,0.0,4.0,0.0,0.0,0.0,0.283,0.0,0.0,2.475_141_5,31.0,12.0,8.0,91.0,2.219_780_2,3.673_913,0.0,0.0,0.0,4.0,0.0,
        2.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,0.0,0.283,0.283,0.283,0.283,0.0,0.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,4.0,0.0,0.0,0.0,0.283,0.0,0.0,2.475_141_5,31.0,12.0,8.0,91.0,2.219_780_2,3.673_913,0.0,0.0,0.0,4.0,0.0,
        3.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,0.0,0.283,0.283,0.283,0.283,0.0,0.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,4.0,0.0,0.0,0.0,0.283,0.0,0.0,2.475_141_5,31.0,12.0,8.0,91.0,2.219_780_2,3.673_913,0.0,0.0,0.0,4.0,0.0,
        4.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,0.0,0.283,0.283,0.283,0.283,0.0,0.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,4.0,0.0,0.0,0.0,0.283,0.0,0.0,2.475_141_5,31.0,12.0,8.0,91.0,2.219_780_2,3.673_913,0.0,0.0,0.0,4.0,0.0,
        5.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,0.0,0.283,0.283,0.283,0.283,0.0,0.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,4.0,0.0,0.0,0.0,0.283,0.0,0.0,2.475_141_5,31.0,12.0,8.0,91.0,2.219_780_2,3.673_913,0.0,0.0,0.0,4.0,0.0,
        6.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,0.0,0.283,0.283,0.283,0.283,0.0,0.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,4.0,0.0,0.0,0.0,0.283,0.0,0.0,2.475_141_5,31.0,12.0,8.0,91.0,2.219_780_2,3.673_913,0.0,0.0,0.0,4.0,0.0,
        7.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,0.0,0.283,0.283,0.283,0.283,0.0,0.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,4.0,0.0,0.0,0.0,0.283,0.0,0.0,2.475_141_5,31.0,12.0,8.0,91.0,2.219_780_2,3.673_913,0.0,0.0,0.0,4.0,0.0,
        8.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,0.0,0.283,0.283,0.283,0.283,0.0,0.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,4.0,0.0,0.0,0.0,0.283,0.0,0.0,2.475_141_5,31.0,12.0,8.0,91.0,2.219_780_2,3.673_913,0.0,0.0,0.0,4.0,0.0,
        9.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,0.0,0.283,0.283,0.283,0.283,0.0,0.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,4.0,0.0,0.0,0.0,0.283,0.0,0.0,2.475_141_5,31.0,12.0,8.0,91.0,2.219_780_2,3.673_913,0.0,0.0,0.0,4.0,0.0,
        10.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,0.0,0.283,0.283,0.283,0.283,0.0,0.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,4.0,0.0,0.0,0.0,0.283,0.0,0.0,2.475_141_5,31.0,12.0,8.0,91.0,2.219_780_2,3.673_913,0.0,0.0,0.0,4.0,0.0,
    ];

    #[test]
    fn test_list_net_end_to_end() {
        let list_net = ListNet::load_from_file(LIST_NET_BIN_PARAMS_PATH).unwrap();

        let inputs = Array1::from(SAMPLE_INPUTS.to_vec())
            .into_shape((10, 50))
            .unwrap();

        let outcome = list_net.run(inputs);

        //FIXME: Check if the values are correct.
        assert_ndarray_eq!(
            f32,
            outcome,
            arr1(&[
                0.30562896,
                0.1503916,
                0.115954444,
                0.093693145,
                0.077109516,
                0.06528697,
                0.056171637,
                0.049953047,
                0.044394825,
                0.04141591,
            ])
        );
    }
}