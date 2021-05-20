//! ListNet implementation using the NdArray crate.

use std::{io::Read, path::Path};

use ndutils::io::{BinParams, LoadingBinParamsFailed};
use thiserror::Error;

use ndarray::{s, Array1, Array2, ArrayView2, Axis, Dimension, IntoDimension, Ix};
use ndlayers::{
    activation::{Linear, Relu, Softmax},
    Dense,
    IncompatibleMatrices,
    LoadingDenseFailed,
};

mod ndlayers;
mod ndutils;

/// ListNet implementations.
///
/// The underlying ListNet is fixed at 10 documents with 50 features each in the shape `(10, 50)`.
/// But the [`ListNet.run()`] method combines a chunked approach with padding to allow any
/// number of documents as input.
///
/// # Underlying FNN Architecture
///
/// 0. Input shape `(10, 50)`
/// 1. Dense layer with 48 units, bias and reLu activation function (out shape: `(10, 48)`)
/// 2. Dense layer with 8 units, bias and reLu activation function (out shape: `(10, 8)`)
/// 3. Dense layer with 1 unit, bias and linear activation function (out shape: `(10, 1)`)
/// 4. Flattening to the number of input documents. (out shape `(10,)`)
/// 5. Dense with `nr_of_input_documents` (10) units, bias and softmax activation function (out shape `(10,)`)
#[allow(unused)] //TODO tmp
pub struct ListNet {
    dense_1: Dense<Relu>,
    dense_2: Dense<Relu>,
    scores: Dense<Linear>,
    scores_prob_dist: Dense<Softmax>,
}

impl ListNet {
    /// Number of documents directly reranked
    const INPUT_NR_DOCUMENTS: Ix = 10;

    /// Number of features per document
    const INPUT_NR_FEATURES: Ix = 50;

    /// Shape of input: `INPUT_NR_DOCUMENTS` x `INPUT_NR_FEATURES`
    const INPUT_SHAPE: [Ix; 2] = [Self::INPUT_NR_DOCUMENTS, Self::INPUT_NR_FEATURES];

    /// Load ListNet from file at given path.
    #[allow(unused)] //TODO tmp
    pub fn load_from_file(path: impl AsRef<Path>) -> Result<Self, LoadingListNetFailed> {
        let params = BinParams::load_from_file(path)?;
        Self::load(params)
    }

    /// Load ListNet from byte reader.
    #[allow(unused)] //TODO tmp
    pub fn load_from_source(params_source: impl Read) -> Result<Self, LoadingListNetFailed> {
        let params = BinParams::load(params_source)?;
        Self::load(params)
    }

    /// Load ListNet from `BinParams`.
    fn load(mut params: BinParams) -> Result<Self, LoadingListNetFailed> {
        let dense_1 = Dense::load(params.with_scope("dense_1"), Relu)?;
        let dense_1_out_shape = dense_1.check_in_out_shapes(Self::INPUT_SHAPE.into_dimension())?;

        let dense_2 = Dense::load(params.with_scope("dense_2"), Relu)?;
        let dense_2_out_shape = dense_2.check_in_out_shapes(dense_1_out_shape)?;

        let scores = Dense::load(params.with_scope("scores"), Linear)?;
        let scores_out_shape = scores.check_in_out_shapes(dense_2_out_shape)?;
        let flattened_shape = [scores_out_shape.size()].into_dimension();

        let scores_prob_dist =
            Dense::load(params.with_scope("scores_prob_dist"), Softmax::default())?;
        let scores_prob_dist_out_shape = scores_prob_dist.check_in_out_shapes(flattened_shape)?;

        if !params.is_empty() {
            Err(LoadingListNetFailed::LeftoverBinParams {
                params: params.keys().map(Into::into).collect(),
            })
        } else if scores_prob_dist_out_shape.slice() == [10] {
            Ok(Self {
                dense_1,
                dense_2,
                scores,
                scores_prob_dist,
            })
        } else {
            Err(IncompatibleMatrices {
                name_left: "scores_prob_dist/output",
                shape_left: scores_prob_dist_out_shape.into_dyn(),
                name_right: "list_net/output",
                shape_right: (10,).into_dimension().into_dyn(),
                hint: "expected scores_prob_dist output shape to be equal to (10,)",
            }
            .into())
        }
    }

    /// The input is a 2-dimensional array with shape `(number_of_documents, number_of_features_per_document)`.
    ///
    /// # Panics
    ///
    /// Panics if the total number of documents is not 10, or if the number of features is not 50.
    fn run_for_10(&self, inputs: Array2<f32>) -> Vec<f32> {
        assert_eq!(
            inputs.shape(),
            Self::INPUT_SHAPE,
            "only exactly {} documents with exactly {} features are supported, got: {:?}",
            Self::INPUT_NR_DOCUMENTS,
            Self::INPUT_NR_FEATURES,
            inputs.shape()
        );
        let dense1_out = self.dense_1.run(inputs);
        let dense2_out = self.dense_2.run(dense1_out);
        let scores = self.scores.run(dense2_out);
        debug_assert_eq!(scores.shape()[1], 1);
        // flattens the array by removing axis 1
        let scores: Array1<f32> = scores.index_axis_move(Axis(1), 0);
        let outputs = self.scores_prob_dist.run(scores);
        debug_assert!(outputs.is_standard_layout());
        outputs.into_raw_vec()
    }

    /// Runs ListNet for up to [`ListNet::INPUT_NR_DOCUMENTS`] documents by using padding.
    ///
    /// The input can be passed in as a single array or two chunks which need to be
    /// concatenated.
    ///
    /// This will pad the input by repeat the last document,
    /// so that it can call `ListNet.run_for_10` with exactly
    /// 10 documents.
    ///
    /// # Panics
    ///
    /// - If the number of features is not equals to [`ListNet::INPUT_NR_FEATURES`].
    /// - If more then [`ListNet::INPUT_NR_DOCUMENTS`] documents where passed into it.
    fn run_padded<'a>(
        &self,
        first: ArrayView2<'a, f32>,
        second: Option<ArrayView2<'a, f32>>,
    ) -> Vec<f32> {
        let mut nr_documents = first.shape()[0];

        let last = second.unwrap_or(first);
        let last_row = last.slice(s![-1.., ..]);

        let mut stack = vec![first.clone().reborrow()];

        if let Some(second) = second {
            nr_documents += second.shape()[0];
            stack.push(second.clone().reborrow());
        }

        if nr_documents < Self::INPUT_NR_DOCUMENTS {
            let nr_missing = Self::INPUT_NR_DOCUMENTS - nr_documents;
            let padding = last_row
                .broadcast((nr_missing, Self::INPUT_NR_FEATURES))
                .unwrap();
            stack.push(padding);
        }

        let inputs = ndarray::stack(Axis(0), &*stack).unwrap();

        let mut outputs = self.run_for_10(inputs);
        outputs.truncate(nr_documents);
        outputs
    }

    /// Runs ListNet on the input. Requires exactly 10 documents (list items).
    ///
    /// Only exactly 50 features are supported.
    ///
    /// The internal implementation only supports exactly 10 documents.
    ///
    /// If there are less then 10 documents, the last document gets
    /// repeated to pad the input to exactly 10 documents. If there
    /// are more then 10 documents the evaluation will be done in
    /// chunks.
    ///
    /// Any results calculated for paddings are removed before returning the
    /// output.
    #[allow(unused)] //TEMP
    pub(crate) fn run(&self, inputs: Array2<f32>) -> Vec<f32> {
        let nr_documents = inputs.shape()[0];

        if nr_documents == 0 {
            return Vec::new();
        }

        // with given algorithm this can only be this size
        let chunk_size = Self::INPUT_NR_DOCUMENTS / 2;
        debug_assert_eq!(Self::INPUT_NR_DOCUMENTS % 2, 0);

        let mut scores = Vec::with_capacity(nr_documents);
        let mut chunks = inputs.axis_chunks_iter(Axis(0), chunk_size);

        let first_chunk = chunks.next().unwrap();
        let second_chunk = chunks.next();

        scores.append(&mut self.run_padded(first_chunk, second_chunk));

        if nr_documents <= Self::INPUT_NR_DOCUMENTS {
            return scores;
        }

        // We could optimize this by:
        // - first running all doc count independent ListNet steps and only then run this partitioned algorithm.
        // We only reach this code if there are chunks left over.
        for chunk in chunks {
            scores.extend_from_slice(&self.run_padded(first_chunk, Some(chunk))[chunk_size..]);
        }

        scores
    }
}

/// ListNet load failure.
#[derive(Debug, Error)]
pub enum LoadingListNetFailed {
    /// Failed to load bin params.
    #[error(transparent)]
    BinParams(#[from] LoadingBinParamsFailed),

    /// Failed to create instance of `Dense`.
    #[error(transparent)]
    Dense(#[from] LoadingDenseFailed),

    /// Tied to load a ListNet containing incompatible matrices.
    #[error(transparent)]
    IncompatibleMatrices(#[from] IncompatibleMatrices),

    /// BinParams file contains additional parameters,
    /// probably due to loading the wrong model.
    #[error("BinParams contains additional parameters, model data is probably wrong: {params:?}")]
    LeftoverBinParams { params: Vec<String> },
}

#[cfg(test)]
mod tests {

    use std::collections::HashSet;

    use ndarray::{Array, IxDyn};
    use once_cell::sync::Lazy;

    use super::*;

    const LIST_NET_BIN_PARAMS_PATH: &str = "../data/ltr_v0000/ltr.binparams";

    static LIST_NET: Lazy<ListNet> =
        Lazy::new(|| ListNet::load_from_file(LIST_NET_BIN_PARAMS_PATH).unwrap());

    /// A single ListNet Input, cast to shape (10, 50).
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

    const EXPECTED_OUTPUTS: &[f32; 10] = &[
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
    ];

    #[rustfmt::skip]
    const SAMPLE_INPUTS_TO_FEW: &[f32; 150] = &[
        1.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,0.0,0.283,0.283,0.283,0.283,0.0,0.0,0.283,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.283,0.0,0.0,4.0,0.0,0.0,0.0,0.283,0.0,0.0,2.475_141_5,31.0,12.0,8.0,91.0,2.219_780_2,3.673_913,0.0,0.0,0.0,4.0,0.0,
        2.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,0.0,0.283,0.283,0.283,0.283,0.0,0.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,4.0,0.0,0.0,0.0,0.283,0.0,0.0,2.475_141_5,31.0,12.0,8.0,91.0,2.219_780_2,3.673_913,0.0,0.0,0.0,4.0,0.0,
        3.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,0.0,0.283,0.283,0.283,0.283,0.0,0.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,4.0,0.0,0.0,0.0,0.283,0.0,0.0,2.475_141_5,31.0,12.0,8.0,91.0,2.219_780_2,3.673_913,0.0,0.0,0.0,4.0,0.0,
    ];

    #[rustfmt::skip]
    const SAMPLE_INPUTS_TO_FEW_PADDED: &[f32; 500] = &[
        1.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,0.0,0.283,0.283,0.283,0.283,0.0,0.0,0.283,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.283,0.0,0.0,4.0,0.0,0.0,0.0,0.283,0.0,0.0,2.475_141_5,31.0,12.0,8.0,91.0,2.219_780_2,3.673_913,0.0,0.0,0.0,4.0,0.0,
        2.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,0.0,0.283,0.283,0.283,0.283,0.0,0.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,4.0,0.0,0.0,0.0,0.283,0.0,0.0,2.475_141_5,31.0,12.0,8.0,91.0,2.219_780_2,3.673_913,0.0,0.0,0.0,4.0,0.0,
        3.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,0.0,0.283,0.283,0.283,0.283,0.0,0.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,4.0,0.0,0.0,0.0,0.283,0.0,0.0,2.475_141_5,31.0,12.0,8.0,91.0,2.219_780_2,3.673_913,0.0,0.0,0.0,4.0,0.0,
        3.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,0.0,0.283,0.283,0.283,0.283,0.0,0.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,4.0,0.0,0.0,0.0,0.283,0.0,0.0,2.475_141_5,31.0,12.0,8.0,91.0,2.219_780_2,3.673_913,0.0,0.0,0.0,4.0,0.0,
        3.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,0.0,0.283,0.283,0.283,0.283,0.0,0.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,4.0,0.0,0.0,0.0,0.283,0.0,0.0,2.475_141_5,31.0,12.0,8.0,91.0,2.219_780_2,3.673_913,0.0,0.0,0.0,4.0,0.0,
        3.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,0.0,0.283,0.283,0.283,0.283,0.0,0.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,4.0,0.0,0.0,0.0,0.283,0.0,0.0,2.475_141_5,31.0,12.0,8.0,91.0,2.219_780_2,3.673_913,0.0,0.0,0.0,4.0,0.0,
        3.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,0.0,0.283,0.283,0.283,0.283,0.0,0.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,4.0,0.0,0.0,0.0,0.283,0.0,0.0,2.475_141_5,31.0,12.0,8.0,91.0,2.219_780_2,3.673_913,0.0,0.0,0.0,4.0,0.0,
        3.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,0.0,0.283,0.283,0.283,0.283,0.0,0.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,4.0,0.0,0.0,0.0,0.283,0.0,0.0,2.475_141_5,31.0,12.0,8.0,91.0,2.219_780_2,3.673_913,0.0,0.0,0.0,4.0,0.0,
        3.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,0.0,0.283,0.283,0.283,0.283,0.0,0.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,4.0,0.0,0.0,0.0,0.283,0.0,0.0,2.475_141_5,31.0,12.0,8.0,91.0,2.219_780_2,3.673_913,0.0,0.0,0.0,4.0,0.0,
        3.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,0.0,0.283,0.283,0.283,0.283,0.0,0.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,4.0,0.0,0.0,0.0,0.283,0.0,0.0,2.475_141_5,31.0,12.0,8.0,91.0,2.219_780_2,3.673_913,0.0,0.0,0.0,4.0,0.0,
    ];

    #[rustfmt::skip]
    const SAMPLE_INPUTS_REMIX: &[f32; 850] = &[
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
        11.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,0.0,0.283,0.283,0.283,0.283,0.0,0.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,4.0,0.0,0.0,0.0,0.283,0.0,0.0,2.475_141_5,31.0,12.0,8.0,91.0,2.219_780_2,3.673_913,0.0,0.0,0.0,4.0,0.0,
        12.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,0.0,0.283,0.283,0.283,0.283,0.0,0.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,4.0,0.0,0.0,0.0,0.283,0.0,0.0,2.475_141_5,31.0,12.0,8.0,91.0,2.219_780_2,3.673_913,0.0,0.0,0.0,4.0,0.0,
        13.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,0.0,0.283,0.283,0.283,0.283,0.0,0.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,4.0,0.0,0.0,0.0,0.283,0.0,0.0,2.475_141_5,31.0,12.0,8.0,91.0,2.219_780_2,3.673_913,0.0,0.0,0.0,4.0,0.0,
        14.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,0.0,0.283,0.283,0.283,0.283,0.0,0.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,4.0,0.0,0.0,0.0,0.283,0.0,0.0,2.475_141_5,31.0,12.0,8.0,91.0,2.219_780_2,3.673_913,0.0,0.0,0.0,4.0,0.0,
        15.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,0.0,0.283,0.283,0.283,0.283,0.0,0.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,4.0,0.0,0.0,0.0,0.283,0.0,0.0,2.475_141_5,31.0,12.0,8.0,91.0,2.219_780_2,3.673_913,0.0,0.0,0.0,4.0,0.0,
        16.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,0.0,0.283,0.283,0.283,0.283,0.0,0.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,4.0,0.0,0.0,0.0,0.283,0.0,0.0,2.475_141_5,31.0,12.0,8.0,91.0,2.219_780_2,3.673_913,0.0,0.0,0.0,4.0,0.0,
        17.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,0.0,0.283,0.283,0.283,0.283,0.0,0.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,4.0,0.0,0.0,0.0,0.283,0.0,0.0,2.475_141_5,31.0,12.0,8.0,91.0,2.219_780_2,3.673_913,0.0,0.0,0.0,4.0,0.0,
    ];

    #[rustfmt::skip]
    const SAMPLE_INPUTS_REMIX_SUB1: &[f32; 500] = &[
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

    #[rustfmt::skip]
    const SAMPLE_INPUTS_REMIX_SUB2: &[f32; 500] = &[
        1.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,0.0,0.283,0.283,0.283,0.283,0.0,0.0,0.283,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.283,0.0,0.0,4.0,0.0,0.0,0.0,0.283,0.0,0.0,2.475_141_5,31.0,12.0,8.0,91.0,2.219_780_2,3.673_913,0.0,0.0,0.0,4.0,0.0,
        2.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,0.0,0.283,0.283,0.283,0.283,0.0,0.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,4.0,0.0,0.0,0.0,0.283,0.0,0.0,2.475_141_5,31.0,12.0,8.0,91.0,2.219_780_2,3.673_913,0.0,0.0,0.0,4.0,0.0,
        3.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,0.0,0.283,0.283,0.283,0.283,0.0,0.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,4.0,0.0,0.0,0.0,0.283,0.0,0.0,2.475_141_5,31.0,12.0,8.0,91.0,2.219_780_2,3.673_913,0.0,0.0,0.0,4.0,0.0,
        4.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,0.0,0.283,0.283,0.283,0.283,0.0,0.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,4.0,0.0,0.0,0.0,0.283,0.0,0.0,2.475_141_5,31.0,12.0,8.0,91.0,2.219_780_2,3.673_913,0.0,0.0,0.0,4.0,0.0,
        5.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,0.0,0.283,0.283,0.283,0.283,0.0,0.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,4.0,0.0,0.0,0.0,0.283,0.0,0.0,2.475_141_5,31.0,12.0,8.0,91.0,2.219_780_2,3.673_913,0.0,0.0,0.0,4.0,0.0,
        11.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,0.0,0.283,0.283,0.283,0.283,0.0,0.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,4.0,0.0,0.0,0.0,0.283,0.0,0.0,2.475_141_5,31.0,12.0,8.0,91.0,2.219_780_2,3.673_913,0.0,0.0,0.0,4.0,0.0,
        12.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,0.0,0.283,0.283,0.283,0.283,0.0,0.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,4.0,0.0,0.0,0.0,0.283,0.0,0.0,2.475_141_5,31.0,12.0,8.0,91.0,2.219_780_2,3.673_913,0.0,0.0,0.0,4.0,0.0,
        13.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,0.0,0.283,0.283,0.283,0.283,0.0,0.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,4.0,0.0,0.0,0.0,0.283,0.0,0.0,2.475_141_5,31.0,12.0,8.0,91.0,2.219_780_2,3.673_913,0.0,0.0,0.0,4.0,0.0,
        14.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,0.0,0.283,0.283,0.283,0.283,0.0,0.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,4.0,0.0,0.0,0.0,0.283,0.0,0.0,2.475_141_5,31.0,12.0,8.0,91.0,2.219_780_2,3.673_913,0.0,0.0,0.0,4.0,0.0,
        15.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,0.0,0.283,0.283,0.283,0.283,0.0,0.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,4.0,0.0,0.0,0.0,0.283,0.0,0.0,2.475_141_5,31.0,12.0,8.0,91.0,2.219_780_2,3.673_913,0.0,0.0,0.0,4.0,0.0,
    ];

    #[rustfmt::skip]
    const SAMPLE_INPUTS_REMIX_SUB3: &[f32; 500] = &[
        1.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,0.0,0.283,0.283,0.283,0.283,0.0,0.0,0.283,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.283,0.0,0.0,4.0,0.0,0.0,0.0,0.283,0.0,0.0,2.475_141_5,31.0,12.0,8.0,91.0,2.219_780_2,3.673_913,0.0,0.0,0.0,4.0,0.0,
        2.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,0.0,0.283,0.283,0.283,0.283,0.0,0.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,4.0,0.0,0.0,0.0,0.283,0.0,0.0,2.475_141_5,31.0,12.0,8.0,91.0,2.219_780_2,3.673_913,0.0,0.0,0.0,4.0,0.0,
        3.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,0.0,0.283,0.283,0.283,0.283,0.0,0.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,4.0,0.0,0.0,0.0,0.283,0.0,0.0,2.475_141_5,31.0,12.0,8.0,91.0,2.219_780_2,3.673_913,0.0,0.0,0.0,4.0,0.0,
        4.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,0.0,0.283,0.283,0.283,0.283,0.0,0.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,4.0,0.0,0.0,0.0,0.283,0.0,0.0,2.475_141_5,31.0,12.0,8.0,91.0,2.219_780_2,3.673_913,0.0,0.0,0.0,4.0,0.0,
        5.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,0.0,0.283,0.283,0.283,0.283,0.0,0.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,4.0,0.0,0.0,0.0,0.283,0.0,0.0,2.475_141_5,31.0,12.0,8.0,91.0,2.219_780_2,3.673_913,0.0,0.0,0.0,4.0,0.0,
        16.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,0.0,0.283,0.283,0.283,0.283,0.0,0.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,4.0,0.0,0.0,0.0,0.283,0.0,0.0,2.475_141_5,31.0,12.0,8.0,91.0,2.219_780_2,3.673_913,0.0,0.0,0.0,4.0,0.0,
        17.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,0.0,0.283,0.283,0.283,0.283,0.0,0.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,4.0,0.0,0.0,0.0,0.283,0.0,0.0,2.475_141_5,31.0,12.0,8.0,91.0,2.219_780_2,3.673_913,0.0,0.0,0.0,4.0,0.0,
        17.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,0.0,0.283,0.283,0.283,0.283,0.0,0.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,4.0,0.0,0.0,0.0,0.283,0.0,0.0,2.475_141_5,31.0,12.0,8.0,91.0,2.219_780_2,3.673_913,0.0,0.0,0.0,4.0,0.0,
        17.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,0.0,0.283,0.283,0.283,0.283,0.0,0.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,4.0,0.0,0.0,0.0,0.283,0.0,0.0,2.475_141_5,31.0,12.0,8.0,91.0,2.219_780_2,3.673_913,0.0,0.0,0.0,4.0,0.0,
        17.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,0.0,0.283,0.283,0.283,0.283,0.0,0.0,0.283,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.283,0.0,0.0,4.0,0.0,0.0,0.0,0.283,0.0,0.0,2.475_141_5,31.0,12.0,8.0,91.0,2.219_780_2,3.673_913,0.0,0.0,0.0,4.0,0.0,
    ];

    #[test]
    fn test_list_net_end_to_end_with_run_for_10() {
        let list_net = &*LIST_NET;

        let inputs = Array1::from(SAMPLE_INPUTS.to_vec())
            .into_shape((10, 50))
            .unwrap();

        let outcome = list_net.run_for_10(inputs);

        assert_approx_eq!(f32, outcome, EXPECTED_OUTPUTS, ulps = 4);
    }

    #[test]
    fn test_directly_supported_input_size_is_even() {
        assert_eq!(ListNet::INPUT_NR_DOCUMENTS % 2, 0);
    }

    #[test]
    fn test_run_works_with_10_inputs() {
        let list_net = &*LIST_NET;

        let inputs = Array1::from(SAMPLE_INPUTS.to_vec())
            .into_shape((10, 50))
            .unwrap();

        let outcome = list_net.run(inputs);

        assert_approx_eq!(f32, outcome, EXPECTED_OUTPUTS, ulps = 4);
    }

    #[test]
    fn test_running_list_net_with_no_inputs_works() {
        let list_net = &*LIST_NET;
        let inputs = Array2::<f32>::zeros((0, 50));
        let outputs = list_net.run(inputs);
        assert!(outputs.is_empty());
    }

    #[test]
    fn test_list_net_for_more_than_10_works() {
        //TODO load once
        let list_net = &*LIST_NET;

        let big_inputs = Array1::from(SAMPLE_INPUTS_REMIX.to_vec())
            .into_shape((17, 50))
            .unwrap();

        let big_out = list_net.run(big_inputs);

        let sub1_inputs = Array1::from(SAMPLE_INPUTS_REMIX_SUB1.to_vec())
            .into_shape((10, 50))
            .unwrap();

        let sub1_out = list_net.run(sub1_inputs);

        let sub2_inputs = Array1::from(SAMPLE_INPUTS_REMIX_SUB2.to_vec())
            .into_shape((10, 50))
            .unwrap();

        let sub2_out = list_net.run(sub2_inputs);

        let sub3_inputs = Array1::from(SAMPLE_INPUTS_REMIX_SUB3.to_vec())
            .into_shape((10, 50))
            .unwrap();

        let sub3_out = list_net.run(sub3_inputs);

        let mut subs_outs = sub1_out;
        subs_outs.extend_from_slice(&sub2_out[5..]);
        subs_outs.extend_from_slice(&sub3_out[5..]);
        subs_outs.truncate(17);

        assert_approx_eq!(f32, big_out, subs_outs, ulps = 4);
    }

    #[test]
    fn test_too_few_inputs() {
        let list_net = &*LIST_NET;

        let to_few_inputs = Array1::from(SAMPLE_INPUTS_TO_FEW.to_vec())
            .into_shape((3, 50))
            .unwrap();

        let out = list_net.run(to_few_inputs);

        let to_few_inputs_padded = Array1::from(SAMPLE_INPUTS_TO_FEW_PADDED.to_vec())
            .into_shape((10, 50))
            .unwrap();

        let out_padded = list_net.run(to_few_inputs_padded);

        assert_approx_eq!(f32, &out, &out_padded[..out.len()], ulps = 4);
        assert_eq!((out.len(), out_padded.len()), (3, 10));
    }

    const EMPTY_BIN_PARAMS: &[u8] = &[0u8; 8];

    const BIN_PARAMS_WITH_EMPTY_ARRAY_AND_KEY: &[u8] = &[
        1u8, 0, 0, 0, 0, 0, 0, 0, // 1 entry
        0, 0, 0, 0, 0, 0, 0, 0, // empty string key
        1, 0, 0, 0, 0, 0, 0, 0, // 1 dimensional array
        0, 0, 0, 0, 0, 0, 0, 0, // the dimension is 0
        0, 0, 0, 0, 0, 0, 0, 0, // and we have 0 bytes of data data
    ];

    const BIN_PARAMS_WITH_SOME_KEYS: &[u8] = &[
        2u8, 0, 0, 0, 0, 0, 0, 0, // 2 entries
        3, 0, 0, 0, 0, 0, 0, 0, b'f', b'o', b'o', // key 1
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // shape [0]
        0, 0, 0, 0, 0, 0, 0, 0, // and data
        3, 0, 0, 0, 0, 0, 0, 0, b'b', b'a', b'r', // key 2
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // shape [0]
        0, 0, 0, 0, 0, 0, 0, 0, // and data
    ];

    #[test]
    fn test_is_empty() {
        let params = BinParams::load(EMPTY_BIN_PARAMS).unwrap();
        assert!(params.is_empty());

        let mut params = BinParams::load(BIN_PARAMS_WITH_EMPTY_ARRAY_AND_KEY).unwrap();
        assert!(!params.is_empty());

        let array: Array<f32, IxDyn> = params.take("").unwrap();
        assert_eq!(array.shape(), &[0]);
    }

    #[test]
    fn test_keys() {
        let params = BinParams::load(BIN_PARAMS_WITH_SOME_KEYS).unwrap();
        let mut expected = HashSet::default();
        expected.insert("foo");
        expected.insert("bar");
        assert_eq!(params.keys().collect::<HashSet<_>>(), expected);
    }
}
