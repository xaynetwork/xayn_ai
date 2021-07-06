//! ListNet implementation using the NdArray crate.

use std::{
    io::Read,
    ops::{Add, Div, MulAssign},
    path::Path,
};

use derive_more::Deref;
use thiserror::Error;

use ndarray::{s, Array1, Array2, ArrayView2, Axis, Dimension, IntoDimension, Ix};

use crate::Relevance;

use self::{
    ndlayers::{
        activation::{Linear, Relu, Softmax},
        Dense,
        DenseGradientSet,
        IncompatibleMatrices,
        LoadingDenseFailed,
    },
    ndutils::{
        io::{BinParams, LoadingBinParamsFailed},
        kl_divergence,
        softmax,
    },
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
#[cfg_attr(test, derive(Clone))]
pub struct ListNet {
    dense_1: Dense<Relu>,
    dense_2: Dense<Relu>,
    scores: Dense<Linear>,
    scores_prob_dist: Dense<Softmax>,
}

impl ListNet {
    /// Number of documents directly reranked
    const INPUT_NR_DOCUMENTS: Ix = 10;

    /// The size by with input is chunked.
    ///
    /// The first chunk is then used together with each other chunk.
    const CHUNK_SIZE: Ix = Self::INPUT_NR_DOCUMENTS / 2;

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

    /// Calculates the intermediate scores.
    ///
    /// The input is a 2-dimensional array with the shape `(number_of_documents, Self::INPUT_NR_FEATURES)`.
    ///
    /// Any number of documents is supported.
    fn calculate_intermediate_scores(
        &self,
        inputs: ArrayView2<f32>,
        for_back_propagation: bool,
    ) -> (Array1<f32>, Option<PartialForwardPassData>) {
        debug_assert_eq!(inputs.shape()[1], Self::INPUT_NR_FEATURES);
        let (dense1_y, dense1_z) = self.dense_1.run(&inputs, for_back_propagation);
        let (dense2_y, dense2_z) = self.dense_2.run(&dense1_y, for_back_propagation);
        let (scores, _) = self.scores.run(&dense2_y, false);
        debug_assert_eq!(scores.shape()[1], 1);

        //flatten
        let scores = scores.index_axis_move(Axis(1), 0);

        let forward_pass = for_back_propagation.then(|| PartialForwardPassData {
            dense1_y,
            dense1_z: dense1_z.unwrap(),
            dense2_y,
            dense2_z: dense2_z.unwrap(),
        });

        (scores, forward_pass)
    }

    /// Calculates the final ListNet scores based on the intermediate scores.
    ///
    /// The input must be based on the output of [`ListNet.calculate_intermediate_scores()`],
    /// but only supports a size of exactly [`Self::INPUT_NR_DOCUMENT`].
    fn calculate_final_scores(&self, scores: &Array1<f32>) -> Array1<f32> {
        debug_assert_eq!(scores.shape()[0], Self::INPUT_NR_DOCUMENTS);
        let (prob_dist_y, _) = self.scores_prob_dist.run(scores, false);
        prob_dist_y
    }

    /// Calculates the final scores for up to [`ListNet::INPUT_NR_DOCUMENTS`] intermediate scores.
    ///
    /// The intermediate scores are provided in one or two chunks (`first`,`second`) which
    /// will be concatenated and then, the last element get repeated to fill up the input to
    /// have exactly [`ListNet::INPUT_NR_DOCUMENTS`] elements.
    ///
    /// # Panics
    ///
    /// Will panic if called with `0` or more than [`Self::INPUT_NR_DOCUMENTS`] elements.
    fn calculate_final_scores_padded(&self, first: &[f32], second: Option<&[f32]>) -> Vec<f32> {
        let mut inputs = Vec::with_capacity(Self::INPUT_NR_DOCUMENTS);
        inputs.extend_from_slice(first);
        if let Some(second) = second {
            inputs.extend_from_slice(second);
        }

        let nr_documents = inputs.len();
        assert!(nr_documents > 0 && nr_documents <= Self::INPUT_NR_DOCUMENTS);

        inputs.resize(
            Self::INPUT_NR_DOCUMENTS,
            inputs.last().copied().unwrap_or_default(),
        );

        let mut outputs = self
            .calculate_final_scores(&Array1::from(inputs))
            .into_raw_vec();
        outputs.truncate(nr_documents);

        outputs
    }

    /// Runs ListNet on the input.
    ///
    /// Only exactly [`Self::INPUT_NR_FEATURES`] features are supported.
    ///
    /// # Panics
    ///
    /// If inputs have not exactly [`Self::INPUT_NR_FEATURES`] features.
    pub(crate) fn run(&self, inputs: Array2<f32>) -> Vec<f32> {
        let nr_documents = inputs.shape()[0];

        assert_eq!(
            inputs.shape()[1],
            Self::INPUT_NR_FEATURES,
            "ListNet expects exactly {} features per document got {}",
            Self::INPUT_NR_FEATURES,
            inputs.shape()[1]
        );

        if nr_documents == 0 {
            return Vec::new();
        }

        let intermediate_scores = {
            let (scores, _) = self.calculate_intermediate_scores(inputs.view(), false);
            debug_assert!(scores.is_standard_layout());
            scores.into_raw_vec()
        };

        let mut scores = Vec::with_capacity(nr_documents);
        let mut chunks = intermediate_scores.chunks(Self::CHUNK_SIZE);

        let first_chunk = chunks.next().unwrap();
        let second_chunk = chunks.next();

        scores.append(&mut self.calculate_final_scores_padded(first_chunk, second_chunk));

        if nr_documents <= Self::INPUT_NR_DOCUMENTS {
            return scores;
        }

        for chunk in chunks {
            scores.extend_from_slice(
                &self.calculate_final_scores_padded(first_chunk, Some(chunk))[Self::CHUNK_SIZE..],
            );
        }

        scores
    }

    /// Given a vec of batches, train set, learning rate and number of epochs trains the ListNet.
    ///
    /// Returns the mean cost of the `test_data` for each epoch.
    //FIXME we probably will need some more complete training loop,
    //      including functionality to external triggered early stop or pause of training
    //      and automatic stop of training if the improvement in the evaluation is too small
    //      ever multiple epochs or similar. But that depends a lot on the integration with
    //      XaynNet and Team Blue so it doesn't make sense at the current point.
    //FIXME remove annotation
    #[allow(dead_code)]
    pub(crate) fn train_with_sdg(
        &mut self,
        training_data: Vec<Vec<(Array2<f32>, Vec<Relevance>)>>,
        test_data: Vec<(Array2<f32>, Vec<Relevance>)>,
        learning_rate: f32,
        epochs: usize,
    ) -> Vec<f32> {
        (0..epochs)
            .map(|_| {
                for batch in &training_data {
                    self.train_batch_with_sdg(batch, learning_rate);
                }
                let (acc, count) =
                    test_data
                        .iter()
                        .fold((0f32, 1), |(acc, count), (inputs, relevances)| {
                            if let Some(cost) = self.evaluate_with_kl_divergence(inputs, relevances)
                            {
                                (acc + cost, count + 1)
                            } else {
                                (acc, count)
                            }
                        });
                if count > 0 {
                    acc / count as f32
                } else {
                    0.
                }
            })
            .collect()
    }

    /// Trains this ListNet on a single batch a single time using SGD with given learning rate.
    fn train_batch_with_sdg<'a>(
        &mut self,
        batch: impl IntoIterator<Item = &'a (Array2<f32>, Vec<Relevance>)>,
        learning_rate: f32,
    ) {
        let gradient_sets = batch.into_iter().flat_map(|(inputs, relevances)| {
            Self::prepare_inputs_for_training(&inputs, &relevances)
                .map(|(inputs, relevances)| self.gradients_for_query(inputs, relevances))
        });

        if let Some(mut gradients) = GradientSet::merge_batch(gradient_sets) {
            gradients *= -learning_rate;
            self.add_gradients(gradients);
        }
    }

    /// Computes the KL Divergence for given inputs and relevances.
    ///
    /// It should be noted that the Kullback-Leibler Divergence returned
    /// is based on bits, while the cost used for back-propagation is
    /// based on nats (it used `ln` instead of `log2`).
    fn evaluate_with_kl_divergence(
        &self,
        inputs: &Array2<f32>,
        relevance: &[Relevance],
    ) -> Option<f32> {
        Self::prepare_inputs_for_training(inputs, relevance).map(|(inputs, targets)| {
            let (scores_y, _) = self.calculate_intermediate_scores(inputs, false);
            let prob_dist_y = self.calculate_final_scores(&scores_y);
            kl_divergence(targets, prob_dist_y)
        })
    }

    /// Computes the gradients for given inputs and target prob. dist.
    ///
    /// # Panics
    ///
    /// If inputs and relevances are not exactly [`Self::INPUT_NR_DOCUMENTS`].
    fn gradients_for_query(
        &self,
        inputs: ArrayView2<f32>,
        target_prob_dist: Array1<f32>,
    ) -> GradientSet {
        assert_eq!(inputs.shape()[0], Self::INPUT_NR_DOCUMENTS);
        assert_eq!(target_prob_dist.len(), Self::INPUT_NR_DOCUMENTS);
        let results = self.train_forward_pass(inputs);
        self.train_back_propagation(inputs, target_prob_dist, results)
    }

    /// Run the the forward pass of back-propagation.
    ///
    /// # Panics
    ///
    /// Expects exactly 10 documents, panics else wise.
    fn train_forward_pass(&self, inputs: ArrayView2<f32>) -> ForwardPassData {
        let (scores_y, partial_forward_pass) = self.calculate_intermediate_scores(inputs, true);
        let prob_dist_y = self.calculate_final_scores(&scores_y);

        ForwardPassData {
            partial_forward_pass: partial_forward_pass.unwrap(),
            scores_y,
            prob_dist_y,
        }
    }

    /// Run back propagation base on given inputs, target prob. dist. and forward pass data.
    fn train_back_propagation(
        &self,
        inputs: ArrayView2<f32>,
        target_prob_dist: Array1<f32>,
        forward_pass: ForwardPassData,
    ) -> GradientSet {
        let nr_documents = inputs.shape()[0];
        let p_cost_and_prob_dist = &forward_pass.prob_dist_y - target_prob_dist;
        let d_prob_dist = self
            .scores_prob_dist
            .gradients_from_partials_1d(forward_pass.scores_y.view(), p_cost_and_prob_dist.view());

        //FIXME transpose weights?
        let p_scores = p_cost_and_prob_dist.dot(&self.scores_prob_dist.weights().t()); // * 1 as af' = 1 as activation function is linear

        let mut d_scores = DenseGradientSet::no_change_for(&self.scores);
        let mut d_dense2 = DenseGradientSet::no_change_for(&self.dense_2);
        let mut d_dense1 = DenseGradientSet::no_change_for(&self.dense_1);

        for row in 0..nr_documents {
            let p_scores = p_scores.slice(s![row..row + 1]);

            let d_scores_part = self
                .scores
                .gradients_from_partials_1d(forward_pass.dense2_y.slice(s![row, ..]), p_scores);
            d_scores += d_scores_part;

            let p_dense2 = p_scores.dot(&self.scores.weights().t())
                * Relu::partial_derivatives_at(&forward_pass.dense2_z.slice(s![row, ..]));
            let d_dense2_part = self.dense_2.gradients_from_partials_1d(
                forward_pass.dense1_y.slice(s![row, ..]),
                p_dense2.view(),
            );
            d_dense2 += d_dense2_part;

            let p_dense1 = p_dense2.dot(&self.dense_2.weights().t())
                * Relu::partial_derivatives_at(&forward_pass.dense1_z.slice(s![row, ..]));
            let d_dense1_part = self
                .dense_1
                .gradients_from_partials_1d(inputs.slice(s![row, ..]), p_dense1.view());
            d_dense1 += d_dense1_part;
        }

        GradientSet {
            dense1: d_dense1,
            dense2: d_dense2,
            scores: d_scores,
            prob_dist: d_prob_dist,
        }
    }

    /// Adds gradients to this `ListNet`.
    fn add_gradients(&mut self, gradients: GradientSet) {
        let GradientSet {
            dense1,
            dense2,
            scores,
            prob_dist,
        } = gradients;

        self.scores_prob_dist.add_gradients(&prob_dist);
        self.scores.add_gradients(&scores);
        self.dense_2.add_gradients(&dense2);
        self.dense_1.add_gradients(&dense1);
    }

    /// Prepare the inputs for training.
    ///
    /// This will:
    ///
    /// - return `None` if there are less then 10 documents
    /// - truncates inputs and relevances to 10 documents
    /// - return `None` if there are no relevant documents (after truncating)
    /// - calculates the target distribution based on the relevances
    fn prepare_inputs_for_training<'a>(
        inputs: &'a Array2<f32>,
        relevances: &'a [Relevance],
    ) -> Option<(ArrayView2<'a, f32>, Array1<f32>)> {
        if inputs.shape()[0] < Self::INPUT_NR_DOCUMENTS {
            None
        } else {
            Some((
                inputs.slice(s![..Self::INPUT_NR_DOCUMENTS, ..]),
                Self::prepare_target_prob_dist(&relevances[..Self::INPUT_NR_DOCUMENTS])?,
            ))
        }
    }

    /// Turns the given relevances into a probability distribution.
    ///
    /// If there was no relevant document in the inputs `None` is returned.
    fn prepare_target_prob_dist(relevance: &[Relevance]) -> Option<Array1<f32>> {
        if !relevance.iter().any(|r| match *r {
            Relevance::Low => false,
            Relevance::High | Relevance::Medium => true,
        }) {
            return None;
        }

        let len = relevance.len() as f32;
        let target: Array1<f32> = relevance
            .iter()
            .enumerate()
            .map(|(idx, relevance)| {
                let idx = idx as f32;
                let relevance = *relevance as u8 as f32;
                // mirrors dart impl. but works for more or less then 10 elements
                // still this seems to be a somewhat arbitrary choice
                (len - idx - 1.0) / len + relevance
            })
            .collect();

        Some(softmax(target, Axis(0)))
    }
}

#[derive(Deref)]
struct ForwardPassData {
    #[deref]
    partial_forward_pass: PartialForwardPassData,
    scores_y: Array1<f32>,
    prob_dist_y: Array1<f32>,
}

struct PartialForwardPassData {
    dense1_y: Array2<f32>,
    dense1_z: Array2<f32>,
    dense2_y: Array2<f32>,
    dense2_z: Array2<f32>,
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

#[cfg_attr(test, derive(Debug, Clone))]
struct GradientSet {
    dense1: DenseGradientSet,
    dense2: DenseGradientSet,
    scores: DenseGradientSet,
    prob_dist: DenseGradientSet,
}

impl GradientSet {
    fn merge_batch(gradient_sets: impl IntoIterator<Item = GradientSet>) -> Option<GradientSet> {
        let mut count = 1;
        gradient_sets
            .into_iter()
            .reduce(|left, right| {
                count += 1;
                left + right
            })
            .map(|gradient_set| gradient_set / count as f32)
    }
}

impl Add for GradientSet {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        self.dense1 += rhs.dense1;
        self.dense2 += rhs.dense2;
        self.scores += rhs.scores;
        self.prob_dist += rhs.prob_dist;
        self
    }
}

impl Div<f32> for GradientSet {
    type Output = Self;

    fn div(mut self, rhs: f32) -> Self::Output {
        self.dense1 /= rhs;
        self.dense2 /= rhs;
        self.scores /= rhs;
        self.prob_dist /= rhs;
        self
    }
}

impl MulAssign<f32> for GradientSet {
    fn mul_assign(&mut self, rhs: f32) {
        self.dense1 *= rhs;
        self.dense2 *= rhs;
        self.scores *= rhs;
        self.prob_dist *= rhs;
    }
}

#[cfg(test)]
mod tests {

    use std::collections::HashSet;

    use ndarray::{arr1, arr2, Array, IxDyn};
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
    fn test_list_net_end_to_end_without_chunking_and_padding() {
        let list_net = &*LIST_NET;

        let inputs = Array1::from(SAMPLE_INPUTS.to_vec())
            .into_shape((10, 50))
            .unwrap();

        let (scores, _) = list_net.calculate_intermediate_scores(inputs.view(), false);
        let scores = scores.into_raw_vec();
        let outcome = list_net.calculate_final_scores_padded(&scores, None);

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

    #[test]
    fn test_chunk_size_is_valid() {
        assert_eq!(ListNet::CHUNK_SIZE * 2, ListNet::INPUT_NR_DOCUMENTS);
    }

    //FIXME create better tests
    #[test]
    fn test_training_list_net_does_not_panic() {
        use Relevance::{High, Low, Medium};
        let mut list_net = LIST_NET.clone();

        let inputs = Array1::from(SAMPLE_INPUTS.to_vec())
            .into_shape((10, 50))
            .unwrap();

        let relevances = vec![Low, Low, Medium, Medium, Low, Medium, High, Low, High, High];
        let data_frame = (inputs, relevances);

        let training_data = vec![vec![
            data_frame.clone(),
            data_frame.clone(),
            data_frame.clone(),
        ]];
        let test_data = vec![data_frame.clone()];

        let nr_epochs = 5;
        let res = list_net.train_with_sdg(training_data, test_data, 0.1, nr_epochs);
        assert_eq!(res.len(), nr_epochs);
        for idx in 0..(res.len() - 1) {
            // I think, this could fail. But with the training data
            // we use in this test it should not.
            assert!(!res[idx].is_nan());
            assert!(res[idx] >= res[idx + 1]);
        }
    }

    #[test]
    fn test_gradients_merge_batch() {
        let res = GradientSet::merge_batch(std::iter::empty());
        assert!(res.is_none());

        let a = GradientSet {
            dense1: DenseGradientSet {
                weight_gradients: arr2(&[[0.1, -2.], [0.3, 0.04]]),
                bias_gradients: arr1(&[0.4, 1.23]),
            },
            dense2: DenseGradientSet {
                weight_gradients: arr2(&[[2., -2.], [-0.3, 0.4]]),
                bias_gradients: arr1(&[0.1, 3.43]),
            },
            scores: DenseGradientSet {
                weight_gradients: arr2(&[[0.125, 2.4], [0.321, 0.454]]),
                bias_gradients: arr1(&[0.42, 2.23]),
            },
            prob_dist: DenseGradientSet {
                weight_gradients: arr2(&[[100., -0.2], [3.25, 0.22]]),
                bias_gradients: arr1(&[-0.42, -2.25]),
            },
        };

        let a2 = GradientSet::merge_batch(Some(a.clone())).unwrap();

        assert_approx_eq!(f32, &a2.dense1.weight_gradients, &a.dense1.weight_gradients);
        assert_approx_eq!(f32, &a2.dense1.bias_gradients, &a.dense1.bias_gradients);
        assert_approx_eq!(f32, &a2.dense2.weight_gradients, &a.dense2.weight_gradients);
        assert_approx_eq!(f32, &a2.dense2.bias_gradients, &a.dense2.bias_gradients);
        assert_approx_eq!(f32, &a2.scores.weight_gradients, &a.scores.weight_gradients);
        assert_approx_eq!(f32, &a2.scores.bias_gradients, &a.scores.bias_gradients);
        assert_approx_eq!(
            f32,
            &a2.prob_dist.weight_gradients,
            &a.prob_dist.weight_gradients
        );
        assert_approx_eq!(
            f32,
            &a2.prob_dist.bias_gradients,
            &a.prob_dist.bias_gradients
        );

        let b = GradientSet {
            dense1: DenseGradientSet {
                weight_gradients: arr2(&[[0.1, 2.], [0.3, 0.04]]),
                bias_gradients: arr1(&[0.4, 1.23]),
            },
            dense2: DenseGradientSet {
                weight_gradients: arr2(&[[0.2, -2.8], [0.3, 0.04]]),
                bias_gradients: arr1(&[0.4, 1.23]),
            },
            scores: DenseGradientSet {
                weight_gradients: arr2(&[[0.1, -2.], [0.3, 0.04]]),
                bias_gradients: arr1(&[0.4, 1.23]),
            },
            prob_dist: DenseGradientSet {
                weight_gradients: arr2(&[[0.0, -2.], [0.3, 0.04]]),
                bias_gradients: arr1(&[0.38, 1.21]),
            },
        };

        let g = GradientSet::merge_batch(vec![a, b]).unwrap();

        assert_approx_eq!(f32, &g.dense1.weight_gradients, [[0.1, 0.], [0.3, 0.04]]);
        assert_approx_eq!(f32, &g.dense1.bias_gradients, [0.4, 1.23]);
        assert_approx_eq!(f32, &g.dense2.weight_gradients, [[1.1, -2.4], [0.0, 0.22]]);
        assert_approx_eq!(f32, &g.dense2.bias_gradients, [0.25, 2.33]);
        assert_approx_eq!(
            f32,
            &g.scores.weight_gradients,
            [[0.1125, 0.2], [0.3105, 0.247]],
            ulps = 4
        );
        assert_approx_eq!(f32, &g.scores.bias_gradients, [0.41, 1.73]);
        assert_approx_eq!(
            f32,
            &g.prob_dist.weight_gradients,
            [[50., -1.1], [1.775, 0.13]]
        );
        assert_approx_eq!(f32, &g.prob_dist.bias_gradients, [-0.02, -0.52], ulps = 4);
    }

    #[test]
    fn test_prepare_target_prop_dist() {
        use Relevance::{High, Low, Medium};

        let relevances = vec![Low; 10];
        let res = ListNet::prepare_target_prob_dist(&relevances);
        assert!(res.is_none());

        let relevances = vec![Low, Low, Medium, Medium, Low, Medium, High, Low, High, High];
        let dist = ListNet::prepare_target_prob_dist(&relevances).unwrap();
        assert_approx_eq!(
            f32,
            dist,
            [
                0.051708338569288054,
                0.04678763956196383,
                0.11507902383029622,
                0.10412780679270388,
                0.03466113589019159,
                0.08525263767174904,
                0.20968765285177007,
                0.025677601016978965,
                0.17167772993048422,
                0.15534043388457408
            ]
        );
    }

    #[test]
    fn test_prepare_inputs_for_training() {
        use Relevance::{High, Low, Medium};

        let relevances = vec![Low, Low, Medium, Medium, Low, Medium, High, Low, High, High];
        let inputs = Array1::from(SAMPLE_INPUTS.to_vec())
            .into_shape((10, 50))
            .unwrap();

        let (processed_inputs, dist) =
            ListNet::prepare_inputs_for_training(&inputs, &relevances).unwrap();

        assert_approx_eq!(f32, &inputs, processed_inputs);

        assert_approx_eq!(
            f32,
            dist,
            [
                0.051708338569288054,
                0.04678763956196383,
                0.11507902383029622,
                0.10412780679270388,
                0.03466113589019159,
                0.08525263767174904,
                0.20968765285177007,
                0.025677601016978965,
                0.17167772993048422,
                0.15534043388457408
            ]
        );

        assert!(ListNet::prepare_inputs_for_training(&inputs, &vec![Low; 10]).is_none());

        let few_inputs = Array1::from(SAMPLE_INPUTS_TO_FEW.to_vec())
            .into_shape((3, 50))
            .unwrap();

        assert!(ListNet::prepare_inputs_for_training(&few_inputs, &relevances).is_none());
    }
}
