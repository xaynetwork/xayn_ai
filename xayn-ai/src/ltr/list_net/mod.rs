//! ListNet implementation using the NdArray crate.

use std::{
    io::Read,
    iter,
    ops::{Add, Div, MulAssign},
};

use thiserror::Error;

use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis, Dimension, IntoDimension, Ix};
use ndutils::he_normal_weights_init;

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
    optimizer::Optimizer,
};

mod ndlayers;
mod ndutils;

pub mod optimizer;

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
    dense1: Dense<Relu>,
    dense2: Dense<Relu>,
    scores: Dense<Linear>,
    prob_dist: Dense<Softmax>,
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
    #[cfg(test)]
    pub fn load_from_file(path: impl AsRef<std::path::Path>) -> Result<Self, LoadingListNetFailed> {
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
        let dense1 = Dense::load(params.with_scope("dense_1"), Relu)?;
        let dense1_out_shape = dense1.check_in_out_shapes(Self::INPUT_SHAPE.into_dimension())?;

        let dense2 = Dense::load(params.with_scope("dense_2"), Relu)?;
        let dense2_out_shape = dense2.check_in_out_shapes(dense1_out_shape)?;

        let scores = Dense::load(params.with_scope("scores"), Linear)?;
        let scores_out_shape = scores.check_in_out_shapes(dense2_out_shape)?;
        let flattened_shape = [scores_out_shape.size()].into_dimension();

        let prob_dist = Dense::load(params.with_scope("scores_prob_dist"), Softmax::default())?;
        let prob_dist_out_shape = prob_dist.check_in_out_shapes(flattened_shape)?;

        if !params.is_empty() {
            Err(LoadingListNetFailed::LeftoverBinParams {
                params: params.keys().map(Into::into).collect(),
            })
        } else if prob_dist_out_shape.slice() == [10] {
            Ok(Self {
                dense1,
                dense2,
                scores,
                prob_dist,
            })
        } else {
            Err(IncompatibleMatrices {
                name_left: "scores_prob_dist/output",
                shape_left: prob_dist_out_shape.into_dyn(),
                name_right: "list_net/output",
                shape_right: (10,).into_dimension().into_dyn(),
                hint: "expected scores_prob_dist output shape to be equal to (10,)",
            }
            .into())
        }
    }

    /// Create a new ListNet instance with random weights.
    ///
    /// The weights are initialized using the He-Normal weight
    /// initializer, the biases are initialized to 0.
    //TODO[maybe] run some tests to see weather or not Glorot or He
    //            Uniform/Normal initialization works better for us.
    #[allow(dead_code)] //FIXME
    pub fn new_with_random_weights() -> Self {
        let mut rng = rand::thread_rng();

        let dense1 = Dense::new(
            he_normal_weights_init(&mut rng, Self::INPUT_NR_FEATURES, 48),
            Array1::zeros((48,)),
            Relu,
        )
        .unwrap();

        let dense2 = Dense::new(
            he_normal_weights_init(&mut rng, 48, 8),
            Array1::zeros((8,)),
            Relu,
        )
        .unwrap();

        let scores = Dense::new(
            he_normal_weights_init(&mut rng, 8, 1),
            Array1::zeros((1,)),
            Linear,
        )
        .unwrap();

        let prob_dist = Dense::new(
            he_normal_weights_init(&mut rng, Self::INPUT_NR_DOCUMENTS, Self::INPUT_NR_DOCUMENTS),
            Array1::zeros((Self::INPUT_NR_DOCUMENTS,)),
            Softmax::default(),
        )
        .unwrap();

        ListNet {
            dense1,
            dense2,
            scores,
            prob_dist,
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
        let (dense1_y, dense1_z) = self.dense1.run(&inputs, for_back_propagation);
        let (dense2_y, dense2_z) = self.dense2.run(&dense1_y, for_back_propagation);
        let (scores, _) = self.scores.run(&dense2_y, false);
        debug_assert_eq!(scores.shape()[1], 1);

        // flattens the array by removing axis 1
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
        let (prob_dist_y, _) = self.prob_dist.run(scores, false);
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

        let outputs = self.calculate_final_scores(&Array1::from(inputs));
        debug_assert!(outputs.is_standard_layout());
        let mut outputs = outputs.into_raw_vec();
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

    /// Computes the gradients and loss for given inputs and target prob. dist.
    ///
    /// # Panics
    ///
    /// If inputs and relevances are not for exactly [`Self::INPUT_NR_DOCUMENTS`]
    /// documents.
    fn gradients_for_query(&self, sample: Sample) -> (GradientSet, f32) {
        let Sample {
            inputs,
            target_prob_dist,
        } = sample;
        assert_eq!(inputs.shape()[0], ListNet::INPUT_NR_DOCUMENTS);
        assert_eq!(target_prob_dist.len(), ListNet::INPUT_NR_DOCUMENTS);
        let results = self.forward_pass(inputs);
        //FIXME[followup PR] if we don't track loss in XaynNet when used in the app we don't need to calculate it.
        let loss = kl_divergence(target_prob_dist.view(), results.prob_dist_y.view());
        let gradients = self.back_propagation(inputs, target_prob_dist, results);
        (gradients, loss)
    }

    /// Run the the forward pass of back-propagation.
    fn forward_pass(&self, inputs: ArrayView2<f32>) -> ForwardPassData {
        let (scores_y, partial_forward_pass) = self.calculate_intermediate_scores(inputs, true);
        let prob_dist_y = self.calculate_final_scores(&scores_y);

        ForwardPassData {
            partial_forward_pass: partial_forward_pass.unwrap(),
            scores_y,
            prob_dist_y,
        }
    }

    /// Run back propagation based on given inputs, target prob. dist. and forward pass data.
    fn back_propagation(
        &self,
        inputs: ArrayView2<f32>,
        target_prob_dist: ArrayView1<f32>,
        forward_pass: ForwardPassData,
    ) -> GradientSet {
        let ForwardPassData {
            partial_forward_pass:
                PartialForwardPassData {
                    dense1_y,
                    dense1_z,
                    dense2_y,
                    dense2_z,
                },
            scores_y,
            prob_dist_y,
        } = forward_pass;

        let nr_documents = inputs.shape()[0];
        let derivatives_of_clipping =
            prob_dist_y.mapv(|v| (f32::EPSILON..=1.).contains(&v) as u8 as f32);
        let p_cost_and_prob_dist = (prob_dist_y - target_prob_dist) * derivatives_of_clipping;

        let d_prob_dist = self
            .prob_dist
            .gradients_from_partials_1d(scores_y.view(), p_cost_and_prob_dist.view());

        // The activation functions of `scores` is the identity function (linear) so
        // it's gradient is 1 at all inputs and we can omit it.
        let p_scores = self.prob_dist.weights().dot(&p_cost_and_prob_dist);

        let mut d_scores = DenseGradientSet::zero_gradients_for(&self.scores);
        let mut d_dense2 = DenseGradientSet::zero_gradients_for(&self.dense2);
        let mut d_dense1 = DenseGradientSet::zero_gradients_for(&self.dense1);

        for row in 0..nr_documents {
            // From here on training is "split" into multiple parallel "path" using
            // shared weights (hence why we add up the gradients).
            let p_scores = p_scores.slice(s![row..row + 1]);

            let d_scores_part = self
                .scores
                .gradients_from_partials_1d(dense2_y.slice(s![row, ..]), p_scores);
            d_scores += d_scores_part;

            let p_dense2 = self.scores.weights().dot(&p_scores)
                * Relu::partial_derivatives_at(&dense2_z.slice(s![row, ..]));
            let d_dense2_part = self
                .dense2
                .gradients_from_partials_1d(dense1_y.slice(s![row, ..]), p_dense2.view());
            d_dense2 += d_dense2_part;

            let p_dense1 = self.dense2.weights().dot(&p_dense2)
                * Relu::partial_derivatives_at(&dense1_z.slice(s![row, ..]));
            let d_dense1_part = self
                .dense1
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
    pub fn add_gradients(&mut self, gradients: GradientSet) {
        let GradientSet {
            dense1,
            dense2,
            scores,
            prob_dist,
        } = gradients;

        self.prob_dist.add_gradients(&prob_dist);
        self.scores.add_gradients(&scores);
        self.dense2.add_gradients(&dense2);
        self.dense1.add_gradients(&dense1);
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

/// Data collected when running the forward pass of back-propagation.
///
/// It extends [`PartialForwardPassData`] by dereferencing to it,
/// so you can treat it as if it also contains all fields from
/// [`PartialForwardPassData`].
///
/// By convention the `y` refers to the outputs of a layer and
/// `z` to the outputs of a layer before that layers activation
/// function has been applied to them.
struct ForwardPassData {
    partial_forward_pass: PartialForwardPassData,
    scores_y: Array1<f32>,
    prob_dist_y: Array1<f32>,
}

/// Some of the data collected when running the forward pass of back-propagation.
struct PartialForwardPassData {
    dense1_y: Array2<f32>,
    dense1_z: Array2<f32>,
    dense2_y: Array2<f32>,
    dense2_z: Array2<f32>,
}

/// A set of gradients for all parameters of `ListNet`.
#[cfg_attr(test, derive(Debug, Clone))]
pub struct GradientSet {
    dense1: DenseGradientSet,
    dense2: DenseGradientSet,
    scores: DenseGradientSet,
    prob_dist: DenseGradientSet,
}

impl GradientSet {
    /// Merges all gradient sets computed for one batch of training data.
    ///
    /// This will create the mean of each gradient across all gradient sets.
    ///
    /// If there are no gradient sets in the input then `None` is returned.
    fn mean_of(gradient_sets: Vec<GradientSet>) -> Option<GradientSet> {
        let count = gradient_sets.len() as f32;
        gradient_sets
            .into_iter()
            .map(|gradient_set| gradient_set / count)
            .reduce(Add::add)
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

/// Trainer allowing the training a ListNets.
pub struct ListNetTrainer<D, C, O>
where
    D: DataSource,
    C: Callbacks,
    O: Optimizer,
{
    list_net: ListNet,
    data_source: D,
    callbacks: C,
    optimizer: O,
}

impl<D, C, O> ListNetTrainer<D, C, O>
where
    D: DataSource,
    C: Callbacks,
    O: Optimizer,
{
    /// Creates a new `ListNetTrainer` instance.
    #[allow(dead_code)] //FIXME
    pub fn new(list_net: ListNet, data_source: D, callbacks: C, optimizer: O) -> Self {
        Self {
            list_net,
            data_source,
            callbacks,
            optimizer,
        }
    }

    /// Trains for the given number of epochs with given batch size.
    ///
    /// This will use the `list_net`, `data_source`, `callbacks` and `optimizer` used
    /// to create this `ListNetTrainer`.
    //FIXME[followup PR?] remove dead code annotation
    #[allow(dead_code)]
    pub fn train(&mut self, epochs: usize, batch_size: usize) {
        self.callbacks.begin_of_training(&self.list_net);

        for _ in 0..epochs {
            self.data_source.reset();
            self.callbacks.begin_of_epoch(&self.list_net);

            while self.train_next_batch(batch_size) {}
            let evaluation_results = self.evaluate_epoch(kl_divergence);

            self.callbacks
                .end_of_epoch(&self.list_net, evaluation_results);
        }

        self.callbacks.end_of_training(&self.list_net);
    }

    /// Trains on next batch of samples.
    ///
    /// If there where no more batches to train on this return false, else this
    /// returns true.
    fn train_next_batch(&mut self, batch_size: usize) -> bool {
        let ListNetTrainer {
            list_net,
            data_source,
            callbacks,
            optimizer,
        } = self;

        let batch = data_source.next_training_batch(batch_size);
        if batch.is_empty() {
            return false;
        }

        callbacks.begin_of_batch();

        let mut losses = Vec::new();
        let gradient_sets = batch
            .into_iter()
            .map(|sample| {
                let (gradients, loss) = list_net.gradients_for_query(sample);
                losses.push(loss);
                gradients
            })
            .collect();

        optimizer.apply_gradients(list_net, gradient_sets);

        callbacks.end_of_batch(losses);

        true
    }

    /// Returns the mean cost over all samples of the evaluation dataset using the given cost function.
    fn evaluate_epoch(
        &mut self,
        cost_function: fn(ArrayView1<f32>, ArrayView1<f32>) -> f32,
    ) -> Option<f32> {
        let (acc, count) = iter::from_fn(|| {
            let list_net = &self.list_net;
            self.data_source.next_evaluation_sample().map(
                |Sample {
                     inputs,
                     target_prob_dist,
                 }| {
                    let (scores_y, _) = list_net.calculate_intermediate_scores(inputs, false);
                    let prob_dist_y = list_net.calculate_final_scores(&scores_y);
                    cost_function(target_prob_dist, prob_dist_y.view())
                },
            )
        })
        .fold((0f32, 0), |(acc, count), cost| (acc + cost, count + 1));

        (count > 0).then(|| acc / count as f32)
    }
}

/// A single sample you can train on.
pub struct Sample<'a> {
    /// Input used for training.
    ///
    /// (At least for now) this must be a `(10, 50)` array
    /// view.
    pub inputs: ArrayView2<'a, f32>,

    /// Target probability distribution.
    ///
    /// Needs to have the same length as `inputs.shape()[0]`.
    pub target_prob_dist: ArrayView1<'a, f32>,
}

/// A source of training and evaluation data.
pub trait DataSource {
    /// Resets the "iteration" of training and evaluation samples.
    ///
    /// This is allowed to also *change* the training and evaluation
    /// samples and/or their order. E.g. this could shuffle them
    /// before every epoch.
    ///
    /// This is called at the *begin* of every epoch.
    fn reset(&mut self);

    /// Returns the next `batch_size` number of training samples.
    ///
    /// Returns a empty vector once all training samples have been returned.
    ///
    /// # Panics
    ///
    /// A batch size of 0 is not valid. And implementors are allowed to
    /// panic if it's passed in.
    fn next_training_batch(&mut self, batch_size: usize) -> Vec<Sample>;

    /// Returns the next evaluation sample.
    ///
    /// Returns `None` once all training sample have been returned.
    fn next_evaluation_sample(&mut self) -> Option<Sample>;
}

/// A trait providing various callbacks used during training.
pub trait Callbacks {
    /// Called before started training a batch.
    fn begin_of_batch(&mut self);

    /// Called after training a batch, the loss for each sample will be passed in.
    fn end_of_batch(&mut self, losses: Vec<f32>);

    /// Called at the begin of each epoch.
    fn begin_of_epoch(&mut self, list_net: &ListNet);

    /// Called at the end of each epoch with the mean of running the evaluation with KL-Divergence.
    ///
    /// The passed in reference to `list_net` can be used to e.g. bump the intermediate training
    /// result every 10 epochs.
    fn end_of_epoch(&mut self, list_net: &ListNet, mean_kv_divergence_evaluation: Option<f32>);

    /// Called at the begin of training.
    fn begin_of_training(&mut self, list_net: &ListNet);

    /// Called after training finished.
    fn end_of_training(&mut self, list_net: &ListNet);
}

/// Prepares the inputs for training.
///
/// This is a static helper function which is meant to be used by `DataSource`
/// implementations to filter and prepare inputs.
///
/// This will:
///
/// - returns `None` if there are less then 10 documents
/// - truncates inputs and relevances to 10 documents
/// - returns `None` if there are no relevant documents (after truncating)
/// - calculates the target distribution based on the relevances
#[allow(dead_code)]
pub fn prepare_inputs_for_training<'a>(
    inputs: &'a Array2<f32>,
    relevances: &'a [Relevance],
) -> Option<(ArrayView2<'a, f32>, Array1<f32>)> {
    if inputs.shape()[0] < ListNet::INPUT_NR_DOCUMENTS {
        None
    } else {
        Some((
            inputs.slice(s![..ListNet::INPUT_NR_DOCUMENTS, ..]),
            prepare_target_prob_dist(&relevances[..ListNet::INPUT_NR_DOCUMENTS])?,
        ))
    }
}

/// Turns the given relevances into a probability distribution.
///
/// This is a static helper function which is meant to be used by `DataSource`
/// implementations to filter and prepare inputs.
///
/// If there is no relevant document in the inputs `None` is returned.
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

#[cfg(test)]
mod tests {

    use std::{collections::HashSet, f32::consts::SQRT_2};

    use ndarray::{arr1, arr2, Array, IxDyn};
    use once_cell::sync::Lazy;

    use super::{ndlayers::ActivationFunction, optimizer::MiniBatchSgd};

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

    struct VecDataSource {
        training_data_idx: usize,
        training_data: Vec<(Array2<f32>, Array1<f32>)>,
        evaluation_data_idx: usize,
        evaluation_data: Vec<(Array2<f32>, Array1<f32>)>,
    }

    impl VecDataSource {
        fn new(
            training_data: Vec<(Array2<f32>, Array1<f32>)>,
            evaluation_data: Vec<(Array2<f32>, Array1<f32>)>,
        ) -> Self {
            Self {
                training_data_idx: 0,
                training_data,
                evaluation_data_idx: 0,
                evaluation_data,
            }
        }
    }

    impl DataSource for VecDataSource {
        fn reset(&mut self) {
            self.training_data_idx = 0;
            self.evaluation_data_idx = 0;
        }

        fn next_training_batch(&mut self, batch_size: usize) -> Vec<Sample> {
            assert!(batch_size > 0);
            let end_idx = self.training_data_idx + batch_size;
            if end_idx <= self.training_data.len() {
                let start_idx = self.training_data_idx;
                self.training_data_idx = end_idx;

                self.training_data[start_idx..end_idx]
                    .iter()
                    .map(|(inputs, target_prop_dist)| Sample {
                        inputs: inputs.view(),
                        target_prob_dist: target_prop_dist.view(),
                    })
                    .collect()
            } else {
                Vec::new()
            }
        }

        fn next_evaluation_sample(&mut self) -> Option<Sample> {
            if self.evaluation_data_idx < self.evaluation_data.len() {
                let idx = self.evaluation_data_idx;
                self.evaluation_data_idx += 1;

                let data = &self.evaluation_data[idx];
                Some(Sample {
                    inputs: data.0.view(),
                    target_prob_dist: data.1.view(),
                })
            } else {
                None
            }
        }
    }

    struct TestCallbacks {
        evaluation_results: Vec<Option<f32>>,
    }

    impl TestCallbacks {
        fn new() -> Self {
            Self {
                evaluation_results: Vec::new(),
            }
        }
    }

    impl Callbacks for TestCallbacks {
        fn begin_of_batch(&mut self) {
            dbg!("begin batch");
        }

        fn end_of_batch(&mut self, losses: Vec<f32>) {
            dbg!(losses);
        }

        fn begin_of_epoch(&mut self, _list_net: &ListNet) {
            dbg!("begin epoch");
        }

        fn end_of_epoch(
            &mut self,
            _list_net: &ListNet,
            mean_kv_divergence_evaluation: Option<f32>,
        ) {
            dbg!(mean_kv_divergence_evaluation);
            self.evaluation_results.push(mean_kv_divergence_evaluation);
        }

        fn begin_of_training(&mut self, _list_net: &ListNet) {
            dbg!("begin of training");
        }

        fn end_of_training(&mut self, _list_net: &ListNet) {
            dbg!("end of training");
        }
    }

    //FIXME[follow up PR] create better tests
    #[ignore = "fails on android unclear reasons, fixed in followup PR"]
    #[test]
    fn test_training_list_net_does_not_panic() {
        use Relevance::{High, Low, Medium};
        let list_net = LIST_NET.clone();

        let inputs = Array1::from(SAMPLE_INPUTS.to_vec())
            .into_shape((10, 50))
            .unwrap();

        let relevances = vec![Low, Low, Medium, Medium, Low, Medium, High, Low, High, High];
        let data_frame = (inputs, prepare_target_prob_dist(&relevances).unwrap());

        let training_data = vec![data_frame.clone(), data_frame.clone(), data_frame.clone()];
        let test_data = vec![data_frame];

        let nr_epochs = 5;
        let data_source = VecDataSource::new(training_data, test_data);
        let callbacks = TestCallbacks::new();
        let optimizer = MiniBatchSgd { learning_rate: 0.1 };
        let mut trainer = ListNetTrainer::new(list_net, data_source, callbacks, optimizer);
        trainer.train(nr_epochs, 3);

        let evaluation_results = trainer.callbacks.evaluation_results;
        assert_eq!(evaluation_results.len(), nr_epochs);
        assert!(
            evaluation_results.iter().all(|v| !v.unwrap().is_nan()),
            "contains NaN values {:?}",
            evaluation_results
        );
        assert!(
            evaluation_results.first() > evaluation_results.last(),
            "unexpected regression of training: {:?}",
            evaluation_results
        );
    }

    #[test]
    fn test_gradients_merge_batch() {
        let res = GradientSet::mean_of(vec![]);
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

        let a2 = GradientSet::mean_of(vec![a.clone()]).unwrap();

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

        let g = GradientSet::mean_of(vec![a, b]).unwrap();

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
        let res = prepare_target_prob_dist(&relevances);
        assert!(res.is_none());

        let relevances = vec![Low, Low, Medium, Medium, Low, Medium, High, Low, High, High];
        let dist = prepare_target_prob_dist(&relevances).unwrap();
        assert_approx_eq!(
            f32,
            dist,
            [
                0.051_708_337,
                0.046_787_64,
                0.115_079_02,
                0.104_127_81,
                0.034_661_137,
                0.085_252_635,
                0.209_687_65,
                0.025_677_6,
                0.171_677_72,
                0.155_340_43,
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

        let (processed_inputs, dist) = prepare_inputs_for_training(&inputs, &relevances).unwrap();

        assert_approx_eq!(f32, &inputs, processed_inputs);

        assert_approx_eq!(
            f32,
            dist,
            [
                0.051_708_337,
                0.046_787_64,
                0.115_079_02,
                0.104_127_81,
                0.034_661_137,
                0.085_252_635,
                0.209_687_65,
                0.025_677_6,
                0.171_677_72,
                0.155_340_43,
            ]
        );

        assert!(prepare_inputs_for_training(&inputs, &[Low; 10]).is_none());

        let few_inputs = Array1::from(SAMPLE_INPUTS_TO_FEW.to_vec())
            .into_shape((3, 50))
            .unwrap();

        assert!(prepare_inputs_for_training(&few_inputs, &relevances).is_none());
    }

    #[test]
    fn test_random_weights_initialization() {
        let ListNet {
            dense1,
            dense2,
            scores,
            prob_dist,
        } = ListNet::new_with_random_weights();

        test_layer(&dense1);
        test_layer(&dense2);
        test_layer(&scores);
        test_layer(&prob_dist);

        fn test_layer(layer: &Dense<impl ActivationFunction<f32>>) {
            for b in layer.bias().iter() {
                assert_approx_eq!(f32, b, 0.0, ulps = 9)
            }
            let weights = layer.weights();
            let std = SQRT_2 / (weights.shape()[0] as f32).sqrt();
            let limit = 2. * std;
            for &w in weights.iter() {
                assert!(
                    -limit <= w && w <= limit,
                    "out of bound weight: {} <= {} <= {}",
                    -limit,
                    w,
                    limit
                );
            }
        }
    }
}
