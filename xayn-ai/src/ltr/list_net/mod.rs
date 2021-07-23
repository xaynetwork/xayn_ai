//! ListNet implementation using the NdArray crate.

use std::{
    error::Error as StdError,
    io::{Read, Write},
    ops::{Add, Div, MulAssign},
    path::Path,
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

#[cfg(test)]
use self::ndutils::io::{BinParamsWithScope, FailedToRetrieveParams};

mod ndlayers;
//Pub for dev-tool
pub mod ndutils;

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
#[derive(Clone)]
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
    pub const INPUT_NR_FEATURES: Ix = 50;

    /// Shape of input: `INPUT_NR_DOCUMENTS` x `INPUT_NR_FEATURES`
    const INPUT_SHAPE: [Ix; 2] = [Self::INPUT_NR_DOCUMENTS, Self::INPUT_NR_FEATURES];

    /// Load ListNet from file at given path.
    pub fn deserialize_from_file(path: impl AsRef<Path>) -> Result<Self, LoadingListNetFailed> {
        let params = BinParams::deserialize_from_file(path)?;
        Self::load(params)
    }

    /// Load ListNet from byte reader.
    pub fn deserialize_from(params_source: impl Read) -> Result<Self, LoadingListNetFailed> {
        let params = BinParams::deserialize_from(params_source)?;
        Self::load(params)
    }

    fn create_bin_params(self) -> BinParams {
        let mut params = BinParams::default();
        self.dense1.store_params(params.with_scope("dense_1"));
        self.dense2.store_params(params.with_scope("dense_2"));
        self.scores.store_params(params.with_scope("scores"));
        self.prob_dist
            .store_params(params.with_scope("scores_prob_dist"));
        params
    }

    /// Serializes the ListNet into given writer.
    pub fn serialize_into(self, writer: impl Write) -> Result<(), Box<bincode::ErrorKind>> {
        self.create_bin_params().serialize_into(writer)
    }

    /// Serializes the ListNet into given file.
    pub fn serialize_into_file(
        self,
        path: impl AsRef<Path>,
    ) -> Result<(), Box<bincode::ErrorKind>> {
        self.create_bin_params().serialize_into_file(path)
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
    pub fn new_with_random_weights() -> Self {
        let mut rng = rand::thread_rng();

        let dense1 = Dense::new(
            he_normal_weights_init(&mut rng, (Self::INPUT_NR_FEATURES, 48)),
            Array1::zeros((48,)),
            Relu,
        )
        .unwrap();

        let dense2 = Dense::new(
            he_normal_weights_init(&mut rng, (48, 8)),
            Array1::zeros((8,)),
            Relu,
        )
        .unwrap();

        let scores = Dense::new(
            he_normal_weights_init(&mut rng, (8, 1)),
            Array1::zeros((1,)),
            Linear,
        )
        .unwrap();

        let prob_dist = Dense::new(
            he_normal_weights_init(
                &mut rng,
                (Self::INPUT_NR_DOCUMENTS, Self::INPUT_NR_DOCUMENTS),
            ),
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

    #[cfg(test)]
    pub(crate) fn store(self, mut bin_params: BinParamsWithScope) {
        let Self {
            dense1,
            dense2,
            scores,
            prob_dist,
        } = self;
        dense1.store(bin_params.with_scope("dense1"));
        dense2.store(bin_params.with_scope("dense2"));
        scores.store(bin_params.with_scope("scores"));
        prob_dist.store(bin_params.with_scope("prob_dist"));
    }

    #[cfg(test)]
    pub(crate) fn load(mut bin_params: BinParamsWithScope) -> Result<Self, FailedToRetrieveParams> {
        let dense1 = DenseGradientSet::load(bin_params.with_scope("dense1"))?;
        let dense2 = DenseGradientSet::load(bin_params.with_scope("dense2"))?;
        let scores = DenseGradientSet::load(bin_params.with_scope("scores"))?;
        let prob_dist = DenseGradientSet::load(bin_params.with_scope("prob_dist"))?;

        Ok(Self {
            dense1,
            dense2,
            scores,
            prob_dist,
        })
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
    C: TrainingController,
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
    C: TrainingController,
    O: Optimizer,
{
    /// Creates a new `ListNetTrainer` instance.
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
    pub fn train(
        mut self,
        epochs: usize,
        batch_size: usize,
    ) -> Result<C::Outcome, TrainingError<D::Error, C::Error>> {
        self.callbacks
            .begin_of_training(epochs, &self.list_net)
            .map_err(TrainingError::Control)?;

        for _ in 0..epochs {
            let nr_batches = self
                .data_source
                .reset(batch_size)
                .map_err(TrainingError::Data)?;
            self.callbacks
                .begin_of_epoch(nr_batches, &self.list_net)
                .map_err(TrainingError::Control)?;

            while self.train_next_batch()? {}
            let evaluation_results = self.evaluate_epoch(kl_divergence)?;

            self.callbacks
                .end_of_epoch(&self.list_net, evaluation_results)
                .map_err(TrainingError::Control)?;
        }

        self.callbacks
            .end_of_training()
            .map_err(TrainingError::Control)?;
        self.callbacks
            .training_result(self.list_net)
            .map_err(TrainingError::Control)
    }

    /// Trains on next batch of samples.
    ///
    /// If there where no more batches to train on this return false, else this
    /// returns true.
    fn train_next_batch(&mut self) -> Result<bool, TrainingError<D::Error, C::Error>> {
        let ListNetTrainer {
            list_net,
            data_source,
            callbacks,
            optimizer,
        } = self;

        let batch = data_source
            .next_training_batch()
            .map_err(TrainingError::Data)?;
        if batch.is_empty() {
            return Ok(false);
        }

        callbacks.begin_of_batch().map_err(TrainingError::Control)?;

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

        callbacks
            .end_of_batch(losses)
            .map_err(TrainingError::Control)?;

        Ok(true)
    }

    /// Returns the mean cost over all samples of the evaluation dataset using the given cost function.
    fn evaluate_epoch(
        &mut self,
        cost_function: fn(ArrayView1<f32>, ArrayView1<f32>) -> f32,
    ) -> Result<Option<f32>, TrainingError<D::Error, C::Error>> {
        let list_net = &self.list_net;
        let mut costs = Vec::new();
        while let Some(Sample {
            inputs,
            target_prob_dist,
        }) = self
            .data_source
            .next_evaluation_sample()
            .map_err(TrainingError::Data)?
        {
            let (scores_y, _) = list_net.calculate_intermediate_scores(inputs, false);
            let prob_dist_y = list_net.calculate_final_scores(&scores_y);
            costs.push(cost_function(target_prob_dist, prob_dist_y.view()));
        }

        let count = costs.len() as f32;
        let mean =
            (count > 0.).then(|| costs.into_iter().fold(0f32, |acc, cost| acc + cost / count));
        Ok(mean)
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
    type Error: StdError + 'static;

    /// Resets/initializes the "iteration" of training and evaluation samples.
    ///
    /// This returns the expected number of batches in the next epoch.
    ///
    /// This is allowed to also *change* the training and evaluation
    /// samples and/or their order. E.g. this could shuffle them
    /// before every epoch.
    ///
    /// This is called at the *begin* of every epoch.
    fn reset(&mut self, batch_size: usize) -> Result<usize, Self::Error>;

    /// Returns the next batch of training samples.
    ///
    /// The batch will have the size last set when calling reset.
    ///
    /// Returns a empty vector once all training samples have been returned.
    ///
    /// If reset wasn't called before this should return a error.
    ///
    /// # Panics
    ///
    /// A batch size of 0 is not valid. And implementors are allowed to
    /// panic if it's passed in.
    fn next_training_batch(&mut self) -> Result<Vec<Sample>, Self::Error>;

    /// Returns the next evaluation sample.
    ///
    /// Returns `None` once all training sample have been returned.
    fn next_evaluation_sample(&mut self) -> Result<Option<Sample>, Self::Error>;
}

/// A trait providing various callbacks used during training.
pub trait TrainingController {
    type Error: StdError + 'static;
    type Outcome;

    /// Called before started training a batch.
    fn begin_of_batch(&mut self) -> Result<(), Self::Error>;

    /// Called after training a batch, the loss for each sample will be passed in.
    fn end_of_batch(&mut self, losses: Vec<f32>) -> Result<(), Self::Error>;

    /// Called at the begin of each epoch.
    fn begin_of_epoch(&mut self, nr_batches: usize, list_net: &ListNet) -> Result<(), Self::Error>;

    /// Called at the end of each epoch with the mean of running the evaluation with KL-Divergence.
    ///
    /// The passed in reference to `list_net` can be used to e.g. bump the intermediate training
    /// result every 10 epochs.
    fn end_of_epoch(
        &mut self,
        list_net: &ListNet,
        mean_kl_divergence_evaluation: Option<f32>,
    ) -> Result<(), Self::Error>;

    /// Called at the begin of training.
    fn begin_of_training(
        &mut self,
        nr_epochs: usize,
        list_net: &ListNet,
    ) -> Result<(), Self::Error>;

    /// Called after training finished.
    fn end_of_training(&mut self) -> Result<(), Self::Error>;

    /// Returns the result of the training.
    fn training_result(self, list_net: ListNet) -> Result<Self::Outcome, Self::Error>;
}

/// Error produced while training.
///
/// This is either a error from the `DataSource` or
/// the `TrainingController`.
#[derive(Error, Debug)]
pub enum TrainingError<DE, CE>
where
    DE: StdError + 'static,
    CE: StdError + 'static,
{
    #[error("Retrieving training/evaluation samples failed: {0}")]
    Data(#[source] DE),
    #[error("Training controller produced an error: {0}")]
    Control(#[source] CE),
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
pub fn prepare_inputs(inputs: &Array2<f32>) -> Option<ArrayView2<f32>> {
    if inputs.shape()[0] < ListNet::INPUT_NR_DOCUMENTS {
        None
    } else {
        Some(inputs.slice(s![..ListNet::INPUT_NR_DOCUMENTS, ..]))
    }
}

/// Turns the given relevances into a probability distribution.
///
/// This is a static helper function which is meant to be used by `DataSource`
/// implementations to filter and prepare inputs.
///
/// If there is no relevant document in the inputs `None` is returned.
pub fn prepare_target_prob_dist(relevance: &[Relevance]) -> Option<Array1<f32>> {
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

    use std::{
        collections::HashSet,
        convert::{Infallible, TryInto},
        env,
        f32::consts::SQRT_2,
        ffi::OsString,
        path::PathBuf,
    };

    use ndarray::{arr1, arr2, Array, IxDyn};
    use once_cell::sync::Lazy;
    use rand::{thread_rng, Rng};

    use super::{
        ndlayers::ActivationFunction,
        ndutils::io::{FlattenedArray, UnexpectedNumberOfDimensions},
        optimizer::MiniBatchSgd,
    };

    use super::*;

    const LIST_NET_BIN_PARAMS_PATH: &str = "../data/ltr_v0000/ltr.binparams";

    static LIST_NET: Lazy<ListNet> =
        Lazy::new(|| ListNet::deserialize_from_file(LIST_NET_BIN_PARAMS_PATH).unwrap());

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
        let params = BinParams::deserialize_from(EMPTY_BIN_PARAMS).unwrap();
        assert!(params.is_empty());

        let mut params = BinParams::deserialize_from(BIN_PARAMS_WITH_EMPTY_ARRAY_AND_KEY).unwrap();
        assert!(!params.is_empty());

        let array: Array<f32, IxDyn> = params.take("").unwrap();
        assert_eq!(array.shape(), &[0]);
    }

    #[test]
    fn test_keys() {
        let params = BinParams::deserialize_from(BIN_PARAMS_WITH_SOME_KEYS).unwrap();
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
        batch_size: usize,
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
                batch_size: 0,
                training_data_idx: 0,
                training_data,
                evaluation_data_idx: 0,
                evaluation_data,
            }
        }
    }

    #[derive(Error, Debug)]
    #[error("Batch Size 0 is not supported (or reset was not called)")]
    struct BatchSize0Error;

    impl DataSource for VecDataSource {
        type Error = BatchSize0Error;

        fn reset(&mut self, batch_size: usize) -> Result<usize, Self::Error> {
            if batch_size == 0 {
                return Err(BatchSize0Error);
            }
            self.batch_size = batch_size;
            self.training_data_idx = 0;
            self.evaluation_data_idx = 0;
            Ok(self.training_data.len() / batch_size)
        }

        fn next_training_batch(&mut self) -> Result<Vec<Sample>, Self::Error> {
            if self.batch_size == 0 {
                return Err(BatchSize0Error);
            }

            let end_idx = self.training_data_idx + self.batch_size;
            if end_idx <= self.training_data.len() {
                let start_idx = self.training_data_idx;
                self.training_data_idx = end_idx;

                let samples = self.training_data[start_idx..end_idx]
                    .iter()
                    .map(|(inputs, target_prop_dist)| Sample {
                        inputs: inputs.view(),
                        target_prob_dist: target_prop_dist.view(),
                    })
                    .collect();
                Ok(samples)
            } else {
                Ok(Vec::new())
            }
        }

        fn next_evaluation_sample(&mut self) -> Result<Option<Sample>, Self::Error> {
            if self.evaluation_data_idx < self.evaluation_data.len() {
                let idx = self.evaluation_data_idx;
                self.evaluation_data_idx += 1;

                let data = &self.evaluation_data[idx];
                Ok(Some(Sample {
                    inputs: data.0.view(),
                    target_prob_dist: data.1.view(),
                }))
            } else {
                Ok(None)
            }
        }
    }

    struct TestController {
        evaluation_results: Vec<Option<f32>>,
    }

    impl TestController {
        fn new() -> Self {
            Self {
                evaluation_results: Vec::new(),
            }
        }
    }

    impl TrainingController for TestController {
        type Error = Infallible;

        type Outcome = (Self, ListNet);

        fn begin_of_batch(&mut self) -> Result<(), Self::Error> {
            eprintln!("begin batch");
            Ok(())
        }

        fn end_of_batch(&mut self, losses: Vec<f32>) -> Result<(), Self::Error> {
            eprintln!("end of batch");
            dbg!(losses);
            Ok(())
        }

        fn begin_of_epoch(
            &mut self,
            _nr_batches: usize,
            _list_net: &ListNet,
        ) -> Result<(), Self::Error> {
            eprintln!("begin of epoch");
            Ok(())
        }

        fn end_of_epoch(
            &mut self,
            _list_net: &ListNet,
            mean_kv_divergence_evaluation: Option<f32>,
        ) -> Result<(), Self::Error> {
            eprintln!("end of epoch");
            dbg!(mean_kv_divergence_evaluation);
            self.evaluation_results.push(mean_kv_divergence_evaluation);
            Ok(())
        }

        fn begin_of_training(
            &mut self,
            _nr_epochs: usize,
            _list_net: &ListNet,
        ) -> Result<(), Self::Error> {
            eprintln!("begin of training");
            Ok(())
        }

        fn end_of_training(&mut self) -> Result<(), Self::Error> {
            eprintln!("end of training");
            Ok(())
        }

        fn training_result(self, list_net: ListNet) -> Result<Self::Outcome, Self::Error> {
            Ok((self, list_net))
        }
    }

    //FIXME[follow up PR] remove test, this is just a sanity check due to the NaN bug
    //WARNING this can take forever!
    #[ignore = "takes to long to run"]
    #[test]
    fn test_ndarray_matrix_multiply() {
        const EPSILON: f32 = f32::EPSILON * 5.;
        const ULPS: i32 = 4;

        let mut rng = thread_rng();

        test(&mut rng, dbg!((48, 8, 1)));
        test(&mut rng, dbg!((10, 50, 48)));
        test(&mut rng, dbg!((10, 1, 10)));
        test(&mut rng, dbg!((8, 1, 1)));

        fn test(rng: &mut impl Rng, (ind, mid, out): (usize, usize, usize)) {
            drop(crate::embedding::qambert::tests::qambert());
            let a = he_normal_weights_init(rng, (ind, mid));
            let b = he_normal_weights_init(rng, (mid, out));
            let expected = naive_matmul(a.view(), b.view());
            let res = a.dot(&b);
            assert_approx_eq!(f32, &res, expected, epsilon = EPSILON, ulps = ULPS);
        }

        fn naive_matmul(a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
            assert_eq!(a.shape()[1], b.shape()[0]);

            let new_rows = a.shape()[0];
            let new_cols = b.shape()[1];

            let mut res = Array2::from_elem((new_rows, new_cols), 4.25f32);

            for row in 0..a.shape()[0] {
                for column in 0..b.shape()[1] {
                    let left_vec = a.slice(s![row, ..]);
                    let right_vec = b.slice(s![.., column]);
                    assert_eq!(left_vec.len(), right_vec.len());
                    res[[row, column]] = left_vec
                        .iter()
                        .zip(right_vec.iter())
                        .fold(0f32, |acc, (&l, &r)| acc + l * r);
                }
            }
            res
        }
    }

    #[test]
    fn test_training_list_net_is_reproducible_for_same_inputs_and_state() {
        //FIXME remove once it no longer causes failure
        drop(crate::embedding::qambert::tests::qambert());

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
        let batch_size = 1;

        // let (ctrl0, ln0) = {
        //     let data_source = VecDataSource::new(training_data.clone(), test_data.clone());
        //     let callbacks = TestController::new();
        //     let optimizer = MiniBatchSgd { learning_rate: 0.1 };
        //     let trainer = ListNetTrainer::new(list_net.clone(), data_source, callbacks, optimizer);
        //     trainer.train(nr_epochs, batch_size).unwrap()
        // };
        let (ctrl1, ln1) = {
            let data_source = VecDataSource::new(training_data.clone(), test_data.clone());
            let callbacks = TestController::new();
            let optimizer = MiniBatchSgd { learning_rate: 0.1 };
            let trainer = ListNetTrainer::new(list_net.clone(), data_source, callbacks, optimizer);
            trainer.train(nr_epochs, batch_size).unwrap()
        };
        let (ctrl2, ln2) = {
            let data_source = VecDataSource::new(training_data, test_data);
            let callbacks = TestController::new();
            let optimizer = MiniBatchSgd { learning_rate: 0.1 };
            let trainer = ListNetTrainer::new(list_net, data_source, callbacks, optimizer);
            trainer.train(nr_epochs, batch_size).unwrap()
        };

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            use std::is_x86_feature_detected;
            if is_x86_feature_detected!("fma") {
                eprintln!("USES: fma");
            } else if is_x86_feature_detected!("avx") {
                eprintln!("USES: avx");
            } else if is_x86_feature_detected!("sse2") {
                eprintln!("USES: sse2");
            }
        }

        assert_approx_eq!(f32, ln1.dense1.weights(), ln2.dense1.weights());
        assert_approx_eq!(f32, ln1.dense1.bias(), ln2.dense1.bias());
        assert_approx_eq!(f32, ln1.dense2.weights(), ln2.dense2.weights());
        assert_approx_eq!(f32, ln1.dense2.bias(), ln2.dense2.bias());
        assert_approx_eq!(f32, ln1.scores.weights(), ln2.scores.weights());
        assert_approx_eq!(f32, ln1.scores.bias(), ln2.scores.bias());
        assert_approx_eq!(f32, ln1.prob_dist.weights(), ln2.prob_dist.weights());
        assert_approx_eq!(f32, ln1.prob_dist.bias(), ln2.prob_dist.bias());

        assert_approx_eq!(f32, &ctrl1.evaluation_results, &ctrl2.evaluation_results);

        assert!(
            ctrl1
                .evaluation_results
                .iter()
                .all(|v| !v.unwrap().is_nan()),
            "contains NaN values {:?}",
            ctrl1.evaluation_results
        );
    }

    struct BinParamsEqTestGuard {
        params: BinParams,
        write_to_path_on_normal_drop: Option<PathBuf>,
    }

    impl BinParamsEqTestGuard {
        fn setup(path: impl AsRef<Path>) -> Self {
            let rewrite_instead_of_test =
                env::var_os("LTR_LIST_NET_TRAINING_INTERMEDIATES_REWRITE")
                    == Some(OsString::from("1"));
            if rewrite_instead_of_test {
                Self {
                    params: BinParams::default(),
                    write_to_path_on_normal_drop: Some(path.as_ref().to_owned()),
                }
            } else {
                Self {
                    params: BinParams::deserialize_from_file(path).unwrap(),
                    write_to_path_on_normal_drop: None,
                }
            }
        }

        fn do_rewrite(&self) -> bool {
            self.write_to_path_on_normal_drop.is_some()
        }

        fn assert_array_eq<D>(&mut self, name: &str, array: &Array<f32, D>)
        where
            FlattenedArray<f32>:
                TryInto<Array<f32, D>, Error = UnexpectedNumberOfDimensions> + From<Array<f32, D>>,
            D: Dimension,
        {
            if self.do_rewrite() {
                self.params.insert(name, array.clone());
            } else {
                eprintln!("Assert approx eq of {}.", name);
                let expected = self.params.take::<Array<f32, D>>(name).unwrap();
                assert_approx_eq!(f32, array, expected);
            }
        }

        fn assert_gradient_eq(&mut self, gradients: GradientSet) {
            if self.do_rewrite() {
                gradients.store(self.params.with_scope("gradients"));
            } else {
                eprintln!("Assert approx eq of gradients.");
                let expected = GradientSet::load(self.params.with_scope("gradients")).unwrap();

                assert_approx_eq!(
                    f32,
                    &gradients.dense1.weight_gradients,
                    &expected.dense1.weight_gradients
                );
                assert_approx_eq!(
                    f32,
                    &gradients.dense1.bias_gradients,
                    &expected.dense1.bias_gradients
                );
                assert_approx_eq!(
                    f32,
                    &gradients.dense2.weight_gradients,
                    &expected.dense2.weight_gradients
                );
                assert_approx_eq!(
                    f32,
                    &gradients.dense2.bias_gradients,
                    &expected.dense2.bias_gradients
                );
                assert_approx_eq!(
                    f32,
                    &gradients.scores.weight_gradients,
                    &expected.scores.weight_gradients
                );
                assert_approx_eq!(
                    f32,
                    &gradients.scores.bias_gradients,
                    &expected.scores.bias_gradients
                );
                assert_approx_eq!(
                    f32,
                    &gradients.prob_dist.weight_gradients,
                    &expected.prob_dist.weight_gradients
                );
                assert_approx_eq!(
                    f32,
                    &gradients.prob_dist.bias_gradients,
                    &expected.prob_dist.bias_gradients
                );
            }
        }
    }

    impl Drop for BinParamsEqTestGuard {
        fn drop(&mut self) {
            if std::thread::panicking() {
                return;
            }

            if let Some(path) = self.write_to_path_on_normal_drop.take() {
                self.params.serialize_into_file(path).unwrap();
            }
        }
    }

    macro_rules! assert_trace_array {
        ($inout:ident =?= $($array:ident),+) => ($(
            $inout.assert_array_eq(stringify!($array), &$array);
        )*);
    }

    #[test]
    fn test_training_with_preset_initial_state_and_input_produces_expected_results() {
        //FIXME remove once it no longer causes failure
        drop(crate::embedding::qambert::tests::qambert());

        use Relevance::{High, Low, Medium};

        let list_net = LIST_NET.clone();

        let inputs = Array1::from(SAMPLE_INPUTS.to_vec())
            .into_shape((10, 50))
            .unwrap();

        let relevances = vec![Low, Low, Medium, Medium, Low, Medium, High, Low, High, High];

        let mut test_guard = BinParamsEqTestGuard::setup(
            "../data/ltr_test_data_v0000/check_training_intermediates.binparams",
        );

        // Run computation steps by hand to get *all* intermediate values.
        let target_prob_dist = prepare_target_prob_dist(&relevances).unwrap();
        assert_trace_array!(test_guard =?= target_prob_dist);
        let (dense1_y, dense1_z) = list_net.dense1.run(&inputs, true);
        let dense1_z = dense1_z.unwrap();
        assert_trace_array!(test_guard =?= dense1_y, dense1_z);
        let (dense2_y, dense2_z) = list_net.dense2.run(&dense1_y, true);
        let dense2_z = dense2_z.unwrap();
        assert_trace_array!(test_guard =?= dense2_y, dense2_z);
        let (scores_y, scores_z) = list_net.scores.run(&dense2_y, true);
        let scores_z = scores_z.unwrap();
        assert_trace_array!(test_guard =?= scores_y, scores_z);

        let scores_y = scores_y.index_axis_move(Axis(1), 0);

        let (prob_dist_y, prob_dist_z) = list_net.prob_dist.run(&scores_y, true);
        let prob_dist_z = prob_dist_z.unwrap();
        assert_trace_array!(test_guard =?= prob_dist_y, prob_dist_z);
        let results = ForwardPassData {
            partial_forward_pass: PartialForwardPassData {
                dense1_y,
                dense1_z,
                dense2_y,
                dense2_z,
            },
            scores_y,
            prob_dist_y,
        };
        let gradients = list_net.back_propagation(inputs.view(), target_prob_dist.view(), results);
        test_guard.assert_gradient_eq(gradients);
    }

    //FIXME[follow up PR] create better tests
    #[ignore = "fails on android unclear reasons, fixed in followup PR"]
    #[test]
    fn test_training_list_net_does_not_panic() {
        use Relevance::{High, Low, Medium};

        //LIST_NET.clone() (sanity check)
        let list_net = ListNet::deserialize_from_file(LIST_NET_BIN_PARAMS_PATH).unwrap();

        let inputs = Array1::from(SAMPLE_INPUTS.to_vec())
            .into_shape((10, 50))
            .unwrap();

        // Not very good checksum :-) (sanity check)
        let sum: f32 = inputs.iter().sum();
        assert_approx_eq!(f32, sum, 1666.1575);

        let relevances = vec![Low, Low, Medium, Medium, Low, Medium, High, Low, High, High];
        let data_frame = (inputs, prepare_target_prob_dist(&relevances).unwrap());

        let training_data = vec![data_frame.clone(), data_frame.clone(), data_frame.clone()];
        let test_data = vec![data_frame];

        let nr_epochs = 5;
        let data_source = VecDataSource::new(training_data, test_data);
        let callbacks = TestController::new();
        let optimizer = MiniBatchSgd { learning_rate: 0.1 };
        let trainer = ListNetTrainer::new(list_net, data_source, callbacks, optimizer);
        let (controller, _list_net) = trainer.train(nr_epochs, 3).unwrap();
        let evaluation_results = controller.evaluation_results;

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
    fn test_prepare_inputs() {
        let inputs = Array1::from(SAMPLE_INPUTS.to_vec())
            .into_shape((10, 50))
            .unwrap();

        let processed_inputs = prepare_inputs(&inputs).unwrap();

        assert_approx_eq!(f32, &inputs, processed_inputs);

        let few_inputs = Array1::from(SAMPLE_INPUTS_TO_FEW.to_vec())
            .into_shape((3, 50))
            .unwrap();

        assert!(prepare_inputs(&few_inputs).is_none());
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

    #[test]
    fn test_serialize_deserialize_list_net() {
        let list_net = ListNet::new_with_random_weights();
        let mut buffer = Vec::new();
        list_net.clone().serialize_into(&mut buffer).unwrap();
        let list_net2 = ListNet::deserialize_from(&*buffer).unwrap();
        assert_approx_eq!(f32, list_net.dense1.weights(), list_net2.dense1.weights());
        assert_approx_eq!(f32, list_net.dense1.bias(), list_net2.dense1.bias());
        assert_approx_eq!(f32, list_net.dense2.weights(), list_net2.dense2.weights());
        assert_approx_eq!(f32, list_net.dense2.bias(), list_net2.dense2.bias());
        assert_approx_eq!(f32, list_net.scores.weights(), list_net2.scores.weights());
        assert_approx_eq!(f32, list_net.scores.bias(), list_net2.scores.bias());
        assert_approx_eq!(
            f32,
            list_net.prob_dist.weights(),
            list_net2.prob_dist.weights()
        );
        assert_approx_eq!(f32, list_net.prob_dist.bias(), list_net2.prob_dist.bias());
    }
}
