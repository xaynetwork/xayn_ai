//! ListNet implementation using the NdArray crate.

use std::{
    error::Error as StdError,
    io::{Read, Write},
    iter,
    ops::{Add, Div, MulAssign},
    path::Path,
    sync::Mutex,
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

    /// Turns this `ListNet` instance into a `BinParams` instance.
    fn into_bin_params(self) -> BinParams {
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
        self.into_bin_params().serialize_into(writer)
    }

    /// Serializes the ListNet into given file.
    pub fn serialize_into_file(
        self,
        path: impl AsRef<Path>,
    ) -> Result<(), Box<bincode::ErrorKind>> {
        self.into_bin_params().serialize_into_file(path)
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
    /// but only supports a size of exactly [`Self::INPUT_NR_DOCUMENTS`].
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

    /// Evaluates the ListNet on a given sample using the given cost function.
    pub fn evaluate(
        &self,
        cost_function: fn(ArrayView1<f32>, ArrayView1<f32>) -> f32,
        sample: SampleView,
    ) -> f32 {
        let SampleView {
            inputs,
            target_prob_dist,
        } = sample;
        let (scores_y, _) = self.calculate_intermediate_scores(inputs, false);
        let prob_dist_y = self.calculate_final_scores(&scores_y);
        cost_function(target_prob_dist, prob_dist_y.view())
    }

    /// Computes the gradients and loss for given inputs and target prob. dist.
    ///
    /// # Panics
    ///
    /// If inputs and relevances are not for exactly [`Self::INPUT_NR_DOCUMENTS`]
    /// documents.
    fn gradients_for_query(&self, sample: SampleView) -> (GradientSet, f32) {
        let SampleView {
            inputs,
            target_prob_dist,
        } = sample;
        assert_eq!(inputs.shape()[0], ListNet::INPUT_NR_DOCUMENTS);
        assert_eq!(target_prob_dist.len(), ListNet::INPUT_NR_DOCUMENTS);
        let results = self.forward_pass(inputs);
        //FIXME[followup PR] if we don't track loss in XayNet when used in the app we don't need to calculate it.
        let loss = kl_divergence(target_prob_dist.view(), results.prob_dist_y.view());
        //UNWRAP_SAFE: Document are not empty.
        let gradients = self
            .back_propagation(inputs, target_prob_dist, results)
            .unwrap();
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
    ) -> Option<GradientSet> {
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
        if nr_documents == 0 {
            return None;
        }

        let derivatives_of_clipping =
            prob_dist_y.mapv(|v| (f32::EPSILON..=1.).contains(&v) as u8 as f32);
        let p_cost_and_prob_dist = (prob_dist_y - target_prob_dist) * derivatives_of_clipping;

        let d_prob_dist = self
            .prob_dist
            .gradients_from_partials_1d(scores_y.view(), p_cost_and_prob_dist.view());

        // The activation functions of `scores` is the identity function (linear) so
        // it's gradient is 1 at all inputs and we can omit it.
        let p_scores = self.prob_dist.weights().dot(&p_cost_and_prob_dist);

        let mut d_scores = Vec::with_capacity(nr_documents);
        let mut d_dense2 = Vec::with_capacity(nr_documents);
        let mut d_dense1 = Vec::with_capacity(nr_documents);

        for row in 0..nr_documents {
            // From here on training is "split" into multiple parallel "path" using
            // shared weights (hence why we add up the gradients).
            let p_scores = p_scores.slice(s![row..row + 1]);

            let d_scores_part = self
                .scores
                .gradients_from_partials_1d(dense2_y.slice(s![row, ..]), p_scores);
            d_scores.push(d_scores_part);

            let p_dense2 = self.scores.weights().dot(&p_scores)
                * Relu::partial_derivatives_at(&dense2_z.slice(s![row, ..]));
            let d_dense2_part = self
                .dense2
                .gradients_from_partials_1d(dense1_y.slice(s![row, ..]), p_dense2.view());
            d_dense2.push(d_dense2_part);

            let p_dense1 = self.dense2.weights().dot(&p_dense2)
                * Relu::partial_derivatives_at(&dense1_z.slice(s![row, ..]));
            let d_dense1_part = self
                .dense1
                .gradients_from_partials_1d(inputs.slice(s![row, ..]), p_dense1.view());
            d_dense1.push(d_dense1_part);
        }

        let d_scores = DenseGradientSet::merge_shared(d_scores).unwrap();
        let d_dense2 = DenseGradientSet::merge_shared(d_dense2).unwrap();
        let d_dense1 = DenseGradientSet::merge_shared(d_dense1).unwrap();

        Some(GradientSet {
            dense1: d_dense1,
            dense2: d_dense2,
            scores: d_scores,
            prob_dist: d_prob_dist,
        })
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

    /// Trains for the given number of epochs.
    ///
    /// This will use the `list_net`, `data_source`, `callbacks`, `optimizer` and `batch_size` used
    /// to create this `ListNetTrainer`.
    pub fn train(mut self, epochs: usize) -> Result<C::Outcome, TrainingError<D::Error, C::Error>> {
        self.callbacks
            .begin_of_training(epochs, &self.list_net)
            .map_err(TrainingError::Control)?;

        for _ in 0..epochs {
            self.data_source.reset().map_err(TrainingError::Data)?;

            self.callbacks
                .begin_of_epoch(self.data_source.number_of_training_batches())
                .map_err(TrainingError::Control)?;

            while self.train_next_batch()? {}

            self.evaluate_epoch(kl_divergence)?;

            self.callbacks
                .end_of_epoch(&self.list_net)
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
    /// If there are no more batches to train on this returns `false`, else this
    /// returns `true`.
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

        let gradient_sets = callbacks
            .run_batch(batch, |sample| list_net.gradients_for_query(sample))
            .map_err(TrainingError::Control)?;

        optimizer.apply_gradients(list_net, gradient_sets);
        Ok(true)
    }

    /// Returns the mean cost over all samples of the evaluation dataset using the given cost function.
    fn evaluate_epoch(
        &mut self,
        cost_function: fn(ArrayView1<f32>, ArrayView1<f32>) -> f32,
    ) -> Result<(), TrainingError<D::Error, C::Error>> {
        let Self {
            data_source,
            callbacks,
            list_net,
            ..
        } = self;

        let nr_samples = data_source.number_of_evaluation_samples();
        let error_slot = Mutex::new(None);
        let sample_iter = iter::from_fn(|| match data_source.next_evaluation_sample() {
            Ok(v) => v,
            Err(err) => {
                let mut error_slot = error_slot.lock().unwrap();
                *error_slot = Some(err);
                None
            }
        });

        callbacks
            .run_evaluation(sample_iter, nr_samples, |sample| {
                list_net.evaluate(cost_function, sample.as_view())
            })
            .map_err(TrainingError::Control)?;

        if let Some(error) = error_slot.into_inner().unwrap() {
            Err(TrainingError::Data(error))
        } else {
            Ok(())
        }
    }
}

/// A single sample you can train on.
pub struct SampleView<'a> {
    /// Inputs used for training.
    ///
    /// (At least for now) this must be a `(10, 50)` array
    /// view.
    pub inputs: ArrayView2<'a, f32>,

    /// Target probability distribution.
    ///
    /// Needs to have the same length as `inputs.shape()[0]`.
    pub target_prob_dist: ArrayView1<'a, f32>,
}

impl<'a> SampleView<'a> {
    pub fn to_owned(&self) -> SampleOwned {
        SampleOwned {
            inputs: self.inputs.to_owned(),
            target_prob_dist: self.target_prob_dist.to_owned(),
        }
    }
}

/// An owned version of [`SampleRef`]
pub struct SampleOwned {
    /// Inputs used for training.
    ///
    /// (At least for now) this must be a `(10, 50)` array
    /// view.
    pub inputs: Array2<f32>,

    /// Target probability distribution.
    ///
    /// Needs to have the same length as `inputs.shape()[0]`.
    pub target_prob_dist: Array1<f32>,
}

impl SampleOwned {
    pub fn as_view(&self) -> SampleView {
        SampleView {
            inputs: self.inputs.view(),
            target_prob_dist: self.target_prob_dist.view(),
        }
    }
}

/// A source of training and evaluation data.
///
/// Settings like the batch size or evaluation split need
/// to be handled when creating this instance.
pub trait DataSource: Send {
    type Error: StdError + 'static + Send;

    /// Resets/initializes the "iteration" of training and evaluation samples.
    ///
    /// This is allowed to also *change* the training and evaluation
    /// samples and/or their order. E.g. this could shuffle them
    /// before every epoch.
    ///
    /// This is called at the *beginning* of every epoch.
    fn reset(&mut self) -> Result<(), Self::Error>;

    /// Returns the expected number of training batches.
    fn number_of_training_batches(&self) -> usize;

    /// Returns the next batch of training samples.
    ///
    /// Returns an empty vector once all training samples have been returned.
    fn next_training_batch(&mut self) -> Result<Vec<SampleView>, Self::Error>;

    /// Returns the expected number of evaluation samples.
    fn number_of_evaluation_samples(&self) -> usize;

    /// Returns the next evaluation sample.
    ///
    /// Returns `None` once all training samples have been returned.
    fn next_evaluation_sample(&mut self) -> Result<Option<SampleOwned>, Self::Error>;
}

/// A trait providing various callbacks used during training.
pub trait TrainingController {
    type Error: StdError + 'static;
    type Outcome;

    /// Runs a batch of training steps.
    ///
    /// Implementations can be both sequential or parallel.
    fn run_batch(
        &mut self,
        batch: Vec<SampleView>,
        map_fn: impl Fn(SampleView) -> (GradientSet, f32) + Send + Sync,
    ) -> Result<Vec<GradientSet>, Self::Error>;

    /// Runs the processing of the sample evaluation.
    fn run_evaluation<I>(
        &mut self,
        samples: I,
        nr_samples: usize,
        eval_fn: impl Fn(SampleOwned) -> f32 + Send + Sync,
    ) -> Result<(), Self::Error>
    where
        I: IntoIterator<Item = SampleOwned>,
        I::IntoIter: Send;

    /// Called at the beginning of each epoch.
    fn begin_of_epoch(&mut self, nr_batches: usize) -> Result<(), Self::Error>;

    /// Called at the end of each epoch.
    ///
    /// The passed in reference to `list_net` can be used to e.g. bump the intermediate training
    /// result every 10 epochs.
    fn end_of_epoch(&mut self, list_net: &ListNet) -> Result<(), Self::Error>;

    /// Called at the beginning of training.
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

/// An error which can occur during training.
///
/// This is either an error from the `DataSource` or
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
mod tests;
