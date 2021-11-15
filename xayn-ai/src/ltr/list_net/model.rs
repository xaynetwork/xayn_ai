use std::{
    io::{Read, Write},
    path::Path,
};

use displaydoc::Display;
use ndarray::{
    s,
    Array1,
    ArrayBase,
    ArrayView1,
    Axis,
    Data,
    Dimension,
    IntoDimension,
    Ix,
    Ix1,
    Ix2,
};
use thiserror::Error;

use crate::ltr::list_net::data::{
    ForwardPassData,
    GradientSet,
    PartialForwardPassData,
    SampleView,
};
use layer::{
    activation::{Linear, Relu, Softmax},
    dense::{Dense, DenseGradientSet},
    io::{BinParams, LoadingBinParamsFailed, LoadingLayerFailed},
    utils::{he_normal_weights_init, kl_divergence, IncompatibleMatrices},
};

/// ListNet load failure.
#[derive(Debug, Display, Error)]
pub enum LoadingListNetFailed {
    /// Failed to load bin params.
    #[displaydoc("{0}")]
    BinParams(#[from] LoadingBinParamsFailed),

    /// Failed to create instance of `Dense`.
    #[displaydoc("{0}")]
    Dense(#[from] LoadingLayerFailed),

    /// Tried to load a ListNet containing incompatible matrices.
    #[displaydoc("{0}")]
    IncompatibleMatrices(#[from] IncompatibleMatrices),

    /// BinParams contains additional parameters, model data is probably wrong: {params:?}
    LeftoverBinParams { params: Vec<String> },
}

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
    pub(super) dense1: Dense<Relu>,
    pub(super) dense2: Dense<Relu>,
    pub(super) scores: Dense<Linear>,
    pub(super) prob_dist: Dense<Softmax>,
}

impl ListNet {
    /// Number of documents directly reranked
    pub(super) const INPUT_NR_DOCUMENTS: Ix = 10;

    /// The size by with input is chunked.
    ///
    /// The first chunk is then used together with each other chunk.
    pub(super) const CHUNK_SIZE: Ix = Self::INPUT_NR_DOCUMENTS / 2;

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
            Err(IncompatibleMatrices::new(
                "scores_prob_dist/output",
                prob_dist_out_shape,
                "list_net/output",
                (10,),
                "expected scores_prob_dist output shape to be equal to (10,)",
            )
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
    pub(super) fn calculate_intermediate_scores(
        &self,
        inputs: ArrayBase<impl Data<Elem = f32>, Ix2>,
        for_back_propagation: bool,
    ) -> (Array1<f32>, Option<PartialForwardPassData>) {
        debug_assert_eq!(inputs.shape()[1], Self::INPUT_NR_FEATURES);
        let (dense1_y, dense1_z) = self.dense1.run(inputs, for_back_propagation);
        let (dense2_y, dense2_z) = self.dense2.run(dense1_y.view(), for_back_propagation);
        let (scores, _) = self.scores.run(dense2_y.view(), false);
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
    /// The input must be based on the output of [`ListNet::calculate_intermediate_scores()`],
    /// but only supports a size of exactly [`Self::INPUT_NR_DOCUMENTS`].
    fn calculate_final_scores(&self, scores: ArrayBase<impl Data<Elem = f32>, Ix1>) -> Array1<f32> {
        debug_assert_eq!(scores.shape()[0], Self::INPUT_NR_DOCUMENTS);
        self.prob_dist.run(scores, false).0
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
    pub(super) fn calculate_final_scores_padded(
        &self,
        first: &[f32],
        second: Option<&[f32]>,
    ) -> Vec<f32> {
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

        let outputs = self.calculate_final_scores(Array1::from(inputs));
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
    pub(crate) fn run(&self, inputs: ArrayBase<impl Data<Elem = f32>, Ix2>) -> Vec<f32> {
        let nr_documents = inputs.shape()[0];

        assert_eq!(
            inputs.shape()[1],
            Self::INPUT_NR_FEATURES,
            "ListNet expects exactly {} features per document got {}",
            Self::INPUT_NR_FEATURES,
            inputs.shape()[1],
        );

        if nr_documents == 0 {
            return Vec::new();
        }

        let intermediate_scores = {
            let (scores, _) = self.calculate_intermediate_scores(inputs, false);
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
        cost_function: impl Fn(ArrayView1<f32>, ArrayView1<f32>) -> f32,
        sample: SampleView,
    ) -> f32 {
        let SampleView {
            inputs,
            target_prob_dist,
        } = sample;
        let (scores_y, _) = self.calculate_intermediate_scores(inputs, false);
        let prob_dist_y = self.calculate_final_scores(scores_y);
        cost_function(target_prob_dist, prob_dist_y.view())
    }

    /// Computes the gradients and loss for given inputs and target prob. dist.
    ///
    /// # Panics
    ///
    /// If inputs and relevances are not for exactly [`Self::INPUT_NR_DOCUMENTS`]
    /// documents.
    pub(super) fn gradients_for_query(&self, sample: SampleView) -> (GradientSet, f32) {
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
    fn forward_pass(&self, inputs: ArrayBase<impl Data<Elem = f32>, Ix2>) -> ForwardPassData {
        let (scores_y, partial_forward_pass) = self.calculate_intermediate_scores(inputs, true);
        let prob_dist_y = self.calculate_final_scores(scores_y.view());

        ForwardPassData {
            partial_forward_pass: partial_forward_pass.unwrap(),
            scores_y,
            prob_dist_y,
        }
    }

    /// Run back propagation based on given inputs, target prob. dist. and forward pass data.
    fn back_propagation(
        &self,
        inputs: ArrayBase<impl Data<Elem = f32>, Ix2>,
        target_prob_dist: ArrayBase<impl Data<Elem = f32>, Ix1>,
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
                * Relu::partial_derivatives_at(dense2_z.slice(s![row, ..]));
            let d_dense2_part = self
                .dense2
                .gradients_from_partials_1d(dense1_y.slice(s![row, ..]), p_dense2.view());
            d_dense2.push(d_dense2_part);

            let p_dense1 = self.dense2.weights().dot(&p_dense2)
                * Relu::partial_derivatives_at(dense1_z.slice(s![row, ..]));
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

    /// Merges multiple `ListNet`'s into one.
    pub fn merge_nets(list_nets: Vec<ListNet>) -> Option<ListNet> {
        let count = list_nets.len();

        // FIXME we might really want to improve numeric stability.
        //    Also if we don't add batch normalization as part of the
        //    training we probably still want to normalize here.
        list_nets
            .into_iter()
            .map(|net| net.div_parameters_by(count as f32))
            .reduce(|l, r| l.add_parameters_of(r))
    }

    fn div_parameters_by(mut self, denominator: f32) -> Self {
        self.dense1.div_parameters_by(denominator);
        self.dense2.div_parameters_by(denominator);
        self.scores.div_parameters_by(denominator);
        self.prob_dist.div_parameters_by(denominator);
        self
    }

    fn add_parameters_of(mut self, other: Self) -> Self {
        let Self {
            dense1,
            dense2,
            scores,
            prob_dist,
        } = other;
        self.dense1.add_parameters_of(dense1);
        self.dense2.add_parameters_of(dense2);
        self.scores.add_parameters_of(scores);
        self.prob_dist.add_parameters_of(prob_dist);
        self
    }
}
