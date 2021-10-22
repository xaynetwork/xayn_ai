use std::{
    error::Error as StdError,
    ops::{Add, Div, MulAssign},
};

use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};

use crate::{data::document::Relevance, ltr::list_net::model::ListNet};
use layer::{dense::DenseGradientSet, utils::softmax};

/// Data collected when running the forward pass of back-propagation.
///
/// It extends [`PartialForwardPassData`] by dereferencing to it,
/// so you can treat it as if it also contains all fields from
/// [`PartialForwardPassData`].
///
/// By convention the `y` refers to the outputs of a layer and
/// `z` to the outputs of a layer before that layers activation
/// function has been applied to them.
pub(crate) struct ForwardPassData {
    pub(super) partial_forward_pass: PartialForwardPassData,
    pub(super) scores_y: Array1<f32>,
    pub(super) prob_dist_y: Array1<f32>,
}

/// Some of the data collected when running the forward pass of back-propagation.
pub(crate) struct PartialForwardPassData {
    pub(super) dense1_y: Array2<f32>,
    pub(super) dense1_z: Array2<f32>,
    pub(super) dense2_y: Array2<f32>,
    pub(super) dense2_z: Array2<f32>,
}

/// A set of gradients for all parameters of `ListNet`.
#[cfg_attr(test, derive(Debug, Clone))]
pub struct GradientSet {
    pub(super) dense1: DenseGradientSet,
    pub(super) dense2: DenseGradientSet,
    pub(super) scores: DenseGradientSet,
    pub(super) prob_dist: DenseGradientSet,
}

impl GradientSet {
    /// Merges all gradient sets computed for one batch of training data.
    ///
    /// This will create the mean of each gradient across all gradient sets.
    ///
    /// If there are no gradient sets in the input then `None` is returned.
    pub(super) fn mean_of(gradient_sets: Vec<GradientSet>) -> Option<GradientSet> {
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

/// An owned version of [`SampleView`]
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
    fn reset(&mut self);

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
