use std::{error::Error as StdError, iter, sync::Mutex};

use ndarray::ArrayView1;
use thiserror::Error;

use crate::ltr::list_net::{
    data::{DataSource, GradientSet, SampleOwned, SampleView},
    model::ListNet,
    optimizer::Optimizer,
};
use layer::utils::kl_divergence;

/// Trainer allowing the training of ListNets.
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
            .begin_of_training(epochs)
            .map_err(TrainingError::Control)?;

        for _ in 0..epochs {
            self.data_source.reset();

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
    fn begin_of_training(&mut self, nr_epochs: usize) -> Result<(), Self::Error>;

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
