#![cfg(not(tarpaulin))]

use std::{
    ops::Add,
    path::Path,
    sync::{Arc, Mutex},
    time::Instant,
};

use bincode::Error;
use indicatif::{FormattedDuration, MultiProgress, ProgressBar, ProgressStyle};
use log::{debug, info, trace};
use rayon::prelude::*;
use xayn_ai::list_net::{GradientSet, ListNet, SampleOwned, SampleView, TrainingController};

/// Builder to create a [`CliTrainingController`].
pub(crate) struct CliTrainingControllerBuilder<F>
where
    F: TrainingFeedback,
{
    /// The output used to store files (e.g. ListNet parameter).
    pub(crate) file_output: OutputDir,

    /// Implementation used to hint the training progress to the user.
    pub(crate) training_feedback: F,

    /// If included dumps the current ListNet parameters every `n` epochs.
    ///
    /// The ListNet will be written to the output folder in the form of
    /// `list_net_{epoch}.binparams`.
    pub(crate) dump_every: Option<usize>,
}

impl<F> CliTrainingControllerBuilder<F>
where
    F: TrainingFeedback,
{
    /// Builds the `CliTrainingController`.
    pub(crate) fn build(self) -> CliTrainingController<F> {
        CliTrainingController {
            setup: self,
            current_epoch: 0,
            current_batch: 0,
            start_time: None,
        }
    }
}

/// Training controller for usage in a CLI setup.
pub(crate) struct CliTrainingController<F>
where
    F: TrainingFeedback,
{
    /// We just wrap the builder, as we would just copy the fields.
    setup: CliTrainingControllerBuilder<F>,
    /// The current active (or just ended) epoch.
    current_epoch: usize,
    /// The current active (or just ended) batch.
    current_batch: usize,
    /// The time at which the training started.
    start_time: Option<Instant>,
}

impl<F> TrainingController for CliTrainingController<F>
where
    F: TrainingFeedback,
{
    type Error = Error;
    type Outcome = ListNet;

    fn run_batch(
        &mut self,
        batch: Vec<SampleView>,
        map_fn: impl Fn(SampleView) -> (GradientSet, f32) + Send + Sync,
    ) -> Result<Vec<GradientSet>, Self::Error> {
        trace!("Start of batch #{}", self.current_batch);
        self.setup
            .training_feedback
            .reset_sample_progress(batch.len() as u64);

        let losses = Mutex::new(Vec::new());
        let gradient_sets = batch
            .into_par_iter()
            .map(|sample| {
                let (gradient_set, loss) = map_fn(sample);
                let mut losses = losses.lock().unwrap();
                losses.push(loss);
                self.setup.training_feedback.hint_sample_progress(1);
                gradient_set
            })
            .collect();

        let losses = losses.into_inner().unwrap();
        let mean_loss = mean_loss(&losses);

        self.setup
            .training_feedback
            .hint_epoch_progress(1, mean_loss);
        trace!("End of batch #{}", self.current_batch);
        self.current_batch += 1;

        Ok(gradient_sets)
    }

    fn run_evaluation<I>(
        &mut self,
        samples: I,
        nr_samples: usize,
        eval_fn: impl Fn(SampleOwned) -> f32 + Send + Sync,
    ) -> Result<(), Self::Error>
    where
        I: IntoIterator<Item = SampleOwned>,
        I::IntoIter: Send,
    {
        trace!("Beginning of evaluation");
        if nr_samples == 0 {
            trace!("Skipping evaluation.");
            return Ok(());
        }

        self.setup
            .training_feedback
            .reset_eval_progress(Some(nr_samples as u64));

        let mean_cost = samples
            .into_iter()
            .par_bridge()
            .map(|sample| {
                let cost = eval_fn(sample);
                self.setup.training_feedback.hint_eval_progress(1, None);
                cost / nr_samples as f32
            })
            .reduce_with(Add::add)
            .unwrap();

        //FIXME This can always happen (after a longer training, mainly
        //      if training with inadequate data or parameters). Still
        //      we want to handle this better in the future.
        if mean_cost.is_nan() {
            panic!("evaluation KL-Divergence cost is NaN");
        }

        self.setup
            .training_feedback
            .hint_eval_progress(0, Some(mean_cost));
        self.setup
            .training_feedback
            .println(&format!("Evaluation Cost: {}", mean_cost));
        trace!("End of evaluation, cost={}", mean_cost);
        Ok(())
    }

    fn begin_of_epoch(&mut self, nr_batches: usize) -> Result<(), Self::Error> {
        debug!(
            "Beginning of epoch #{:0>4} (#batch {})",
            self.current_epoch, nr_batches
        );
        self.current_batch = 0;
        self.setup
            .training_feedback
            .reset_epoch_progress(nr_batches as u64);
        self.setup.training_feedback.reset_eval_progress(None);
        Ok(())
    }

    fn end_of_epoch(&mut self, list_net: &ListNet) -> Result<(), Self::Error> {
        debug!("End of epoch #{}", self.current_epoch);

        if let Some(dump_every) = self.setup.dump_every {
            if (self.current_epoch + 1) % dump_every == 0 {
                trace!("Dumping parameters.");
                self.setup.file_output.save_list_net_parameters(
                    list_net.clone(),
                    &format!("{}", self.current_epoch),
                )?;
            }
        }

        self.setup.training_feedback.hint_train_progress(1);
        self.current_epoch += 1;
        Ok(())
    }

    fn begin_of_training(&mut self, nr_epochs: usize) -> Result<(), Self::Error> {
        self.start_time = Some(Instant::now());
        info!("Beginning of training for {}", nr_epochs);

        self.setup.training_feedback.start_feedback();
        self.setup
            .training_feedback
            .reset_train_progress(nr_epochs as u64);
        Ok(())
    }

    fn end_of_training(&mut self) -> Result<(), Self::Error> {
        let elapsed = self.start_time.map(|t| t.elapsed()).unwrap_or_default();
        info!("End of training. Duration: {}", FormattedDuration(elapsed));
        Ok(())
    }

    fn training_result(mut self, list_net: ListNet) -> Result<Self::Outcome, Self::Error> {
        self.setup.training_feedback.end_feedback();
        Ok(list_net)
    }
}

pub(crate) trait TrainingFeedback: Sync + Send {
    fn start_feedback(&mut self);
    fn end_feedback(&mut self);

    fn reset_sample_progress(&mut self, max_process: u64);
    fn reset_train_progress(&mut self, max_process: u64);
    fn reset_epoch_progress(&mut self, max_process: u64);

    /// Resets the evaluation progress.
    ///
    /// Passing in `None` will just set the current process to `0`.
    ///
    /// Passing in `Some(X)` will do the same as passing in `None`
    /// but will also set the `max` process.
    fn reset_eval_progress(&mut self, max_process: Option<u64>);

    fn hint_sample_progress(&self, inc_process: u64);
    fn hint_train_progress(&self, inc_process: u64);
    fn hint_epoch_progress(&self, inc_process: u64, loss: f32);
    fn hint_eval_progress(&self, inc_process: u64, mean_eval_cost: Option<f32>);

    fn println(&self, line: &str);
}

/// Gives no feedback.
///
/// When emulating XayNet we run multiple trainings in parallel and as such we
/// can't give a progress bar like feedback on the CLI.
pub(crate) struct NoFeedback;

impl TrainingFeedback for NoFeedback {
    fn start_feedback(&mut self) {}

    fn end_feedback(&mut self) {}

    fn reset_sample_progress(&mut self, _max_process: u64) {}

    fn reset_train_progress(&mut self, _max_process: u64) {}

    fn reset_epoch_progress(&mut self, _max_process: u64) {}

    fn reset_eval_progress(&mut self, _max_process: Option<u64>) {}

    fn hint_sample_progress(&self, _inc_process: u64) {}

    fn hint_train_progress(&self, _inc_process: u64) {}

    fn hint_epoch_progress(&self, _inc_process: u64, _loss: f32) {}

    fn hint_eval_progress(&self, _inc_process: u64, _mean_eval_cost: Option<f32>) {}

    fn println(&self, _line: &str) {}
}

/// Training feedback
pub(crate) struct ProgressBarTrainingFeedback {
    /// A (CLI) progress bar used to track the progress of the current batch.
    sample_progress_bar: ProgressBar,
    /// A (CLI) progress bar used to track the progress of the current training.
    train_progress_bar: ProgressBar,
    /// A (CLI) progress bar used to track the progress of the current epoch.
    epoch_progress_bar: ProgressBar,
    /// A (CLI) progress bar used to track the progress of the current evaluation.
    eval_progress_bar: ProgressBar,
}

impl ProgressBarTrainingFeedback {
    pub(crate) fn new() -> Self {
        let train_progress_bar = ProgressBar::new(0);
        train_progress_bar.set_style(
            ProgressStyle::default_bar()
                .template(
                    "Epochs:  [{bar:30.green}] {percent:>3}% ({pos:>5}/{len:>5}) {elapsed_precise}",
                )
                .progress_chars("=> "),
        );
        train_progress_bar.set_draw_delta(5);

        let epoch_progress_bar = ProgressBar::new(0);
        epoch_progress_bar.set_style(
            ProgressStyle::default_bar()
                .template("Batches: [{bar:30.green}] {percent:>3}% ({pos:>5}/{len:>5}) {msg}")
                .progress_chars("=> "),
        );
        epoch_progress_bar.set_draw_delta(5);

        let sample_progress_bar = ProgressBar::new(0);
        sample_progress_bar.set_style(
            ProgressStyle::default_bar()
                .template("Samples: [{bar:30.green}] {percent:>3}% ({pos:>5}/{len:>5})")
                .progress_chars("=> "),
        );
        sample_progress_bar.set_draw_delta(5);

        let eval_progress_bar = ProgressBar::new(0);
        eval_progress_bar.set_style(
            ProgressStyle::default_bar()
                .template("Evaluation: [{bar:27.green}] {percent:>3}% ({pos:>5}/{len:>5}) {msg}")
                .progress_chars("=> "),
        );
        eval_progress_bar.set_draw_delta(5);

        Self {
            sample_progress_bar,
            train_progress_bar,
            epoch_progress_bar,
            eval_progress_bar,
        }
    }
}

impl TrainingFeedback for ProgressBarTrainingFeedback {
    fn start_feedback(&mut self) {
        let multi_bar = MultiProgress::new();
        multi_bar.add(self.train_progress_bar.clone());
        multi_bar.add(self.epoch_progress_bar.clone());
        multi_bar.add(self.sample_progress_bar.clone());
        multi_bar.add(self.eval_progress_bar.clone());
        // Needed or else bars won't print to the screen.
        std::thread::spawn(move || multi_bar.join());

        self.train_progress_bar.enable_steady_tick(100);
        self.epoch_progress_bar.tick();
        self.eval_progress_bar.set_message("--");
        self.eval_progress_bar.set_length(1);
    }

    fn end_feedback(&mut self) {
        self.train_progress_bar.finish_at_current_pos();
        self.epoch_progress_bar.finish_at_current_pos();
        self.sample_progress_bar.finish_at_current_pos();
        self.eval_progress_bar.finish_at_current_pos();
    }

    fn reset_sample_progress(&mut self, max_process: u64) {
        self.sample_progress_bar.set_position(0);
        self.sample_progress_bar.set_length(max_process);
    }

    fn reset_train_progress(&mut self, max_process: u64) {
        self.train_progress_bar.set_position(0);
        self.train_progress_bar.set_length(max_process);
    }

    fn reset_epoch_progress(&mut self, max_process: u64) {
        self.epoch_progress_bar.set_position(0);
        self.epoch_progress_bar.set_length(max_process);
    }

    fn reset_eval_progress(&mut self, max_process: Option<u64>) {
        self.eval_progress_bar.set_position(0);
        if let Some(len) = max_process {
            self.eval_progress_bar.set_length(len);
        }
    }

    fn hint_sample_progress(&self, inc_process: u64) {
        self.sample_progress_bar.inc(inc_process);
    }

    fn hint_train_progress(&self, inc_process: u64) {
        self.train_progress_bar.inc(inc_process);
    }

    fn hint_epoch_progress(&self, inc_process: u64, loss: f32) {
        self.epoch_progress_bar.inc(inc_process);
        self.epoch_progress_bar
            .set_message(format!("loss={:.5}", loss));
    }

    fn hint_eval_progress(&self, inc_process: u64, mean_eval_cost: Option<f32>) {
        self.eval_progress_bar.inc(inc_process);
        if let Some(cost) = mean_eval_cost {
            self.eval_progress_bar
                .set_message(format!("cost={:.5}", cost));
        }
    }

    fn println(&self, line: &str) {
        self.train_progress_bar.println(line);
    }
}

/// Provides functionality for storing ListNets.
///
/// To be more specific it stores the `ListNet` parameters
/// not the ListNets structure.
#[derive(Clone)]
pub(crate) struct OutputDir {
    out_dir: Arc<Path>,
}

impl OutputDir {
    pub const SUFFIX_FINAL_LIST_NET: &'static str = "final";
    pub const SUFFIX_INITIAL_LIST_NET: &'static str = "initial";

    /// Creates a new instance using given output dir.
    pub(crate) fn new(out_dir: impl AsRef<Path>) -> Self {
        Self {
            out_dir: out_dir.as_ref().into(),
        }
    }

    /// Saves the current ListNet parameters to the output dir using the given suffix.
    ///
    /// The file path will be:
    ///
    /// `<out_dir>/list_net_<suffix>.binparams`
    pub(crate) fn save_list_net_parameters(
        &self,
        list_net: ListNet,
        suffix: &str,
    ) -> Result<(), Error> {
        let file_path = self
            .out_dir
            .join(format!("list_net_{:0>4}.binparams", suffix));
        list_net.serialize_into_file(file_path).map_err(Into::into)
    }
}

/// Calculates the mean loss.
fn mean_loss(losses: &[f32]) -> f32 {
    let count = losses.len() as f32;
    losses.iter().fold(0f32, |acc, v| acc + v / count)
}

#[cfg(test)]
mod tests {
    use xayn_ai::assert_approx_eq;

    use super::*;

    #[test]
    fn test_mean_loss() {
        assert_approx_eq!(f32, mean_loss(&[0.5, 0.75, 0.25, 0.5]), 0.5);
    }
}
