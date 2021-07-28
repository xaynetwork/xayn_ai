use std::{convert::TryInto, path::PathBuf, sync::Mutex, time::Instant};

use bincode::Error;
use indicatif::{FormattedDuration, MultiProgress, ProgressBar, ProgressStyle};
use log::{debug, info, trace};
use rayon::prelude::*;
use xayn_ai::list_net::{GradientSet, ListNet, Sample, TrainingController};

/// Builder to create a [`CliTrainingController`].
pub(crate) struct CliTrainingControllerBuilder {
    /// The output dir into which files are written.
    pub(crate) out_dir: PathBuf,

    /// If true dumps the initial parameters.
    ///
    /// This is can be useful if the initial parameters
    /// have been freshly created using some random
    /// initializer.
    pub(crate) dump_initial_parameters: bool,

    /// If included dumps the current ListNet parameters every `n` epochs.
    ///
    /// The ListNet will be written to the output folder in the form of
    /// `list_net_{epoch}.binparams`.
    pub(crate) dump_every: Option<usize>,
}

impl CliTrainingControllerBuilder {
    /// Builds the `CliTrainingController`.
    pub(crate) fn build(self) -> CliTrainingController {
        let train_progress_bar = ProgressBar::new(0);
        train_progress_bar.set_style(
            ProgressStyle::default_bar()
                .template(
                    "Epochs:  [{bar:30.green}] {percent:>3}% ({pos:>5}/{len:>5}) {elapsed_precise}",
                )
                .progress_chars("=> "),
        );
        let epoch_progress_bar = ProgressBar::new(0);
        epoch_progress_bar.set_style(
            ProgressStyle::default_bar()
                .template("Batches: [{bar:30.green}] {percent:>3}% ({pos:>5}/{len:>5}) {msg}")
                .progress_chars("=> "),
        );
        let eval_progress_bar = ProgressBar::new(0);
        eval_progress_bar.set_style(
            ProgressStyle::default_bar()
                .template("Evaluation: [{bar:27.green}] {percent:>3}% ({pos:>5}/{len:>5}) {msg}")
                .progress_chars("=> "),
        );

        CliTrainingController {
            setting: self,
            current_epoch: 0,
            current_batch: 0,
            start_time: None,
            epoch_progress_bar,
            train_progress_bar,
            eval_progress_bar,
            current_mean_evaluation_cost_and_len: None,
        }
    }
}

/// Training controller for usage in a CLI setup.
pub(crate) struct CliTrainingController {
    /// The settings to use when controlling the training.
    setting: CliTrainingControllerBuilder,
    /// The current active (or just ended) epoch.
    current_epoch: usize,
    /// The current active (or just ended) batch.
    current_batch: usize,
    /// The time at which the training did start.
    start_time: Option<Instant>,
    /// A (CLI) progress bar used to track the progress of the current training.
    train_progress_bar: ProgressBar,
    /// A (CLI) progress bar used to track the progress of the current epoch.
    epoch_progress_bar: ProgressBar,
    /// A (CLI) progress bar used to track the progress of the current evaluation.
    eval_progress_bar: ProgressBar,
    /// The current mean of an in-progress evaluation as well as the nr of samples.
    current_mean_evaluation_cost_and_len: Option<(f32, usize)>,
}

impl CliTrainingController {
    /// Safe the current ListNet parameters to the output dir using the given suffix.
    ///
    /// The file path will be:
    ///
    /// `<out_dir>/list_net_<suffix>.binparams`
    fn save_parameters(&self, list_net: ListNet, suffix: &str) -> Result<(), Error> {
        let file_path = self
            .setting
            .out_dir
            .join(format!("list_net_{:0>4}.binparams", suffix));
        list_net.serialize_into_file(file_path).map_err(Into::into)
    }
}

impl TrainingController for CliTrainingController {
    type Error = Error;
    type Outcome = ();

    fn run_batch(
        &mut self,
        batch: Vec<Sample>,
        map_fn: impl Fn(Sample) -> (GradientSet, f32) + Send + Sync,
    ) -> Vec<GradientSet> {
        let losses = Mutex::new(Vec::new());
        let gradient_sets = batch
            .into_iter()
            .par_bridge()
            .map(|sample| {
                let (gradient_set, loss) = map_fn(sample);
                let mut losses = losses.lock().unwrap();
                losses.push(loss);
                gradient_set
            })
            .collect();

        let losses = losses.into_inner().unwrap();
        let mean_loss = mean_loss(&losses);
        self.epoch_progress_bar
            .set_message(format!("loss={:.5}", mean_loss));
        gradient_sets
    }

    fn begin_of_batch(&mut self) -> Result<(), Self::Error> {
        trace!("Start of batch #{}", self.current_batch);
        Ok(())
    }

    fn end_of_batch(&mut self) -> Result<(), Self::Error> {
        trace!("End of batch #{}", self.current_batch);
        self.epoch_progress_bar.inc(1);
        self.current_batch += 1;
        Ok(())
    }

    fn begin_of_epoch(&mut self, nr_batches: usize) -> Result<(), Self::Error> {
        debug!(
            "Begin of epoch #{:0>4} (#batch {})",
            self.current_epoch, nr_batches
        );
        self.current_batch = 0;
        self.epoch_progress_bar.set_position(0);
        self.epoch_progress_bar
            .set_length(nr_batches.try_into().unwrap());
        self.eval_progress_bar.set_position(0);
        Ok(())
    }

    fn end_of_epoch(&mut self, list_net: &ListNet) -> Result<(), Self::Error> {
        debug!("End of epoch #{}", self.current_epoch);

        if let Some(dump_every) = self.setting.dump_every {
            if (self.current_epoch + 1) % dump_every == 0 {
                trace!("Dumping parameters.");
                self.save_parameters(list_net.clone(), &format!("{}", self.current_epoch))?;
            }
        }

        self.train_progress_bar.inc(1);
        self.current_epoch += 1;
        Ok(())
    }

    fn begin_of_evaluation(&mut self, nr_samples: usize) -> Result<(), Self::Error> {
        self.current_mean_evaluation_cost_and_len = Some((0.0, nr_samples));
        self.eval_progress_bar.set_message("");
        self.eval_progress_bar.set_position(0);
        self.eval_progress_bar
            .set_length(nr_samples.try_into().unwrap());
        Ok(())
    }

    fn evaluation_result(&mut self, cost: f32) -> Result<(), Self::Error> {
        //FIXME This can always happen (after a longer training, mainly
        //      if training with inadequate data or parameters). Still
        //      we want to handle this better in the future.
        if cost.is_nan() {
            panic!("evaluation KL-Divergence cost is NaN");
        }

        let (mean, len) = self.current_mean_evaluation_cost_and_len.as_mut().unwrap();
        *mean += cost / *len as f32;

        self.eval_progress_bar.inc(1);
        Ok(())
    }

    fn end_of_evaluation(&mut self) -> Result<(), Self::Error> {
        let (cost, _) = self.current_mean_evaluation_cost_and_len.take().unwrap();
        self.eval_progress_bar
            .set_message(format!("cost={:.5}", cost));
        self.train_progress_bar
            .println(format!("Evaluation Cost: {}", cost));
        Ok(())
    }

    fn begin_of_training(
        &mut self,
        nr_epochs: usize,
        list_net: &ListNet,
    ) -> Result<(), Self::Error> {
        self.start_time = Some(Instant::now());
        info!("Begin of training for {}", nr_epochs);
        if self.setting.dump_initial_parameters {
            trace!("Dumping initial parameters.");
            self.save_parameters(list_net.clone(), "initial")?;
        }

        let multi_bar = MultiProgress::new();
        multi_bar.add(self.train_progress_bar.clone());
        multi_bar.add(self.epoch_progress_bar.clone());
        multi_bar.add(self.eval_progress_bar.clone());
        // Needed or else bars won't print to the screen.
        std::thread::spawn(move || multi_bar.join());

        self.train_progress_bar
            .set_length(nr_epochs.try_into().unwrap());

        self.train_progress_bar.enable_steady_tick(100);
        self.epoch_progress_bar.tick();
        self.eval_progress_bar.set_message("--");
        Ok(())
    }

    fn end_of_training(&mut self) -> Result<(), Self::Error> {
        let elapsed = self.start_time.map(|t| t.elapsed()).unwrap_or_default();
        info!("End of training. Duration: {}", FormattedDuration(elapsed));
        Ok(())
    }

    fn training_result(self, list_net: ListNet) -> Result<Self::Outcome, Self::Error> {
        self.epoch_progress_bar.finish_at_current_pos();
        self.train_progress_bar.finish_at_current_pos();
        self.eval_progress_bar.finish_at_current_pos();
        info!("Save final ListNet parameters.");
        self.save_parameters(list_net, "final")?;
        Ok(())
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
