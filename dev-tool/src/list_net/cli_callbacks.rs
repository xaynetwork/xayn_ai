use std::{convert::TryInto, path::PathBuf, time::Instant};

use bincode::Error;
use indicatif::{FormattedDuration, MultiProgress, ProgressBar, ProgressStyle};
use log::{debug, info, trace};
use xayn_ai::list_net::{ListNet, TrainingController};

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
        CliTrainingController {
            setting: self,
            current_epoch: 0,
            current_batch: 0,
            start_time: None,
            epoch_progress_bar: None,
            train_progress_bar: None,
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
    /// A (CLI) progress bar used to track the progress of the current epoch.
    epoch_progress_bar: Option<ProgressBar>,
    /// A (CLI) progress bar used to track the progress of the current training.
    train_progress_bar: Option<ProgressBar>,
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

    fn begin_of_batch(&mut self) -> Result<(), Self::Error> {
        trace!("Start of batch #{}", self.current_batch);
        Ok(())
    }

    fn end_of_batch(&mut self, losses: Vec<f32>) -> Result<(), Self::Error> {
        let loss = mean_loss(&losses);
        trace!("End of batch #{}, mean loss = {}", self.current_batch, loss);
        if let Some(bar) = &self.epoch_progress_bar {
            bar.inc(1);
        }
        self.current_batch += 1;
        Ok(())
    }

    fn begin_of_epoch(
        &mut self,
        nr_batches: usize,
        _list_net: &ListNet,
    ) -> Result<(), Self::Error> {
        debug!(
            "Begin of epoch #{:0>4} (#batch {})",
            self.current_epoch, nr_batches
        );
        self.current_batch = 0;
        if let Some(bar) = &self.epoch_progress_bar {
            bar.set_position(0);
            bar.set_length(nr_batches.try_into().unwrap());
        }
        Ok(())
    }

    fn end_of_epoch(
        &mut self,
        list_net: &ListNet,
        mean_kl_divergence_evaluation: Option<f32>,
    ) -> Result<(), Self::Error> {
        debug!(
            "End of epoch #{}, Mean eval. KL-Divergence {:?}",
            self.current_epoch, mean_kl_divergence_evaluation
        );

        if let Some(dump_every) = self.setting.dump_every {
            if (self.current_epoch + 1) % dump_every == 0 {
                trace!("Dumping parameters.");
                self.save_parameters(list_net.clone(), &format!("{}", self.current_epoch))?;
            }
        }

        if let Some(bar) = &self.train_progress_bar {
            bar.inc(1);
        }

        if let Some(cost) = mean_kl_divergence_evaluation {
            //FIXME This can always happen (after a longer training, mainly
            //      if training with inadequate data or parameters). Still
            //      we want to handle this better in the future.
            if cost.is_nan() {
                panic!("evaluation KL-Divergence cost is NaN");
            }
            if let Some(bar) = &self.train_progress_bar {
                bar.println(format!("Evaluation Cost: {}", cost));
            }
        }

        self.current_epoch += 1;
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

        let multibar = MultiProgress::new();
        let train_bar = ProgressBar::new(nr_epochs.try_into().unwrap());
        train_bar.set_style(
            ProgressStyle::default_bar()
                .template("Epochs:  [{bar:50.green}] {percent:>3}% ({pos:>5}/{len:>5})")
                .progress_chars("=> "),
        );
        multibar.add(train_bar.clone());
        let epoch_bar = ProgressBar::new(1);
        epoch_bar.set_style(
            ProgressStyle::default_bar()
                .template("Batches: [{bar:50.green}] {percent:>3}% ({pos:>5}/{len:>5})")
                .progress_chars("=> "),
        );
        multibar.add(epoch_bar.clone());
        // Needed or else bars won't print to the screen.
        std::thread::spawn(move || multibar.join());
        train_bar.tick();
        epoch_bar.tick();

        self.train_progress_bar = Some(train_bar);
        self.epoch_progress_bar = Some(epoch_bar);
        Ok(())
    }

    fn end_of_training(&mut self) -> Result<(), Self::Error> {
        let elapsed = self.start_time.map(|t| t.elapsed()).unwrap_or_default();
        info!("End of training. Duration: {}", FormattedDuration(elapsed));
        Ok(())
    }

    fn training_result(self, list_net: ListNet) -> Result<Self::Outcome, Self::Error> {
        if let Some(bar) = &self.epoch_progress_bar {
            bar.finish_at_current_pos();
        }
        if let Some(bar) = &self.train_progress_bar {
            bar.finish_at_current_pos();
        }
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
