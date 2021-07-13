use std::path::PathBuf;

use bincode::Error;
use log::{info, trace};
use xayn_ai::list_net::{ListNet, TrainingController};

pub(crate) struct CliTrainingControllerBuilder {
    pub(crate) out_dir: PathBuf,
    pub(crate) dump_initial_parameters: bool,
    pub(crate) dump_every: Option<usize>,
}

impl CliTrainingControllerBuilder {
    pub(crate) fn build(self) -> CliTrainingController {
        CliTrainingController {
            setting: self,
            current_epoch: 0,
            current_batch: 0,
        }
    }
}

pub(crate) struct CliTrainingController {
    setting: CliTrainingControllerBuilder,
    current_epoch: usize,
    current_batch: usize,
}

impl CliTrainingController {
    fn save_parameters(&self, list_net: ListNet, suffix: &str) -> Result<(), Error> {
        let file_path = self
            .setting
            .out_dir
            .join(format!("list_net_{}.binparams", suffix));
        list_net.serialize_into_file(file_path).map_err(Into::into)
    }
}

impl TrainingController for CliTrainingController {
    type Error = Error;

    type Outcome = ();

    fn begin_of_batch(&mut self) -> Result<(), Self::Error> {
        info!("Start of batch #{}", self.current_batch);
        Ok(())
    }

    fn end_of_batch(&mut self, losses: Vec<f32>) -> Result<(), Self::Error> {
        let loss = mean_loss(&losses);
        info!("End of batch #{}, mean loss = {}", self.current_batch, loss);
        self.current_batch += 1;
        Ok(())
    }

    fn begin_of_epoch(&mut self, _list_net: &ListNet) -> Result<(), Self::Error> {
        info!("Begin of epoch #{}", self.current_epoch);
        Ok(())
    }

    fn end_of_epoch(
        &mut self,
        list_net: &ListNet,
        mean_kl_divergence_evaluation: Option<f32>,
    ) -> Result<(), Self::Error> {
        info!(
            "End of epoch #{}, Mean eval. KL-Divergence {:?}",
            self.current_epoch, mean_kl_divergence_evaluation
        );
        if let Some(dump_every) = self.setting.dump_every {
            if (self.current_epoch + 1) % dump_every == 0 {
                trace!("Storing Parameters");
                self.save_parameters(list_net.clone(), &format!("{}", self.current_epoch))?;
            }
        }
        self.current_epoch += 1;
        Ok(())
    }

    fn begin_of_training(&mut self, list_net: &ListNet) -> Result<(), Self::Error> {
        info!("Begin of training");
        if self.setting.dump_initial_parameters {
            trace!("Storing Initial Parameters");
            self.save_parameters(list_net.clone(), "initial")?;
        }
        Ok(())
    }

    fn end_of_training(&mut self) -> Result<(), Self::Error> {
        info!("End of training");
        Ok(())
    }

    fn training_result(self, list_net: ListNet) -> Result<Self::Outcome, Self::Error> {
        self.save_parameters(list_net, "final")?;
        Ok(())
    }
}

fn mean_loss(losses: &[f32]) -> f32 {
    let count = losses.len() as f32;
    losses.iter().fold(0f32, |acc, v| acc + v / count)
}
