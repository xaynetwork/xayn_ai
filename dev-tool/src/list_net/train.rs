use std::path::PathBuf;

use anyhow::{Context, Error};
use structopt::StructOpt;
use xayn_ai::list_net::{optimizer::MiniBatchSgd, ListNet, ListNetTrainer};

use crate::exit_code::NO_ERROR;

use super::{
    cli_callbacks::CliTrainingControllerBuilder,
    data_source::{DataSource, InMemoryData},
};

#[derive(StructOpt, Debug)]
pub struct TrainCmd {
    /// A file containing samples we can train and evaluate on.
    #[structopt(long)]
    data: PathBuf,

    /// The number of epochs to run.
    #[structopt(long)]
    epochs: usize,

    /// The batch size to use.
    #[structopt(long, default_value = "32")]
    batch_size: usize,

    /// The percent of samples to use for evaluation.
    ///
    /// The percentage of evaluation samples will be taken
    /// from the back.
    #[structopt(long, default_value = "0.2")]
    evaluation_split: f32,

    /// The learning rate to use.
    #[structopt(long, default_value = "0.1")]
    learning_rate: f32,

    /// Uses given parameters instead of initializing them randomly.
    #[structopt(long)]
    use_initial_parameters: Option<PathBuf>,

    /// Dire to store outputs in.
    #[structopt(short, long, default_value = ".")]
    out_dir: PathBuf,

    /// After how many epochs a intermediate result should be dumped.
    #[structopt(long)]
    dump_every: Option<usize>,

    /// Dumps the initial parameters before any training was done
    #[structopt(long)]
    dump_initial_parameters: bool,
}

impl TrainCmd {
    pub fn run(self) -> Result<i32, Error> {
        let TrainCmd {
            data: database,
            epochs,
            batch_size,
            evaluation_split,
            learning_rate,
            use_initial_parameters,
            out_dir,
            dump_every,
            dump_initial_parameters,
        } = self;

        let storage = InMemoryData::deserialize_from_file(database)
            .context("loading training/eval data failed")?;
        let data_source =
            DataSource::new(storage, evaluation_split).context("Creating DataSource failed")?;

        let callbacks = CliTrainingControllerBuilder {
            out_dir,
            dump_initial_parameters,
            dump_every,
        }
        .build();

        let optimizer = MiniBatchSgd { learning_rate };

        let list_net = if let Some(initial_params_file) = use_initial_parameters {
            ListNet::deserialize_from_file(initial_params_file)?
        } else {
            ListNet::new_with_random_weights()
        };

        let trainer = ListNetTrainer::new(list_net, data_source, callbacks, optimizer);
        trainer.train(epochs, batch_size)?;
        Ok(NO_ERROR)
    }
}
