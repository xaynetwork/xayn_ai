#![cfg(not(tarpaulin))]
use std::{path::PathBuf, str::FromStr};

use anyhow::{bail, Context, Error};
use ndarray::Array2;
use rand::prelude::ThreadRng;
use structopt::StructOpt;
use xayn_ai::list_net::{
    ndutils::initializer::{
        glorot_normal_weights_init,
        glorot_uniform_weights_init,
        he_normal_weights_init,
        he_uniform_weights_init,
    },
    optimizer::MiniBatchSgd,
    ListNet,
    ListNetTrainer,
};

use crate::{exit_code::NO_ERROR, utils::progress_spin_until_done};

use super::{
    cli_callbacks::CliTrainingControllerBuilder,
    data_source::{DataSource, InMemorySamples},
};

/// Trains a ListNet.
#[derive(StructOpt, Debug)]
pub struct TrainCmd {
    /// A file containing samples we can train and evaluate on.
    #[structopt(long)]
    samples: PathBuf,

    /// The number of epochs to run.
    #[structopt(long)]
    epochs: usize,

    /// The batch size to use.
    ///
    /// Setting the `batch-size` to `0` will automatically
    /// set it to the number of training samples. I.e. there
    /// will only be one batch per epoch.
    ///
    /// WARNING: This is not optimized for a `0` `batch-size`
    /// with huge number of samples in the batch. It's mainly
    /// meant to be used with XaynNet emulation modes.
    #[structopt(long, default_value = "32")]
    batch_size: usize,

    /// The percent of samples to use for evaluation.
    ///
    /// The evaluation samples will be taken
    /// from the back of the data source. This
    /// means the same split used with the same
    /// samples file will yield the exact same
    /// training and evaluation samples every
    /// time.
    #[structopt(long, default_value = "0.2")]
    evaluation_split: f32,

    /// The learning rate to use.
    #[structopt(long, default_value = "0.1")]
    learning_rate: f32,

    /// Uses given parameters instead of initializing them randomly.
    #[structopt(long)]
    use_initial_parameters: Option<PathBuf>,

    /// Directory to store outputs in.
    #[structopt(short, long, default_value = "./")]
    out_dir: PathBuf,

    /// After how many epochs a intermediate result should be dumped (if at all).
    #[structopt(long)]
    dump_every: Option<usize>,

    /// Dumps the initial parameters before any training was done.
    #[structopt(long)]
    dump_initial_parameters: bool,

    /// Selects the weight initializer.
    ///
    /// Is ignored if `use_initial_parameters` is used.
    #[structopt(long, default_value = "he-normal", parse(try_from_str))]
    initializer: WeightInitializer,
}

impl TrainCmd {
    pub fn run(self) -> Result<i32, Error> {
        let TrainCmd {
            samples,
            epochs,
            batch_size,
            evaluation_split,
            learning_rate,
            use_initial_parameters,
            out_dir,
            dump_every,
            dump_initial_parameters,
            initializer,
        } = self;

        let data_source = progress_spin_until_done("Loading samples", || {
            let storage = InMemorySamples::deserialize_from_file(samples)
                .context("Loading training & evaluation samples failed.")?;
            DataSource::new(storage, evaluation_split, batch_size)
                .context("Creating DataSource failed.")
        })?;

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
            ListNet::new_with_random_weights(initializer.as_fn())
        };

        let trainer = ListNetTrainer::new(list_net, data_source, callbacks, optimizer);
        trainer.train(epochs)?;
        Ok(NO_ERROR)
    }
}

#[derive(Debug, Clone, Copy)]
enum WeightInitializer {
    HeNormal,
    HeUniform,
    GlorotNormal,
    GlorotUniform,
}

impl WeightInitializer {
    fn as_fn(self) -> for<'r> fn(&'r mut ThreadRng, (usize, usize)) -> Array2<f32> {
        use WeightInitializer::*;

        match self {
            HeNormal => he_normal_weights_init,
            HeUniform => he_uniform_weights_init,
            GlorotNormal => glorot_normal_weights_init,
            GlorotUniform => glorot_uniform_weights_init,
        }
    }
}

impl FromStr for WeightInitializer {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        use WeightInitializer::*;

        let res = match s.trim() {
            "he-normal" => HeNormal,
            "he-uniform" => HeUniform,
            "glorot-normal" => GlorotNormal,
            "glorot-uniform" => GlorotUniform,
            _ => bail!("Unexpected weight initializer use he/glorot-uniform/normal."),
        };
        Ok(res)
    }
}

impl Default for WeightInitializer {
    fn default() -> Self {
        Self::HeNormal
    }
}
