#![cfg(not(tarpaulin))]

use std::{iter, ops::Add, path::PathBuf, sync::Arc};

use anyhow::{bail, Context, Error};
use indicatif::{ProgressBar, ProgressStyle};
use rayon::iter::{ParallelBridge, ParallelIterator};
use structopt::StructOpt;

use super::data_source::{DataSource, InMemoryStorage};
use crate::{exit_code::NO_ERROR, utils::progress_spin_until_done};
use layer::utils::kl_divergence;
use xayn_ai::list_net::{DataSource as _, ListNet};

/// Runs a single evaluation pass on a ListNet.
#[derive(StructOpt, Debug)]
pub struct EvaluateCmd {
    /// A file containing samples we can train and evaluate on.
    #[structopt(long)]
    samples: PathBuf,

    /// File containing a ListNet parameter set.
    ///
    /// E.g. `list_net.binparams`
    #[structopt(long)]
    parameters: PathBuf,

    /// The percentage of samples to use for evaluation.
    ///
    /// The percentage of evaluation samples will be taken
    /// from the end of the sample set.
    #[structopt(long, default_value = "0.2")]
    evaluation_split: f32,
}

impl EvaluateCmd {
    pub fn run(self) -> Result<i32, Error> {
        let Self {
            samples,
            parameters,
            evaluation_split,
        } = self;

        let mut data_source = progress_spin_until_done("Loading samples", || {
            let storage = InMemoryStorage::deserialize_from_file(samples)
                .context("Loading training & evaluation samples failed.")?;
            DataSource::new(Arc::new(storage), evaluation_split, 1)
                .context("Creating DataSource failed.")
        })?;

        let list_net = ListNet::deserialize_from_file(parameters)?;

        let nr_samples = data_source.number_of_evaluation_samples();
        let progress_bar = ProgressBar::new(nr_samples as u64);
        progress_bar.set_style(
            ProgressStyle::default_bar()
                .template("Evaluation: [{bar:27.green}] {percent:>3}% ({pos:>5}/{len:>5}) {elapsed_precise}")
                .progress_chars("=> "),
        );
        progress_bar.tick();

        let mean_cost = run_evaluation(&list_net, &mut data_source, || progress_bar.inc(1))?;

        progress_bar.finish();
        println!("mean_evaluation_cost={}", mean_cost);
        Ok(NO_ERROR)
    }
}

pub(crate) fn run_evaluation(
    list_net: &ListNet,
    data_source: &mut DataSource,
    progress_hint: impl Fn() + Sync + Send,
) -> Result<f32, Error> {
    let nr_samples = data_source.number_of_evaluation_samples();

    if nr_samples == 0 {
        bail!("No evaluation samples with given data source and evaluation split");
    }

    // Make sure we always iterate over all evaluation samples.
    data_source.reset();

    let mut error_slot = None;

    let iter = iter::from_fn(|| match data_source.next_evaluation_sample() {
        Ok(v) => v,
        Err(err) => {
            error_slot = Some(err);
            None
        }
    });

    let nr_samples = nr_samples as f32;
    let mean_cost = iter
        .par_bridge()
        .map(|sample| {
            let cost = list_net.evaluate(kl_divergence, sample.as_view());
            progress_hint();
            cost
        })
        .fold(|| 0.0, |acc, cost| acc + cost / nr_samples)
        .reduce_with(Add::add)
        .unwrap();

    if let Some(error) = error_slot {
        Err(error.into())
    } else {
        Ok(mean_cost)
    }
}
