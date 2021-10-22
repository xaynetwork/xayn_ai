#![cfg(not(tarpaulin))]

use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::{Context, Error};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use rand::{prelude::SliceRandom, thread_rng, Rng};
use rayon::iter::{ParallelBridge, ParallelIterator};
use structopt::StructOpt;

use super::{
    cli_callbacks::{
        CliTrainingControllerBuilder,
        NoFeedback,
        OutputDir,
        ProgressBarTrainingFeedback,
    },
    data_source::{DataSource, InMemoryStorage, SplitDataSource},
    evaluate::run_evaluation,
};
use crate::{exit_code::NO_ERROR, utils::progress_spin_until_done};
use xayn_ai::list_net::{
    data::DataSource as _,
    model::ListNet,
    optimizer::MiniBatchSgd,
    trainer::ListNetTrainer,
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
    /// meant to be used with XayNet emulation modes.
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

    /// After how many epochs an intermediate result should be dumped (if at all).
    ///
    /// If used with XayNet emulation, `dump_every` determines after how many XayNet
    /// iterations the emulator should dump the current parameters.
    #[structopt(long)]
    dump_every: Option<usize>,

    /// Dumps the initial parameters before any training was done.
    #[structopt(long)]
    dump_initial_parameters: bool,

    /// Enables XayNet Emulation mode.
    ///
    /// In this mode the ListNet is distributed to N users. Each user
    /// trains it using the given parameters. Then the
    /// separately trained ListNets are merged. This is then
    /// repeated for a number of times leading to the final
    /// trained ListNet.
    ///
    /// Requires: --xne-steps, --xne-users-per-step and --xne-merge-users
    #[structopt(long, requires_all(&["xne-steps", "xne-users-per-step", "xne-merge-users"]))]
    xayn_net_emulation: bool,

    /// The number of fork-join training steps, similar to epoch for normal training.
    ///
    /// Epochs will still be applied for each user separately.
    #[structopt(long, requires("xayn-net-emulation"))]
    xne_steps: Option<usize>,

    /// How many users should be included in the training per-step.
    #[structopt(long, requires("xayn-net-emulation"))]
    xne_users_per_step: Option<usize>,

    /// How many users should be merged and treated as one user.
    #[structopt(long, requires("xayn-net-emulation"))]
    xne_merge_users: Option<usize>,
}

struct XayNetEmulationSettings {
    steps: usize,
    users_per_step: usize,
    merge_users: usize,
}

impl TrainCmd {
    pub fn run(self) -> Result<i32, Error> {
        let (shared_setup, xayn_net_emulation_setup) = self.shared_setup()?;
        let file_output = shared_setup.file_output.clone();
        let list_net = if let Some(emulation_setup) = xayn_net_emulation_setup {
            shared_setup.soundgarden_xayn_net_emulation_training(emulation_setup)?
        } else {
            shared_setup.soundgarden_classic_training()?
        };
        file_output.save_list_net_parameters(list_net, OutputDir::SUFFIX_FINAL_LIST_NET)?;
        Ok(NO_ERROR)
    }

    fn shared_setup(self) -> Result<(SharedSetup, Option<XayNetEmulationSettings>), Error> {
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
            xayn_net_emulation,
            xne_steps,
            xne_users_per_step,
            xne_merge_users,
        } = self;

        let storage = load_sample(samples)?;
        let file_output = OutputDir::new(out_dir);
        let list_net = create_initial_list_net(use_initial_parameters)?;
        let optimizer = MiniBatchSgd { learning_rate };
        if dump_initial_parameters {
            file_output
                .save_list_net_parameters(list_net.clone(), OutputDir::SUFFIX_INITIAL_LIST_NET)?;
        }

        let xayn_net_emulation = xayn_net_emulation.then(|| XayNetEmulationSettings {
            steps: xne_steps.unwrap(),
            users_per_step: xne_users_per_step.unwrap(),
            merge_users: xne_merge_users.unwrap(),
        });

        let shared_setup = SharedSetup {
            storage,
            file_output,
            list_net,
            optimizer,
            evaluation_split,
            batch_size,
            epochs,
            dump_every,
        };

        Ok((shared_setup, xayn_net_emulation))
    }
}

struct SharedSetup {
    storage: Arc<InMemoryStorage>,
    file_output: OutputDir,
    list_net: ListNet,
    optimizer: MiniBatchSgd,
    evaluation_split: f32,
    batch_size: usize,
    epochs: usize,
    dump_every: Option<usize>,
}

impl SharedSetup {
    fn soundgarden_classic_training(self) -> Result<ListNet, Error> {
        let Self {
            storage,
            file_output,
            list_net,
            optimizer,
            evaluation_split,
            batch_size,
            epochs,
            dump_every,
        } = self;

        let data_source = DataSource::new(storage, evaluation_split, batch_size)
            .context("Creating DataSource failed.")?;

        let controller = CliTrainingControllerBuilder {
            file_output: file_output.clone(),
            dump_every,
            training_feedback: ProgressBarTrainingFeedback::new(),
        }
        .build();

        let trainer = ListNetTrainer::new(list_net, data_source, controller, optimizer);
        let list_net = trainer.train(epochs)?;
        file_output.save_list_net_parameters(list_net.clone(), OutputDir::SUFFIX_FINAL_LIST_NET)?;
        Ok(list_net)
    }

    fn soundgarden_xayn_net_emulation_training(
        self,
        xayn_net_emu_setup: XayNetEmulationSettings,
    ) -> Result<ListNet, Error> {
        let Self {
            storage,
            file_output,
            mut list_net,
            optimizer,
            evaluation_split,
            batch_size,
            epochs,
            dump_every,
        } = self;

        let XayNetEmulationSettings {
            steps,
            users_per_step,
            merge_users,
        } = xayn_net_emu_setup;

        let SplitDataSource {
            training_only_sources,
            mut evaluation_only_source,
        } = DataSource::new_split(storage, evaluation_split, batch_size, merge_users)
            .context("Creating DataSource failed.")?;

        let train_bar = ProgressBar::new(steps as u64);
        train_bar.set_style(
            ProgressStyle::default_bar()
                .template("XNE-Steps:  [{bar:27.green}] {percent:>3}% ({pos:>5}/{len:>5}) {elapsed_precise}")
                .progress_chars("=> "),
        );
        let users_bar = ProgressBar::new(users_per_step as u64);
        users_bar.set_style(
            ProgressStyle::default_bar()
                .template("Users:      [{bar:27.green}] {percent:>3}% ({pos:>5}/{len:>5})")
                .progress_chars("=> "),
        );
        let eval_bar = ProgressBar::new(
            evaluation_only_source
                .as_ref()
                .map(|s| s.number_of_evaluation_samples())
                .unwrap_or_default() as u64,
        );
        eval_bar.set_style(
            ProgressStyle::default_bar()
                .template("Evaluation: [{bar:27.green}] {percent:>3}% ({pos:>5}/{len:>5}) {msg}")
                .progress_chars("=> "),
        );

        let multi_bar = MultiProgress::new();
        multi_bar.add(train_bar.clone());
        multi_bar.add(users_bar.clone());
        multi_bar.add(eval_bar.clone());
        std::thread::spawn(move || multi_bar.join());

        let mut rng = thread_rng();

        for step in 0..steps {
            users_bar.set_position(0);
            let list_nets: Result<Vec<_>, _> =
                sample_users(&mut rng, &training_only_sources, users_per_step)
                    .into_iter()
                    .par_bridge()
                    .map(|user_data| {
                        let controller = CliTrainingControllerBuilder {
                            file_output: file_output.clone(),
                            dump_every: None,
                            training_feedback: NoFeedback,
                        }
                        .build();

                        let trainer = ListNetTrainer::new(
                            list_net.clone(),
                            user_data,
                            controller,
                            optimizer.clone(),
                        );
                        let res = trainer.train(epochs);
                        users_bar.inc(1);
                        res
                    })
                    .collect();

            if let Some(new_net) = ListNet::merge_nets(list_nets?) {
                list_net = new_net;
            }

            if let Some(eval_source) = &mut evaluation_only_source {
                eval_bar.set_position(0);
                eval_bar.set_message("--");
                let mean_cost = run_evaluation(&list_net, eval_source, || eval_bar.inc(1))?;
                eval_bar.set_message(format!("cost={:.5}", mean_cost));
                eval_bar.println(format!("Evaluation Cost: {:.5}", mean_cost));
            }

            if let Some(every) = dump_every {
                if (step + 1) % every == 0 {
                    file_output.save_list_net_parameters(list_net.clone(), &format!("{}", step))?;
                }
            }

            train_bar.inc(1)
        }

        train_bar.finish_at_current_pos();
        users_bar.finish_at_current_pos();
        eval_bar.finish_at_current_pos();

        file_output.save_list_net_parameters(list_net.clone(), OutputDir::SUFFIX_FINAL_LIST_NET)?;
        Ok(list_net)
    }
}

fn load_sample(path: impl AsRef<Path>) -> Result<Arc<InMemoryStorage>, Error> {
    progress_spin_until_done("Loading samples", || {
        let storage = InMemoryStorage::deserialize_from_file(path)
            .context("Loading training & evaluation samples failed.")?;
        Ok(Arc::new(storage))
    })
}

fn create_initial_list_net(
    use_initial_parameters: Option<impl AsRef<Path>>,
) -> Result<ListNet, Error> {
    if let Some(initial_params_file) = use_initial_parameters {
        ListNet::deserialize_from_file(initial_params_file).map_err(Into::into)
    } else {
        Ok(ListNet::new_with_random_weights())
    }
}

fn sample_users<'a>(
    rng: &mut impl Rng,
    users: &'a [DataSource],
    count: usize,
) -> impl Iterator<Item = DataSource> + 'a {
    users.choose_multiple(rng, count).cloned()
}
