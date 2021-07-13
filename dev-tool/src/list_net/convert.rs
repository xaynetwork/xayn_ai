use std::{
    ffi::OsStr,
    fs,
    path::{Path, PathBuf},
};

use anyhow::Error;
use structopt::StructOpt;
use xayn_ai::{list_net_training_data_from_history, DocumentHistory};

use super::data_source::InMemoryData;

#[derive(Debug, StructOpt)]
pub struct ConvertCmd {
    #[structopt(short = "d", long)]
    soundgarden_user_df_dir: PathBuf,
    #[structopt(short = "o", long)]
    out: PathBuf,
}

impl ConvertCmd {
    pub fn run(self) -> Result<(), Error> {
        let ConvertCmd {
            soundgarden_user_df_dir,
            out,
        } = self;

        let mut storage = InMemoryData::default();
        for entry in fs::read_dir(&soundgarden_user_df_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension() != Some(OsStr::new("csv")) {
                continue;
            }

            let history = load_history(path)?;
            for (inputs, target_prob_dist) in list_net_training_data_from_history(&history) {
                storage.add_sample(inputs.view(), target_prob_dist.view());
            }
        }

        storage.write_to_file(out)?;

        Ok(())
    }
}

//FIXME[follow up PR]: Change ltr feature extraction tests to also use this (or at least reuse some code and use the soundgarden user df csv data format)
fn load_history(_path: impl AsRef<Path>) -> Result<Vec<DocumentHistory>, Error> {
    todo!();
}
