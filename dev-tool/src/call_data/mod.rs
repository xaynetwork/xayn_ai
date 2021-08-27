use std::{
    fs::File,
    io::{self, BufReader, BufWriter},
    path::Path,
};

use anyhow::Error;
use serde::{Deserialize, Serialize};
use structopt::StructOpt;

use xayn_ai::{Document, DocumentHistory, RerankMode};

use self::{generate::GenerateCallDataCmd, run::RunCallDataCmd};
use crate::utils::serde_opt_bytes_as_base64;

mod generate;
mod run;

/// Commands related to training ListNet (train, convert, evaluate).
#[derive(StructOpt, Debug)]
pub enum CallDataCmd {
    Generate(GenerateCallDataCmd),
    Run(RunCallDataCmd),
}

impl CallDataCmd {
    pub fn run(self) -> Result<i32, Error> {
        use CallDataCmd::*;
        match self {
            Generate(cmd) => cmd.run(),
            Run(cmd) => cmd.run(),
        }
    }
}

#[derive(Deserialize, Serialize)]
struct CallData {
    rerank_mode: RerankMode,
    histories: Vec<DocumentHistory>,
    documents: Vec<Document>,
    #[serde(with = "serde_opt_bytes_as_base64")]
    serialized_state: Option<Vec<u8>>,
}

impl CallData {
    fn load_from_file(path: &Path) -> Result<Self, io::Error> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        serde_json::from_reader(reader).map_err(Into::into)
    }

    fn save_to_file(&self, path: &Path) -> Result<(), io::Error> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, self).map_err(Into::into)
    }
}
