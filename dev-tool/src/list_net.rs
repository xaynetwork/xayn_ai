#![cfg(not(tarpaulin))]
use anyhow::Error;
use structopt::StructOpt;

use self::{convert::ConvertCmd, evaluate::EvaluateCmd, inspect::InspectCmd, train::TrainCmd};

mod cli_callbacks;
mod convert;
mod data_source;
mod evaluate;
mod inspect;
mod train;

/// Commands related to training ListNet (train, convert, inspect).
#[derive(StructOpt, Debug)]
pub enum ListNetCmd {
    Train(TrainCmd),
    Convert(ConvertCmd),
    Inspect(InspectCmd),
    Evaluate(EvaluateCmd),
}

impl ListNetCmd {
    pub fn run(self) -> Result<i32, Error> {
        use ListNetCmd::*;
        match self {
            Train(cmd) => cmd.run(),
            Convert(cmd) => cmd.run(),
            Inspect(cmd) => cmd.run(),
            Evaluate(cmd) => cmd.run(),
        }
    }
}
