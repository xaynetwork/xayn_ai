#![cfg(not(tarpaulin))]
use anyhow::Error;
use structopt::StructOpt;

use self::{convert::ConvertCmd, evaluate::EvaluateCmd, train::TrainCmd};

mod cli_callbacks;
mod convert;
mod data_source;
mod evaluate;
mod train;

/// Commands related to training ListNet (train, convert, evaluate).
#[derive(StructOpt, Debug)]
pub enum ListNetCmd {
    Convert(ConvertCmd),
    Train(TrainCmd),
    Evaluate(EvaluateCmd),
}

impl ListNetCmd {
    pub fn run(self) -> Result<i32, Error> {
        use ListNetCmd::*;
        match self {
            Convert(cmd) => cmd.run(),
            Train(cmd) => cmd.run(),
            Evaluate(cmd) => cmd.run(),
        }
    }
}
