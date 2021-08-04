#![cfg(not(tarpaulin))]
use anyhow::Error;
use structopt::StructOpt;

use self::{convert::ConvertCmd, inspect::InspectCmd, train::TrainCmd};

mod cli_callbacks;
mod convert;
mod data_source;
mod inspect;
mod train;

/// Commands related to training ListNet (train, convert, inspect).
#[derive(StructOpt, Debug)]
pub enum ListNetCmd {
    Convert(ConvertCmd),
    Train(TrainCmd),
    Inspect(InspectCmd),
}

impl ListNetCmd {
    pub fn run(self) -> Result<i32, Error> {
        use ListNetCmd::*;
        match self {
            Convert(cmd) => cmd.run(),
            Train(cmd) => cmd.run(),
            Inspect(cmd) => cmd.run(),
        }
    }
}
