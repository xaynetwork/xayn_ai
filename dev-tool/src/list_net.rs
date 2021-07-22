use anyhow::Error;
use structopt::StructOpt;

use self::{convert::ConvertCmd, inspect::InspectCmd, train::TrainCmd};

mod cli_callbacks;
mod convert;
mod data_source;
mod inspect;
mod train;

#[derive(StructOpt, Debug)]
pub enum ListNetCmd {
    Train(TrainCmd),
    Convert(ConvertCmd),
    Inspect(InspectCmd),
}

impl ListNetCmd {
    pub fn run(self) -> Result<i32, Error> {
        use ListNetCmd::*;
        match self {
            Train(cmd) => cmd.run(),
            Convert(cmd) => cmd.run(),
            Inspect(cmd) => cmd.run(),
        }
    }
}
