use anyhow::Error;
use structopt::StructOpt;

use crate::exit_code::NO_ERROR;

use self::{convert::ConvertCmd, train::TrainCmd};

mod cli_callbacks;
mod convert;
mod data_source;
mod train;

#[derive(StructOpt, Debug)]
pub enum ListNetCmd {
    Train(TrainCmd),
    Convert(ConvertCmd),
}

impl ListNetCmd {
    pub fn run(self) -> Result<i32, Error> {
        use ListNetCmd::*;
        match self {
            Train(cmd) => cmd.run()?,
            Convert(cmd) => cmd.run()?,
        }
        Ok(NO_ERROR)
    }
}
