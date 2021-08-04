#![cfg(not(tarpaulin))]
use anyhow::Error;
use structopt::StructOpt;

use self::convert::ConvertCmd;
mod convert;
mod data_source;

/// Commands related to training ListNet (train, convert, inspect).
#[derive(StructOpt, Debug)]
pub enum ListNetCmd {
    Convert(ConvertCmd),
}

impl ListNetCmd {
    pub fn run(self) -> Result<i32, Error> {
        use ListNetCmd::*;
        match self {
            Convert(cmd) => cmd.run(),
        }
    }
}
