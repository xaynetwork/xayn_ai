use std::process::exit;

use anyhow::Error;
use structopt::StructOpt;

use crate::exit_code::FATAL_ERROR;

mod call_data;
mod exit_code;

/// Tooling for the developers of XaynAi.
#[derive(StructOpt, Debug)]
enum CommandArgs {
    RunCallData(call_data::CallDataCmd),
}

impl CommandArgs {
    fn run(self) -> Result<i32, Error> {
        match self {
            CommandArgs::RunCallData(cmd) => cmd.run(),
        }
    }
}

fn main() {
    let exit_code = match CommandArgs::from_args().run() {
        Ok(exit_code) => exit_code,
        Err(error) => {
            eprintln!("{:?}", error);
            FATAL_ERROR
        }
    };

    exit(exit_code);
}
