#![cfg(not(tarpaulin))]
use std::process::exit;

use anyhow::Error;
use log::error;
use structopt::StructOpt;

use crate::exit_code::{FATAL_ERROR, NO_ERROR};

mod call_data;
mod exit_code;
mod list_net;

/// Tooling for the developers of XaynAi.
#[derive(StructOpt, Debug)]
enum CommandArgs {
    RunCallData(call_data::CallDataCmd),
    ListNet(list_net::ListNetCmd),
    //TODO BinParams(...) dev-tool bin-params inspect --print-data <file>
}

impl CommandArgs {
    fn run(self) -> Result<i32, Error> {
        use CommandArgs::*;

        match self {
            RunCallData(cmd) => cmd.run(),
            ListNet(cmd) => cmd.run(),
        }
    }
}

fn main() {
    env_logger::init();

    let exit_code = match CommandArgs::from_args().run() {
        Ok(exit_code) => exit_code,
        Err(error) => {
            error!("FATAL: {}\n{:?}", error, error);
            FATAL_ERROR
        }
    };

    if exit_code == NO_ERROR {
        eprint!("DONE");
    } else {
        eprint!("EXIT WITH ERRORS");
    }
    exit(exit_code);
}
