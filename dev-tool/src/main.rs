#![cfg(not(tarpaulin))]
#![cfg(not(any(target_os = "android", target_os = "ios")))]
use std::{process::exit, time::Instant};

use anyhow::Error;
use indicatif::HumanDuration;
use log::error;
use structopt::StructOpt;

use crate::exit_code::{FATAL_ERROR, NO_ERROR};

mod call_data;
mod exit_code;
mod list_net;
mod utils;

/// Tooling for the developers of XaynAi.
#[derive(StructOpt, Debug)]
enum CommandArgs {
    RunCallData(call_data::CallDataCmd),
    ListNet(list_net::ListNetCmd),
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

    let start_time = Instant::now();

    let exit_code = match CommandArgs::from_args().run() {
        Ok(exit_code) => exit_code,
        Err(error) => {
            error!("FATAL: {}\n{:?}", error, error);
            FATAL_ERROR
        }
    };

    let duration = HumanDuration(start_time.elapsed());
    if exit_code == NO_ERROR {
        eprintln!("DONE ({})", duration);
    } else {
        eprintln!("EXIT WITH ERRORS ({})", duration);
    }
    exit(exit_code);
}
