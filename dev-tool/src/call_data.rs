#![cfg(not(tarpaulin))]
use std::{
    env::current_dir,
    fs::File,
    io::{self, BufReader},
    path::{Path, PathBuf},
};

use anyhow::{anyhow, Context, Error};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use structopt::StructOpt;

use xayn_ai::{
    Analytics,
    Builder as AiBuilder,
    Document,
    DocumentHistory,
    RerankMode,
    RerankingOutcomes,
};

use crate::exit_code::{NON_FATAL_ERROR, NO_ERROR};

/// Run a debug call data dump.
#[derive(StructOpt, Debug)]
pub struct CallDataCmd {
    /// The directory with the model data.
    #[structopt(long)]
    pub data_dir: Option<PathBuf>,

    /// Run the reranking with the document/history n times to change the internal state.
    #[structopt(short, long, default_value = "0")]
    pub pre_run: usize,

    /// If true return the serialized state from after running the reranking.
    #[structopt(long)]
    pub return_serialized_state: bool,

    #[structopt(long)]
    pub pretty: bool,

    /// The file with the call data JSON dump.
    pub call_data: PathBuf,
}

impl CallDataCmd {
    const QAMBERT_MODEL_PATH: &'static str = "qambert_v0001/qambert.onnx";
    const QAMBERT_VOCAB_PATH: &'static str = "qambert_v0001/vocab.txt";
    const SMBERT_MODEL_PATH: &'static str = "smbert_v0000/smbert.onnx";
    const SMBERT_VOCAB_PATH: &'static str = "smbert_v0000/vocab.txt";
    const LTR_MODEL_PATH: &'static str = "ltr_v0000/ltr.binparams";

    pub fn run(self) -> Result<i32, Error> {
        let CallDataCmd {
            data_dir,
            pre_run,
            return_serialized_state,
            pretty,
            call_data,
        } = self;

        check_call_data_path(&call_data)?;

        let data_dir = data_dir.map_or_else(find_data_dir, Ok)?;

        check_data_dir(&data_dir)?;

        let call_data = CallData::load_from_file(&call_data).context("Parsing Call Data Failed")?;

        let mut xayn_ai = AiBuilder::default()
            .with_serialized_database(call_data.serialized_state)
            .context("Deserializing database failed.")?
            .with_qambert_from_file(
                data_dir.join(Self::QAMBERT_VOCAB_PATH),
                data_dir.join(Self::QAMBERT_MODEL_PATH),
            )
            .context("Loading QA-mBert failed.")?
            .with_smbert_from_file(
                data_dir.join(Self::SMBERT_VOCAB_PATH),
                data_dir.join(Self::SMBERT_MODEL_PATH),
            )
            .context("Loading S-mBert failed.")?
            .with_domain_from_file(data_dir.join(Self::LTR_MODEL_PATH))
            .context("Loading LTR failed.")?
            .build()
            .context("Building XaynAi failed.")?;

        for _ in 0..pre_run {
            xayn_ai.rerank(
                call_data.rerank_mode,
                &call_data.histories,
                &call_data.documents,
            );
        }

        let outcomes = xayn_ai.rerank(
            call_data.rerank_mode,
            &call_data.histories,
            &call_data.documents,
        );
        let analytics = xayn_ai.analytics().cloned();
        let errors = xayn_ai.errors();
        let new_serialized_state = if return_serialized_state {
            Some(xayn_ai.serialize().context("Serializing State Failed")?)
        } else {
            None
        };

        for error in errors {
            eprintln!("{}", error);
        }

        let result = CallDataCmdResult {
            outcomes,
            analytics,
            new_serialized_state,
        };

        let serialized = if pretty {
            serde_json::to_string_pretty(&result)?
        } else {
            serde_json::to_string(&result)?
        };

        println!("{}", serialized);

        let exit_code = if errors.is_empty() {
            NO_ERROR
        } else {
            NON_FATAL_ERROR
        };

        Ok(exit_code)
    }
}

fn check_data_dir(dir: &Path) -> Result<(), Error> {
    if dir.is_dir() {
        Ok(())
    } else {
        Err(anyhow!("Data Dir is not a dir: {}", dir.display()))
    }
}

fn check_call_data_path(path: &Path) -> Result<(), Error> {
    if path.is_file() {
        Ok(())
    } else {
        Err(anyhow!("Call Data File is not a file: {}", path.display()))
    }
}

fn find_data_dir() -> Result<PathBuf, Error> {
    let current_dir = current_dir()?;
    let mut base_dir: &Path = &current_dir;
    let mut data_dir = base_dir.join("data");
    if data_dir.exists() {
        return Ok(data_dir);
    }
    while let Some(parent) = base_dir.parent() {
        base_dir = parent;
        data_dir = base_dir.join("data");
        if data_dir.exists() {
            return Ok(data_dir);
        }
        if base_dir.join(".git").exists() {
            break;
        }
    }

    return Err(anyhow!(
        "No Data Dir found in {} or dirs up to {}",
        current_dir.display(),
        base_dir.display()
    ));
}

#[derive(Deserialize)]
struct CallData {
    rerank_mode: RerankMode,
    histories: Vec<DocumentHistory>,
    documents: Vec<Document>,
    #[serde(deserialize_with = "deser_opt_bytes_from_base64")]
    serialized_state: Option<Vec<u8>>,
}

impl CallData {
    fn load_from_file(path: &Path) -> Result<Self, io::Error> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        serde_json::from_reader(reader).map_err(Into::into)
    }
}

#[derive(Serialize)]
struct CallDataCmdResult {
    outcomes: RerankingOutcomes,
    analytics: Option<Analytics>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(serialize_with = "ser_opt_bytes_to_base64")]
    new_serialized_state: Option<Vec<u8>>,
}

fn ser_opt_bytes_to_base64<S>(bytes: &Option<Vec<u8>>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    let encoded = bytes.as_ref().map(base64::encode);
    encoded.serialize(serializer)
}

fn deser_opt_bytes_from_base64<'de, D>(deserializer: D) -> Result<Option<Vec<u8>>, D::Error>
where
    D: Deserializer<'de>,
{
    if let Some(encoded) = Option::<String>::deserialize(deserializer)? {
        base64::decode(encoded)
            .map(Some)
            .map_err(<D::Error as serde::de::Error>::custom)
    } else {
        Ok(None)
    }
}
