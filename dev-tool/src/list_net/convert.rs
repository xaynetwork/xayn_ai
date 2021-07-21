use std::{
    convert::TryInto,
    ffi::OsStr,
    fs,
    path::{Path, PathBuf},
};

use anyhow::Error;
use itertools::Itertools;
use log::debug;
use serde::Deserialize;
use structopt::StructOpt;
use uuid::Uuid;
use xayn_ai::{
    list_net_training_data_from_history,
    DayOfWeek,
    DocumentHistory,
    DocumentId,
    QueryId,
    Relevance,
    SessionId,
    UserAction,
    UserFeedback,
};

use super::data_source::InMemoryData;

#[derive(Debug, StructOpt)]
pub struct ConvertCmd {
    #[structopt(short = "d", long)]
    soundgarden_user_df_dir: PathBuf,
    #[structopt(short = "o", long)]
    out: PathBuf,
}

impl ConvertCmd {
    pub fn run(self) -> Result<(), Error> {
        let ConvertCmd {
            soundgarden_user_df_dir,
            out,
        } = self;

        let mut storage = InMemoryData::default();

        for entry in fs::read_dir(&soundgarden_user_df_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension() != Some(OsStr::new("csv")) {
                continue;
            }

            debug!("Loading history for {:?}.", path.file_name().unwrap());
            let history = load_history(path)?;
            debug!("Processing history");
            for (inputs, target_prob_dist) in list_net_training_data_from_history(&history) {
                storage.add_sample(inputs.view(), target_prob_dist.view());
            }
        }

        debug!("Write binparams file.");
        storage.write_to_file(out)?;

        Ok(())
    }
}

//FIXME[follow up PR]: Change ltr feature extraction tests to also use this (or at least reuse some code and use the soundgarden user df csv data format)
fn load_history(path: impl AsRef<Path>) -> Result<Vec<DocumentHistory>, Error> {
    let mut reader = csv::Reader::from_path(path)?;

    let mut history = Vec::new();
    for (counter, record) in reader.deserialize().enumerate() {
        let record: SoundgardenUserDfRecord = record?;
        history.push(
            record.into_document_history_with_bad_user_action(
                counter
                    .try_into()
                    .expect("User with more then 2^32-1 records."),
            ),
        );
    }

    fix_user_actions_in_history(&mut history);

    Ok(history)
}

#[derive(Deserialize)]
struct SoundgardenUserDfRecord {
    session_id: usize,
    query_id: usize,
    user_id: usize,
    day: usize,
    query_words: String,
    url: String,
    domain: String,
    relevance: Relevance,
    position: usize,
    query_counter: usize,
}

impl SoundgardenUserDfRecord {
    fn into_document_history_with_bad_user_action(
        self,
        per_user_query_counter: u32,
    ) -> DocumentHistory {
        let Self {
            session_id,
            query_id,
            user_id,
            day,
            query_words,
            url,
            domain,
            relevance,
            position,
            query_counter,
        } = self;

        DocumentHistory {
            id: DocumentId(id2uuid(user_id, Some(per_user_query_counter))),
            relevance,
            user_feedback: UserFeedback::NotGiven,
            session: SessionId(id2uuid(session_id, None)),
            query_count: query_counter,
            query_id: QueryId(id2uuid(query_id, None)),
            query_words,
            day: day_from_day_offset(day),
            url,
            domain,
            rank: position.checked_sub(1).unwrap(),
            user_action: UserAction::Click,
        }
    }
}

fn fix_user_actions_in_history(histories: &mut [DocumentHistory]) -> bool {
    // Soundgarden User Df are already grouped query in order past to present.
    // FIXME error/sort if they are not.
    histories
        .iter_mut()
        .rev()
        .group_by(|d| d.query_id)
        .into_iter()
        .fold(false, |has_clicks, (_, query)| {
            has_clicks | fix_user_actions_in_query_reverse_order(query)
        })
}

fn fix_user_actions_in_query_reverse_order<'a>(
    query: impl IntoIterator<Item = &'a mut DocumentHistory>,
) -> bool {
    let mut has_clicked_docs_after_it = false;
    for doc in query.into_iter() {
        doc.user_action = match doc.relevance {
            Relevance::High | Relevance::Medium => {
                has_clicked_docs_after_it = true;
                UserAction::Click
            }
            Relevance::Low => {
                if has_clicked_docs_after_it {
                    UserAction::Skip
                } else {
                    UserAction::Miss
                }
            }
        }
    }

    has_clicked_docs_after_it
}

/// Crate a DayOfWeek from the offset.
fn day_from_day_offset(day: usize) -> DayOfWeek {
    use DayOfWeek::*;
    static DAYS: &[DayOfWeek] = &[Mon, Tue, Wed, Thu, Fri, Sat, Sun];
    DAYS[day % 7]
}

/// Creates a UUID by combining `d006a685-eb92-4d36-XXXX-XXXXXXXXXXXX` with given `sub_id`(S).
fn id2uuid(sub_id: usize, sub_id2: Option<u32>) -> Uuid {
    const BASE_UUID: u128 = 0x00000000_eb92_4d36_0000_000000000000;
    const DEFAULT_SUB_ID_2: u32 = 0xd006a685;
    let sub_id2 = sub_id2.unwrap_or(DEFAULT_SUB_ID_2);
    let sub_id2 = (sub_id2 as u128) << 96;
    uuid::Uuid::from_u128(sub_id2 | BASE_UUID | (sub_id as u128))
}
