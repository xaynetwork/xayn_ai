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

use super::data_source::InMemorySamples;
use crate::exit_code::NO_ERROR;

/// Converts different file formats.
///
/// This is mainly used to prepare a samples file
/// based on a directory containing soundgarden user
/// data-frame csv files.
#[derive(Debug, StructOpt)]
pub struct ConvertCmd {
    /// Directory in which soundgarden saved user data-frames (one csv file per user).
    #[structopt(short = "d", long)]
    from_soundgarden: PathBuf,
    /// File in which the samples should be stored, e.g. `data.samples`.
    #[structopt(short = "o", long)]
    to_samples: PathBuf,
}

impl ConvertCmd {
    pub fn run(self) -> Result<i32, Error> {
        let ConvertCmd {
            from_soundgarden,
            to_samples,
        } = self;

        let mut storage = InMemorySamples::default();

        for entry in fs::read_dir(&from_soundgarden)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension() != Some(OsStr::new("csv")) {
                continue;
            }

            debug!("Loading history for {:?}.", path.file_name().unwrap());
            let history = load_history(path)?;
            debug!("Processing history");
            for (inputs, target_prob_dist) in list_net_training_data_from_history(&history)? {
                storage.add_sample(inputs.view(), target_prob_dist.view())?;
            }
        }

        debug!("Write output file.");
        storage.write_to_file(to_samples)?;

        Ok(NO_ERROR)
    }
}

//FIXME[follow up PR]: Change ltr feature extraction tests to share code with
//                     this implementation and then to use the normal soundgarden
//                     user data-frames instead of specially pre-processed csv
//                     files.
//
//FIXME maybe verify the structural integrity of the file, currently it's expected
// to have the records partitioned by query as well as sorted both in each query
// (first ranked record to last ranked record) and in between queries (oldest
// query to newest query).
//
/// Loads a history from a soundgarden user data-frame csv file.
///
/// The [`UserAction`] values will be inferred from the data in the
/// same way soundgarden does so.
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

/// Struct to load a soundgarden user data-frame csv file record into.
#[derive(Deserialize)]
struct SoundgardenUserDfRecord {
    /// Session Id, is a incremental usize in soundgarden.
    session_id: usize,
    /// Query Id, same for all queries using the same parameter.
    query_id: usize,
    /// User Id, arbitrary usize.
    ///
    /// We treat all data as if it's from the same user, so we
    /// don't really need it but we use it to infer a appropriate
    /// deterministic document id.
    user_id: usize,
    /// Day since the start of the training but starting as `Tue == 1`.
    ///
    /// As such `Mon` is `0`, `Wed` is `2` etc.
    ///
    /// The number is not limited to `0..=6`, e.g. all of `1`,`8`,`15` are `Tue`
    day: usize,
    /// The words of the query as *comma* separated string.
    ///
    /// This are not actual words but word id's as string encoded,
    /// e.g. `"2142,53423"` but treating `2142` as a word is not
    /// a problem for our use-case.
    query_words: String,
    /// The url of the document.
    ///
    /// Theoretically this is also an "arbitrary" usize id, but
    /// practically it's simpler to just treat it as a string.
    url: String,
    /// The domain of the document.
    ///
    /// Theoretically this is also an "arbitrary" usize id, but
    /// practically it's simpler to just treat it as a string.
    domain: String,
    /// The relevance of the document.
    relevance: Relevance,
    /// The position of the document, *starting with `1`*.
    position: usize,
    /// The index of the query in the session it belongs to.
    query_counter: usize,
}

impl SoundgardenUserDfRecord {
    /// Creates a [`DocumentHistory`] from this record.
    ///
    /// The `user_action` field *will* be incorrect and
    /// needs to be fixed separately. This is the case
    /// as the following documents need to be known to
    /// infer it correctly.
    fn into_document_history_with_bad_user_action(
        self,
        per_user_result_counter: u32,
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
            // By combining the `user_id` with a monotonic increasing per-user counter
            // we can create deterministically an appropriate document id.
            id: DocumentId(id2uuid(user_id, Some(per_user_result_counter))),
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
            user_action: UserAction::Miss,
        }
    }
}

/// Updates the history with user actions inferred from the context.
///
/// This uses the same approach as in soundgarden.
fn fix_user_actions_in_history(histories: &mut [DocumentHistory]) {
    // Soundgarden User Df are already grouped query in order past to present.
    // (There is a FIXME for this above.)
    histories
        .iter_mut()
        .rev()
        .group_by(|d| d.query_id)
        .into_iter()
        .for_each(|(_, query)| fix_user_actions_in_query_reverse_order(query))
}

/// Split out from [`fix_user_actions_in_history`] **do not reuse elsewhere**.
///
/// The iterator must iterate over queries from last to first, and fixes
/// user actions.
fn fix_user_actions_in_query_reverse_order<'a>(
    query: impl IntoIterator<Item = &'a mut DocumentHistory>,
) {
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
}

/// Crate a DayOfWeek from the offset.
fn day_from_day_offset(day: usize) -> DayOfWeek {
    use DayOfWeek::*;
    static DAYS: &[DayOfWeek] = &[Mon, Tue, Wed, Thu, Fri, Sat, Sun];
    DAYS[day % 7]
}

/// Creates an UUID by combining `YYYYYYYY-eb92-4d36-XXXX-XXXXXXXXXXXX` with given `sub_id`(s).
///
/// - `X..` is replaced with the `sub_id`.
///
/// - `Y..` is replaced with the `sub_id2` if given. If
///       not given `0xd006a685` is used instead.
fn id2uuid(sub_id: usize, sub_id2: Option<u32>) -> Uuid {
    const BASE_UUID: u128 = 0x00000000_eb92_4d36_0000_000000000000;
    const DEFAULT_SUB_ID_2: u32 = 0xd006a685;
    let sub_id2 = sub_id2.unwrap_or(DEFAULT_SUB_ID_2);
    let sub_id2 = (sub_id2 as u128) << 96;
    uuid::Uuid::from_u128(sub_id2 | BASE_UUID | (sub_id as u128))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_id2uuid() {
        assert_eq!(
            id2uuid(0, Some(0)),
            Uuid::from_u128(0x00000000_eb92_4d36_0000_000000000000)
        );
        assert_eq!(
            id2uuid(0, None),
            Uuid::from_u128(0xd006a685_eb92_4d36_0000_000000000000)
        );
        assert_eq!(
            id2uuid(usize::MAX, Some(u32::MAX)),
            Uuid::from_u128(0xffffffff_eb92_4d36_ffff_ffffffffffff)
        )
    }

    #[test]
    fn test_day_from_offset() {
        assert_eq!(day_from_day_offset(0), DayOfWeek::Mon);
        assert_eq!(day_from_day_offset(1), DayOfWeek::Tue);
        assert_eq!(day_from_day_offset(2), DayOfWeek::Wed);
        assert_eq!(day_from_day_offset(3), DayOfWeek::Thu);
        assert_eq!(day_from_day_offset(4), DayOfWeek::Fri);
        assert_eq!(day_from_day_offset(5), DayOfWeek::Sat);
        assert_eq!(day_from_day_offset(6), DayOfWeek::Sun);
        assert_eq!(day_from_day_offset(7), DayOfWeek::Mon);
        assert_eq!(day_from_day_offset(8), DayOfWeek::Tue);
    }

    #[test]
    fn test_soundgarden_record_to_document_history() {
        let record = SoundgardenUserDfRecord {
            session_id: 10,
            query_id: 23,
            user_id: 142313,
            day: 17,
            query_words: "123,432".to_owned(),
            url: "32".to_owned(),
            domain: "23".to_owned(),
            relevance: Relevance::Medium,
            position: 4,
            query_counter: 12,
        };
        let expected = DocumentHistory {
            id: DocumentId(id2uuid(142313, Some(u32::MAX))),
            relevance: Relevance::Medium,
            user_feedback: UserFeedback::NotGiven,
            session: SessionId(id2uuid(10, None)),
            query_count: 12,
            query_id: QueryId(id2uuid(23, None)),
            query_words: "123,432".to_owned(),
            day: DayOfWeek::Thu,
            url: "32".to_owned(),
            domain: "23".to_owned(),
            rank: 3,
            user_action: UserAction::Miss,
        };

        let history = record.into_document_history_with_bad_user_action(u32::MAX);

        assert_eq!(history, expected);
    }

    #[test]
    fn test_fix_user_action_in_history() {
        let mut history = vec![
            DocumentHistory {
                id: DocumentId(id2uuid(12, Some(0))),
                relevance: Relevance::Low,
                user_feedback: UserFeedback::NotGiven,
                session: SessionId(id2uuid(3441, None)),
                query_count: 0,
                query_id: QueryId(id2uuid(42, None)),
                query_words: "foo,bar".to_owned(),
                day: DayOfWeek::Wed,
                url: "foo/bar/a".to_owned(),
                domain: "bar/foo".to_owned(),
                rank: 0,
                user_action: UserAction::Miss,
            },
            DocumentHistory {
                id: DocumentId(id2uuid(12, Some(1))),
                relevance: Relevance::Medium,
                user_feedback: UserFeedback::NotGiven,
                session: SessionId(id2uuid(3441, None)),
                query_count: 0,
                query_id: QueryId(id2uuid(42, None)),
                query_words: "foo,bar".to_owned(),
                day: DayOfWeek::Wed,
                url: "foo/bar/b".to_owned(),
                domain: "bar/foo".to_owned(),
                rank: 1,
                user_action: UserAction::Miss,
            },
            DocumentHistory {
                id: DocumentId(id2uuid(12, Some(2))),
                relevance: Relevance::Low,
                user_feedback: UserFeedback::NotGiven,
                session: SessionId(id2uuid(3441, None)),
                query_count: 0,
                query_id: QueryId(id2uuid(42, None)),
                query_words: "foo,bar".to_owned(),
                day: DayOfWeek::Wed,
                url: "foo/bar".to_owned(),
                domain: "bar/foo/t".to_owned(),
                rank: 2,
                user_action: UserAction::Miss,
            },
            DocumentHistory {
                id: DocumentId(id2uuid(12, Some(3))),
                relevance: Relevance::High,
                user_feedback: UserFeedback::NotGiven,
                session: SessionId(id2uuid(3441, None)),
                query_count: 0,
                query_id: QueryId(id2uuid(42, None)),
                query_words: "foo,bar".to_owned(),
                day: DayOfWeek::Wed,
                url: "foo/bar".to_owned(),
                domain: "bar-foo/fu".to_owned(),
                rank: 3,
                user_action: UserAction::Miss,
            },
            DocumentHistory {
                id: DocumentId(id2uuid(12, Some(4))),
                relevance: Relevance::Low,
                user_feedback: UserFeedback::NotGiven,
                session: SessionId(id2uuid(3441, None)),
                query_count: 0,
                query_id: QueryId(id2uuid(42, None)),
                query_words: "foo,bar".to_owned(),
                day: DayOfWeek::Wed,
                url: "bar/bar".to_owned(),
                domain: "bar".to_owned(),
                rank: 4,
                user_action: UserAction::Miss,
            },
            /* ---- */
            DocumentHistory {
                id: DocumentId(id2uuid(12, Some(5))),
                relevance: Relevance::Low,
                user_feedback: UserFeedback::NotGiven,
                session: SessionId(id2uuid(3441, None)),
                query_count: 1,
                query_id: QueryId(id2uuid(52, None)),
                query_words: "bar,bar".to_owned(),
                day: DayOfWeek::Wed,
                url: "foo/bar".to_owned(),
                domain: "bar/foo".to_owned(),
                rank: 0,
                user_action: UserAction::Miss,
            },
            DocumentHistory {
                id: DocumentId(id2uuid(12, Some(6))),
                relevance: Relevance::Low,
                user_feedback: UserFeedback::NotGiven,
                session: SessionId(id2uuid(3441, None)),
                query_count: 1,
                query_id: QueryId(id2uuid(52, None)),
                query_words: "bar,bar".to_owned(),
                day: DayOfWeek::Wed,
                url: "foobar".to_owned(),
                domain: "barfoo".to_owned(),
                rank: 1,
                user_action: UserAction::Miss,
            },
            /* ---- */
            DocumentHistory {
                id: DocumentId(id2uuid(12, Some(7))),
                relevance: Relevance::High,
                user_feedback: UserFeedback::NotGiven,
                session: SessionId(id2uuid(3441, None)),
                query_count: 2,
                query_id: QueryId(id2uuid(42, None)),
                query_words: "foo,bar".to_owned(),
                day: DayOfWeek::Wed,
                url: "foo/bar".to_owned(),
                domain: "bar/foo".to_owned(),
                rank: 0,
                user_action: UserAction::Miss,
            },
            DocumentHistory {
                id: DocumentId(id2uuid(12, Some(8))),
                relevance: Relevance::Low,
                user_feedback: UserFeedback::NotGiven,
                session: SessionId(id2uuid(3441, None)),
                query_count: 2,
                query_id: QueryId(id2uuid(42, None)),
                query_words: "foo,bar".to_owned(),
                day: DayOfWeek::Wed,
                url: "foo/bar".to_owned(),
                domain: "bar/foo".to_owned(),
                rank: 1,
                user_action: UserAction::Miss,
            },
            DocumentHistory {
                id: DocumentId(id2uuid(12, Some(9))),
                relevance: Relevance::Low,
                user_feedback: UserFeedback::NotGiven,
                session: SessionId(id2uuid(3441, None)),
                query_count: 2,
                query_id: QueryId(id2uuid(42, None)),
                query_words: "foo,bar".to_owned(),
                day: DayOfWeek::Wed,
                url: "foo/bar".to_owned(),
                domain: "bar/foo".to_owned(),
                rank: 2,
                user_action: UserAction::Miss,
            },
            /* ---- */
            DocumentHistory {
                id: DocumentId(id2uuid(12, Some(10))),
                relevance: Relevance::Low,
                user_feedback: UserFeedback::NotGiven,
                session: SessionId(id2uuid(3441, None)),
                query_count: 3,
                query_id: QueryId(id2uuid(42, None)),
                query_words: "foo,bar".to_owned(),
                day: DayOfWeek::Wed,
                url: "foo/bar".to_owned(),
                domain: "bar/foo".to_owned(),
                rank: 0,
                user_action: UserAction::Miss,
            },
        ];
        let fixed_history = &[
            doc_with_action(&history[0], UserAction::Skip),
            doc_with_action(&history[1], UserAction::Click),
            doc_with_action(&history[2], UserAction::Skip),
            doc_with_action(&history[3], UserAction::Click),
            doc_with_action(&history[4], UserAction::Miss),
            /* ---- */
            doc_with_action(&history[5], UserAction::Miss),
            doc_with_action(&history[6], UserAction::Miss),
            /* ---- */
            doc_with_action(&history[7], UserAction::Click),
            doc_with_action(&history[8], UserAction::Miss),
            doc_with_action(&history[9], UserAction::Miss),
            doc_with_action(&history[10], UserAction::Miss),
        ];

        assert_eq!(history.len(), fixed_history.len());

        fix_user_actions_in_history(&mut history);

        assert_eq!(history, fixed_history);

        fn doc_with_action(doc: &DocumentHistory, action: UserAction) -> DocumentHistory {
            let mut doc = doc.clone();
            doc.user_action = action;
            doc
        }
    }
}
