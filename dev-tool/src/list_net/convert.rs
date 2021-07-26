#![cfg(not(tarpaulin))]

use std::{
    convert::TryInto,
    ffi::OsStr,
    fs,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use anyhow::{bail, Context, Error};
use indicatif::{ProgressBar, ProgressStyle};
use itertools::Itertools;
use log::trace;
use rayon::iter::{ParallelBridge, ParallelIterator};
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
use crate::{exit_code::NO_ERROR, utils::progress_spin_until_done};

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

        // Estimate Number of samples based on the number of users.
        let nr_users = progress_spin_until_done("Estimating number of samples", || {
            list_csv_files(&from_soundgarden)?.fold(
                Ok(0),
                |acc, entry_res| -> Result<usize, Error> {
                    entry_res?;
                    acc.map(|acc| acc + 1)
                },
            )
        })?;

        let storage = Arc::new(Mutex::new(InMemorySamples::with_sample_capacity(nr_users)));

        let progress_bar = ProgressBar::new(nr_users.try_into().unwrap()).with_style(
            ProgressStyle::default_bar()
                .template("Creating Storage: [{bar:43.green}] {percent:>3}% ({pos:>5}/{len:>5})")
                .progress_chars("=> "),
        );
        let bar = progress_bar.clone();
        list_csv_files(&from_soundgarden)?
            .par_bridge()
            .try_for_each_with(storage.clone(), move |storage, res| {
                res.map_err(Error::from).and_then(|path| {
                    trace!("Processing {:?}.", path.file_name().unwrap());
                    let history = load_history(&path)?;
                    let new_samples =
                        list_net_training_data_from_history(&history).with_context(|| {
                            format!(
                                "Invariants in soundgarden user-df broken for input file: {}",
                                path.display()
                            )
                        })?;
                    let new_samples = InMemorySamples::prepare_samples(new_samples)?;
                    let mut storage = storage.lock().unwrap();
                    storage.add_prepared_samples(new_samples);
                    drop(storage);
                    bar.inc(1);
                    Ok(())
                })
            })?;

        progress_bar.finish();

        let storage = storage
            .try_lock()
            .expect("Lock was leaked by rayon or poisoned.");

        progress_spin_until_done("Writing storage to disk", || {
            storage.serialize_into_file(to_samples)
        })?;
        Ok(NO_ERROR)
    }
}

fn the_first_error_has_not_been_returned_predicate<T, E>() -> impl FnMut(&Result<T, E>) -> bool {
    let mut exit_next = false;
    move |val| {
        if exit_next {
            false
        } else {
            exit_next = val.is_err();
            true
        }
    }
}

fn list_csv_files(
    dir: impl AsRef<Path>,
) -> Result<impl Iterator<Item = Result<PathBuf, Error>>, Error> {
    let err_msg = "Reading dir with user data-frames failed.";
    let iter = fs::read_dir(&dir)
        .context(err_msg)?
        .map(move |entry| {
            let entry = entry?;
            let file_type = entry.file_type()?;
            let path = entry.path();
            if path.extension() == Some(OsStr::new("csv")) {
                if file_type.is_file() {
                    Ok(Some(path))
                } else {
                    bail!(
                        "Dir entry with .csv ending is not a file: {} (type={:?})",
                        path.display(),
                        file_type
                    );
                }
            } else {
                Ok(None)
            }
        })
        .flat_map(|res| res.transpose())
        .take_while(the_first_error_has_not_been_returned_predicate());

    Ok(iter)
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
    /// Session Id, is an incremental unsigned integer in soundgarden.
    session_id: u64,
    /// Query Id, same for all queries using the same parameters.
    query_id: u64,
    /// User Id, arbitrary unsigned integer.
    ///
    /// We treat all data as if it's from the same user, so we
    /// don't really need it but we use it to infer an appropriate
    /// deterministic document id.
    user_id: u64,
    /// Number of days since the start of the soundgarden history.
    ///
    /// Starting with `Tue == 1`.
    ///
    /// As such `Mon` is `0`, `Wed` is `2` etc.
    ///
    /// The number is not limited to `0..=6`, e.g. all of `1`,`8`,`15` are `Tue`
    day: usize,
    /// The words of the query as *comma* separated string.
    ///
    /// This are not actual words but string encoded word ids.
    /// E.g. `"2142,53423"` but treating `2142` as a word is not
    /// a problem for our use-case.
    query_words: String,
    /// The url of the document.
    ///
    /// Theoretically this is also an "arbitrary" unsigned integer id, but
    /// practically it's simpler to just treat it as a string.
    url: String,
    /// The domain of the document.
    ///
    /// Theoretically this is also an "arbitrary" unsigned integer id, but
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
    /// as the other documents need to be known to
    /// infer the [`UserAction`] correctly.
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
            query_words: query_words.replace(",", " "),
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
    // Soundgarden User Df are already grouped by query and sorted, ordered from past to present.
    // (There is a FIXME for this above.)
    histories
        .iter_mut()
        .rev()
        .group_by(|d| d.query_id)
        .into_iter()
        .for_each(|(_, query)| fix_user_actions_in_query_reverse_order(query))
}

/// Function Split out from [`fix_user_actions_in_history`] **do not reuse elsewhere**.
///
/// The iterator must iterate over queries from last to first.
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

/// Crate a [`DayOfWeek`] instance from an offset.
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
fn id2uuid(sub_id: u64, sub_id2: Option<u32>) -> Uuid {
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
            id2uuid(u64::MAX, Some(u32::MAX)),
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
            query_words: "123 432".to_owned(),
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
                query_words: "foo bar".to_owned(),
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
                query_words: "foo bar".to_owned(),
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
                query_words: "foo bar".to_owned(),
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
                query_words: "foo bar".to_owned(),
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
                query_words: "foo bar".to_owned(),
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
                query_words: "bar bar".to_owned(),
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
                query_words: "bar bar".to_owned(),
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
                query_words: "foo bar".to_owned(),
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
                query_words: "foo bar".to_owned(),
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
                query_words: "foo bar".to_owned(),
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
                query_words: "foo bar".to_owned(),
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
