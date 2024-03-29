#![cfg(not(tarpaulin))]

use std::{
    cmp::min,
    collections::{HashMap, HashSet},
    fmt,
    fs::File,
    io::BufReader,
    path::{Path, PathBuf},
};

use anyhow::{bail, Error};
use rand::{
    prelude::{IteratorRandom, SliceRandom, ThreadRng},
    thread_rng,
    Rng,
};
use serde::Deserialize;
use structopt::StructOpt;
use uuid::Uuid;
use xayn_ai::{
    DayOfWeek,
    Document,
    DocumentHistory,
    DocumentId,
    QueryId,
    Relevance,
    RerankMode,
    SessionId,
    UserAction,
    UserFeedback,
};

use crate::exit_code::NO_ERROR;

use super::CallData;

/// Generate call data for the example.
#[derive(StructOpt, Debug)]
pub struct GenerateCallDataCmd {
    /// Json file to write the generate call data to.
    #[structopt(short, long)]
    out: PathBuf,

    /// File with snippets to use.
    #[structopt(long, default_value = "data/snippets_for_example_data_generation.json")]
    snippets: PathBuf,

    /// Number of documents in current query.
    ///
    /// I.e. the number of documents the result of query (we
    /// pretend to run) in the example app has.
    #[structopt(short = "q", long, default_value = "30")]
    number_of_current_documents: usize,

    /// Number of queries in the users history.
    ///
    /// Each query will randomly have between 5 and 30 documents.
    #[structopt(short = "h", long)]
    number_of_historic_queries: usize,

    /// Extend domains and urls.
    ///
    /// Normally urls will be generated using following
    /// schema: `url://dom_{domain_id}/{url_id}`.
    ///
    /// This uses more or less incremental ids but this
    /// means that domains and urls are unrealistic short.
    ///
    /// Using this option will make the domains and URLs
    /// longer. You can use it multiple times.
    #[structopt(short = "l", parse(from_occurrences))]
    lengthen_urls: usize,
}

const MIN_QUERY_NR_DOCS: usize = 5;
const MAX_QUERY_NR_DOCS: usize = 30;

impl GenerateCallDataCmd {
    pub fn run(self) -> Result<i32, Error> {
        let GenerateCallDataCmd {
            out,
            snippets,
            number_of_current_documents,
            number_of_historic_queries,
            lengthen_urls,
        } = self;

        if number_of_current_documents < 1 {
            bail!("The number_of_current_documents argument needs to be greater than 0.");
        }

        if number_of_historic_queries < 1 {
            bail!(
                "At least 1 historic query is needed as the current documents are based on the last historic query."
            );
        }

        let snippets = parse_snippets(snippets)?;
        let mut gen_state = GenState::new(snippets, lengthen_urls);
        let histories = gen_histories(
            number_of_historic_queries,
            number_of_current_documents,
            &mut gen_state,
        );
        let start_of_last_query = histories.len() - number_of_current_documents;
        let documents =
            gen_current_query_from_history(&histories[start_of_last_query..], &mut gen_state);

        let call_data = CallData {
            rerank_mode: RerankMode::PersonalizedSearch,
            histories,
            documents,
            serialized_state: None,
        };
        call_data.save_to_file(&out)?;
        Ok(NO_ERROR)
    }
}

#[derive(Deserialize)]
struct Snippet {
    title: String,
    snippet: String,
}

type Snippets = HashMap<String, Vec<Snippet>>;

fn parse_snippets(snippets_file: impl AsRef<Path>) -> Result<Snippets, Error> {
    let reader = BufReader::new(File::open(snippets_file)?);
    serde_json::from_reader(reader).map_err(Into::into)
}
struct QueryWordsAndSnippets {
    query_words: String,
    snippets: Vec<Snippet>,
}

/// "Fake" domain.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Domain {
    id: usize,
    lengthen_urls: usize,
}

impl Domain {
    /// Returns a new Domain new-type with a different id.
    fn with_id(&self, id: usize) -> Self {
        Self {
            id,
            lengthen_urls: self.lengthen_urls,
        }
    }
}

impl fmt::Display for Domain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // We do not want to accidentally generate invalid domains.
        // (And we care more about longer urls then domains, with 4
        // domains are 20+ characters long, which is enough).
        for _ in 0..min(self.lengthen_urls, 4) {
            write!(f, "foo")?;
        }
        write!(f, "dom_{}.test", self.id)
    }
}

/// "Fake" url.
struct Url(Domain, usize);

impl fmt::Display for Url {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "url://{}/{}", self.0, self.1)?;
        for _ in 0..self.0.lengthen_urls {
            write!(f, "/foobarbaz")?;
        }
        Ok(())
    }
}

struct GenState {
    rng: ThreadRng,
    queries: HashMap<QueryId, QueryWordsAndSnippets>,
    used_urls_in_query: HashMap<Domain, usize>,
    next_unused_domain: Domain,
    used_domains_in_query: HashSet<Domain>,
    last_day: DayOfWeek,
}

impl GenState {
    fn new(snippets: Snippets, lengthen_urls: usize) -> Self {
        let queries = snippets
            .into_iter()
            .map(|(query_words, snippets)| {
                (
                    QueryId(Uuid::new_v4()),
                    QueryWordsAndSnippets {
                        query_words,
                        snippets,
                    },
                )
            })
            .collect();

        let mut rng = thread_rng();
        let last_day = DayOfWeek::from_day_offset(rng.gen_range(0..7));
        Self {
            rng,
            queries,
            used_urls_in_query: HashMap::new(),
            next_unused_domain: Domain {
                id: 0,
                lengthen_urls,
            },
            used_domains_in_query: Default::default(),
            last_day,
        }
    }

    fn clear_query_state(&mut self) {
        self.used_urls_in_query.clear();
        self.used_domains_in_query.clear();
    }

    fn pick_query(&mut self) -> (QueryId, String) {
        self.queries
            .iter()
            .choose(&mut self.rng)
            .map(|(id, words_and_snippets)| (*id, words_and_snippets.query_words.clone()))
            .expect("snippets for at least one query to exists")
    }

    fn pick_title_and_snippet(&mut self, query: QueryId) -> (String, String) {
        self.queries[&query]
            .snippets
            .choose(&mut self.rng)
            .map(|snippet| (snippet.title.clone(), snippet.snippet.clone()))
            .expect("snippets for given query exists")
    }

    /// Picks a domain.
    ///
    /// This remembers that given domain was picked in the current query.
    ///
    /// - With a 20% chance pick (if possible) a domain which already appeared in this query.
    /// - Else with a 30%  chance pick (if possible) a domain which already appeared in *any* query.
    /// - Else pick a domain not yet used anywhere
    fn pick_domain_in_query(&mut self) -> Domain {
        let domain = self.pick_domain();
        self.used_domains_in_query.insert(domain);
        domain
    }

    fn pick_domain(&mut self) -> Domain {
        if self.rng.gen_bool(0.2) {
            if let Some(domain) = self.pick_in_query_used_domain() {
                return domain;
            }
        }

        if self.rng.gen_bool(0.3) {
            if let Some(domain) = self.pick_used_domain() {
                return domain;
            }
        }

        self.new_unused_domain()
    }

    fn new_unused_domain(&mut self) -> Domain {
        let domain = self.next_unused_domain;
        self.next_unused_domain = domain.with_id(domain.id + 1);
        domain
    }

    fn pick_used_domain(&mut self) -> Option<Domain> {
        let next_unused_domain = self.next_unused_domain;
        let id = next_unused_domain.id;
        (id != 0).then(|| next_unused_domain.with_id(self.rng.gen_range(0..id)))
    }

    fn pick_in_query_used_domain(&mut self) -> Option<Domain> {
        self.used_domains_in_query
            .iter()
            .choose(&mut self.rng)
            .copied()
    }

    /// Picks a url using the given domain.
    ///
    /// This makes sure no url appears twice in the same query, but
    /// makes it likely that the url appeared in a different query *iff*
    /// the domain appeared in a different query.
    fn pick_url_in_query(&mut self, domain: Domain) -> Url {
        let count_ref = self.used_urls_in_query.entry(domain).or_insert(0);
        let next_url = Url(domain, *count_ref);
        *count_ref += 1;
        next_url
    }

    fn gen_session(&mut self) -> SessionId {
        SessionId(Uuid::new_v4())
    }

    fn pick_number_of_documents_in_query(&mut self) -> usize {
        self.rng.gen_range(MIN_QUERY_NR_DOCS..=MAX_QUERY_NR_DOCS)
    }

    /// Picks the day for the new query.
    ///
    /// If it's a new session:
    ///
    /// - With 20% chance it's still the same day.
    /// - Else with 60% chance it's the next day.
    /// - Else it's a fully random day.
    ///
    /// If it's not a new session:
    ///
    /// - With 80% chance it's still the same day.
    /// - Else with 90% chance it's the next day.
    /// - Else it's a fully random day.
    fn pick_day_relative_to_last_pick(&mut self, new_session: bool) -> DayOfWeek {
        let same_day_prob = if new_session { 0.2 } else { 0.8 };
        if self.rng.gen_bool(same_day_prob) {
            return self.last_day;
        }

        let next_day_prob = if new_session { 0.6 } else { 0.9 };

        self.last_day = if self.rng.gen_bool(next_day_prob) {
            DayOfWeek::from_day_offset(self.last_day as usize + 1)
        } else {
            DayOfWeek::from_day_offset(self.rng.gen_range(0..7))
        };

        self.last_day
    }

    /// Picks which `UserFeedback` was given.
    ///
    /// - With 50% chance no feedback was given.
    /// - With 30% chance negative feedback was given.
    /// - With 20% chance positive feedback was given.
    fn pick_user_feedback(&mut self) -> UserFeedback {
        [
            (5, UserFeedback::NotGiven),
            (3, UserFeedback::Irrelevant),
            (2, UserFeedback::NotGiven),
        ]
        .choose_weighted(&mut self.rng, |v| v.0)
        .unwrap()
        .1
    }
}

/// We generate the current query from the history.
///
/// This is needed as the examples doesn't update the history and
/// in turn wouldn't have a feedback loop otherwise. This also
/// means it would not have or get any CoIs, which would make it
/// unsuited for testing and benchmarking.
fn gen_current_query_from_history(
    histories: &[DocumentHistory],
    gen_state: &mut GenState,
) -> Vec<Document> {
    gen_state.clear_query_state();

    histories
        .iter()
        .map(|hist_doc| {
            let DocumentHistory {
                id,
                session,
                query_count,
                query_id,
                query_words,
                url,
                domain,
                rank,
                ..
            } = hist_doc;

            let (title, snippet) = gen_state.pick_title_and_snippet(*query_id);
            Document {
                id: *id,
                rank: *rank,
                title,
                snippet,
                session: *session,
                query_count: *query_count,
                query_id: *query_id,
                query_words: query_words.clone(),
                url: url.clone(),
                domain: domain.clone(),
            }
        })
        .collect()
}

fn gen_histories(
    number_of_queries: usize,
    number_of_documents_in_last_query: usize,
    gen_state: &mut GenState,
) -> Vec<DocumentHistory> {
    let mut histories = Vec::new();
    let mut current_session = gen_state.gen_session();
    let mut query_count = 0;

    for total_id in 0..number_of_queries {
        if gen_state.rng.gen_bool(0.15) {
            current_session = gen_state.gen_session();
            query_count = 0;
        }

        let number_of_documents = if total_id == number_of_queries - 1 {
            number_of_documents_in_last_query
        } else {
            gen_state.pick_number_of_documents_in_query()
        };

        gen_historic_query(
            current_session,
            query_count,
            number_of_documents,
            &mut histories,
            gen_state,
        );
        query_count += 1;
    }

    histories
}

fn gen_historic_query(
    session: SessionId,
    query_count: usize,
    number_of_documents: usize,
    out: &mut Vec<DocumentHistory>,
    gen_state: &mut GenState,
) {
    gen_state.clear_query_state();

    let (query_id, query_words) = gen_state.pick_query();
    let day = gen_state.pick_day_relative_to_last_pick(query_count == 0);

    let first_click_at = gen_state.rng.gen_range(0..number_of_documents * 7 / 6);
    let last_click_at = if first_click_at < number_of_documents {
        gen_state.rng.gen_range(first_click_at..number_of_documents)
    } else {
        number_of_documents
    };

    for rank in 0..number_of_documents {
        let id = DocumentId(Uuid::new_v4());
        let domain = gen_state.pick_domain_in_query();
        let url = gen_state.pick_url_in_query(domain);
        let user_feedback = gen_state.pick_user_feedback();

        let (relevance, user_action) = if rank < first_click_at {
            (Relevance::Low, UserAction::Skip)
        } else if rank == last_click_at {
            (Relevance::High, UserAction::Click)
        } else if rank == first_click_at {
            (Relevance::Medium, UserAction::Click)
        } else if rank < last_click_at {
            if gen_state.rng.gen_bool(0.4) {
                (Relevance::Medium, UserAction::Click)
            } else {
                (Relevance::Low, UserAction::Skip)
            }
        } else {
            (Relevance::Low, UserAction::Miss)
        };

        out.push(DocumentHistory {
            id,
            relevance,
            user_feedback,
            session,
            query_count,
            query_id,
            query_words: query_words.clone(),
            day,
            url: url.to_string(),
            domain: domain.to_string(),
            rank,
            user_action,
        });
    }
}
