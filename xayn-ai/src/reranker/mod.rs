pub(crate) mod database;
pub mod public;
pub(crate) mod systems;

use serde::{Deserialize, Serialize};

use crate::{
    analytics::Analytics,
    data::{
        document::{Document, DocumentHistory, DocumentsRank},
        document_data::{
            DocumentContentComponent, DocumentDataWithDocument, DocumentDataWithEmbedding,
            DocumentDataWithMab, DocumentIdComponent,
        },
        UserInterests,
    },
    error::Error,
    reranker::systems::{CoiSystemData, CommonSystems},
    to_vec_of_ref_of,
};

#[cfg(test)]
use derive_more::From;

/// Update cois from user feedback
fn learn_user_interests<CS>(
    common_systems: &CS,
    history: &[DocumentHistory],
    prev_documents: &[&dyn CoiSystemData],
    user_interests: UserInterests,
) -> Result<UserInterests, Error>
where
    CS: CommonSystems,
{
    common_systems
        .coi()
        .update_user_interests(history, &prev_documents, user_interests)
}

/// Compute and save analytics
fn collect_analytics<CS>(
    common_systems: &CS,
    history: &[DocumentHistory],
    prev_documents: &[DocumentDataWithMab],
) -> Result<Analytics, Error>
where
    CS: CommonSystems,
{
    common_systems
        .analytics()
        .compute_analytics(history, prev_documents)
}

fn make_documents_with_embedding<CS>(
    common_systems: &CS,
    documents: &[Document],
) -> Result<Vec<DocumentDataWithEmbedding>, Error>
where
    CS: CommonSystems,
{
    let documents: Vec<_> = documents
        .iter()
        .map(|document| DocumentDataWithDocument {
            document_id: DocumentIdComponent {
                id: document.id.clone(),
            },
            document_content: DocumentContentComponent {
                snippet: document.snippet.clone(),
            },
        })
        .collect();

    common_systems.bert().compute_embedding(documents)
}

fn rerank<CS>(
    common_systems: &CS,
    history: &[DocumentHistory],
    documents: &[Document],
    user_interests: UserInterests,
) -> Result<(Vec<DocumentDataWithMab>, UserInterests, DocumentsRank), Error>
where
    CS: CommonSystems,
{
    let documents = make_documents_with_embedding(common_systems, documents)?;
    let documents = common_systems
        .coi()
        .compute_coi(documents, &user_interests)?;
    let documents = common_systems.ltr().compute_ltr(history, documents)?;
    let documents = common_systems.context().compute_context(documents)?;
    let (documents, user_interests) = common_systems
        .mab()
        .compute_mab(documents, user_interests)?;

    let rank = documents
        .iter()
        .map(|document| (document.document_id.id.clone(), document.mab.rank))
        .collect();

    Ok((documents, user_interests, rank))
}

#[cfg_attr(test, derive(Clone, From, Debug, PartialEq))]
#[derive(Serialize, Deserialize)]
enum PreviousDocuments {
    Embedding(Vec<DocumentDataWithEmbedding>),
    Mab(Vec<DocumentDataWithMab>),
}

impl Default for PreviousDocuments {
    fn default() -> Self {
        Self::Embedding(Vec::new())
    }
}

impl PreviousDocuments {
    fn to_coi_system_data(&self) -> Vec<&dyn CoiSystemData> {
        match self {
            PreviousDocuments::Embedding(documents) => {
                to_vec_of_ref_of!(documents, &dyn CoiSystemData)
            }
            PreviousDocuments::Mab(documents) => {
                to_vec_of_ref_of!(documents, &dyn CoiSystemData)
            }
        }
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        match self {
            PreviousDocuments::Embedding(documents) => documents.len(),
            PreviousDocuments::Mab(documents) => documents.len(),
        }
    }
}

#[cfg_attr(test, derive(Clone, PartialEq, Debug))]
#[derive(Default, Serialize, Deserialize)]
pub(crate) struct RerankerData {
    user_interests: UserInterests,
    prev_documents: PreviousDocuments,
}

#[cfg(test)]
impl RerankerData {
    pub(crate) fn new_with_mab(
        user_interests: UserInterests,
        prev_documents: Vec<DocumentDataWithMab>,
    ) -> Self {
        let prev_documents = PreviousDocuments::Mab(prev_documents);
        Self {
            user_interests,
            prev_documents,
        }
    }
}

pub(crate) struct Reranker<CS> {
    common_systems: CS,
    data: RerankerData,
    errors: Vec<Error>,
    /// Analytics of the previous call to `rerank`.
    analytics: Option<Analytics>,
}

impl<CS> Reranker<CS>
where
    CS: CommonSystems,
{
    pub(crate) fn new(common_systems: CS) -> Result<Self, Error> {
        // load the last valid state from the database
        let data = common_systems
            .database()
            .load_data()?
            .unwrap_or_else(RerankerData::default);

        Ok(Self {
            common_systems,
            data,
            errors: Vec::new(),
            analytics: None,
        })
    }

    pub(crate) fn errors(&self) -> &Vec<Error> {
        &self.errors
    }

    /// Returns the analytics for penultimate call to `rerank`.
    /// Analytics will be provided only if the penultimate call to `rerank` was able
    /// to run the full model without error, and the correct history is passed to the
    /// last call to `rerank`.
    pub(crate) fn analytics(&self) -> &Option<Analytics> {
        &self.analytics
    }

    /// Create a byte representation of the internal state of the Reranker.
    pub(crate) fn serialize(&self) -> Result<Vec<u8>, Error> {
        self.common_systems.database().serialize_data(&self.data)
    }

    pub(crate) fn rerank(
        &mut self,
        history: &[DocumentHistory],
        documents: &[Document],
    ) -> DocumentsRank {
        // The number of errors it can contain is very limited. By using `clear` we avoid
        // re-allocating the vector on each method call.
        self.errors.clear();

        // feedback loop and analytics
        {
            let user_interests = self.data.user_interests.clone();
            let prev_documents = self.data.prev_documents.to_coi_system_data();

            match learn_user_interests(
                &self.common_systems,
                history,
                prev_documents.as_slice(),
                user_interests,
            ) {
                Ok(user_interests) => self.data.user_interests = user_interests,
                Err(e) => self.errors.push(e),
            };

            if let PreviousDocuments::Mab(ref prev_documents) = self.data.prev_documents {
                self.analytics = collect_analytics(&self.common_systems, history, &prev_documents)
                    .map_err(|e| self.errors.push(e))
                    .ok();
            }
        }

        let user_interests = self.data.user_interests.clone();

        rerank(&self.common_systems, history, documents, user_interests)
            .map(|(prev_documents, user_interests, rank)| {
                self.data.prev_documents = PreviousDocuments::Mab(prev_documents);
                self.data.user_interests = user_interests;

                rank
            })
            .unwrap_or_else(|e| {
                self.errors.push(e);

                let prev_documents = make_documents_with_embedding(&self.common_systems, documents)
                    .unwrap_or_default();
                self.data.prev_documents = PreviousDocuments::Embedding(prev_documents);

                documents
                    .iter()
                    .map(|document| (document.id.clone(), document.rank))
                    .collect()
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::{
        coi::CoiSystemError,
        data::document::{Relevance, UserFeedback},
        tests::{
            data_with_embedding, document_history, documents_from_ids, expected_rerank_unchanged,
            history_for_prev_docs, MemDb, MockCommonSystems,
        },
    };

    macro_rules! check_error {
        ($reranker: expr, $error:pat) => {
            assert!($reranker
                .errors()
                .iter()
                .any(|e| matches!(e.downcast_ref().unwrap(), $error)));
        };
    }

    mod car_interest_example {
        use super::*;
        use crate::{
            data::UserInterests,
            tests::{cois_from_words, data_with_mab, documents_from_words, mocked_bert_system},
            Document, DocumentId,
        };

        pub(super) fn reranker_data_with_mab() -> RerankerData {
            reranker_data(data_with_mab((0..10).map(|id| (id, vec![id as f32; 128]))))
        }

        pub(super) fn reranker_data(docs: impl Into<PreviousDocuments>) -> RerankerData {
            RerankerData {
                prev_documents: docs.into(),
                user_interests: UserInterests {
                    positive: cois_from_words(&["vehicle"], mocked_bert_system()),
                    ..Default::default()
                },
            }
        }

        pub(super) fn documents() -> Vec<Document> {
            documents_from_words(
                (0..6).zip(&["ship", "car", "auto", "flugzeug", "plane", "vehicle"]),
            )
        }

        pub(super) fn expected_rerank() -> DocumentsRank {
            [5, 3, 4, 0, 2, 1]
                .iter()
                .zip(0..6)
                .map(|(id, rank)| (DocumentId(id.to_string()), rank))
                .collect()
        }
    }

    /// A user performs the very first search that returns no results/`Documents`.
    /// In this case, the `Reranker` should return an empty `DocumentsRank`.
    #[test]
    fn test_first_search_without_search_results() {
        let cs = MockCommonSystems::default();
        let mut reranker = Reranker::new(cs).unwrap();

        let rank = reranker.rerank(&[], &[]);

        assert_eq!(rank, []);
    }

    /// A user performs the very first search that returns results/`Document`s.
    /// The `Reranker` is not yet aware of any user interests and can therefore
    /// not perform any reranking. In this case, the `Reranker` should
    /// return the results/`Document`s in an unchanged order. Furthermore, the
    /// `Reranker` should create `DocumentDataWithEmbedding` from the `Document`s,
    /// from which the first user interests will be learned in the next call
    /// to `rerank`.
    #[test]
    fn test_first_search_with_search_results() {
        let cs = MockCommonSystems::default();
        let mut reranker = Reranker::new(cs).unwrap();
        let documents = documents_from_ids(0..10);

        let rank = reranker.rerank(&[], &documents);

        assert_eq!(rank, expected_rerank_unchanged(&documents));
        assert_eq!(reranker.data.prev_documents.len(), 10);
        assert!(reranker.data.user_interests.positive.is_empty());
        assert!(reranker.data.user_interests.negative.is_empty());

        check_error!(reranker, CoiSystemError::NoCoi);
    }

    /// A user performed the very first search. The `Reranker` created the
    /// previous documents from that search. The next time the user
    /// searches, the user interests should be learned from the previous
    /// documents and the current results/`Document`s should be reranked
    /// based on the newly learned user interests.
    #[test]
    fn test_first_and_second_search_learn_cois_and_rerank() {
        let cs = MockCommonSystems::default();
        let mut reranker = Reranker::new(cs).unwrap();
        let documents = car_interest_example::documents();

        let _rank = reranker.rerank(&[], &documents);

        let history = history_for_prev_docs(
            &reranker.data.prev_documents.to_coi_system_data(),
            vec![
                (Relevance::Low, UserFeedback::Irrelevant),
                (Relevance::Low, UserFeedback::Relevant),
                (Relevance::Low, UserFeedback::Relevant),
                (Relevance::Low, UserFeedback::Irrelevant),
                (Relevance::Low, UserFeedback::Irrelevant),
                (Relevance::Low, UserFeedback::Relevant),
            ],
        );

        let documents = documents_from_ids(10..20);

        let _rank = reranker.rerank(&history, &documents);
        assert!(reranker.errors().is_empty());
        assert_eq!(reranker.data.prev_documents.len(), 10);

        assert_eq!(reranker.data.user_interests.positive.len(), 3);
        assert_eq!(reranker.data.user_interests.negative.len(), 3);
    }

    /// A user performed a couple of searches. The `Reranker` data holds the
    /// previous documents from the last search. The user decides to clear
    /// their history and then do a new search. The `Reranker` should skip
    /// the learning step, discard the previous documents, rerank the
    /// current `Document`s based on the current user interests and
    /// create the previous documents from the current `Document`s.
    #[test]
    fn test_rerank_no_history() {
        let cs = MockCommonSystems::new()
            .set_db(|| MemDb::from_data(car_interest_example::reranker_data_with_mab()));
        let mut reranker = Reranker::new(cs).unwrap();

        let rank = reranker.rerank(&[], &car_interest_example::documents());

        assert_eq!(rank, car_interest_example::expected_rerank());
        assert_eq!(
            reranker.data.prev_documents.len(),
            car_interest_example::documents().len()
        );
    }

    /// This case is unlikely because the app always sends the complete
    /// history. If the app decides to only send a subset of the history like
    /// "news" or "search" history (for example if the user switches from the
    /// search to the news screen), this case will be more likely. The
    /// `Reranker` should fail in the learning step with a `NoMatchingDocuments`
    /// error, create the previous documents from the current `Document`s
    /// and rerank the current `Document`s based on the current user interests.
    #[test]
    fn test_rerank_no_matching_documents() {
        let cs = MockCommonSystems::new()
            .set_db(|| MemDb::from_data(car_interest_example::reranker_data_with_mab()));
        let mut reranker = Reranker::new(cs).unwrap();

        // creates a history with one document with the id 11
        let history = document_history(vec![(11, Relevance::Low, UserFeedback::Relevant)]);
        let rank = reranker.rerank(&history, &car_interest_example::documents());

        assert_eq!(rank, car_interest_example::expected_rerank());
        assert_eq!(reranker.data.prev_documents.len(), 6);

        check_error!(reranker, CoiSystemError::NoMatchingDocuments);
    }

    /// A user performed the very first search. The `Reranker` created the
    /// previous documents from that search. The user decides to clear
    /// their history and then do a new search. The `Reranker` should skip
    /// the learning step, discard the previous documents, rerank the
    /// current `Document`s based on the current user interests and
    /// create the previous documents from the current `Document`s.
    /// Since the learning step was skipped, the `Reranker` is not yet
    /// aware of any user interests and can therefore not perform any
    /// reranking. The `Reranker` should return the results/`Document`s
    /// in an unchanged order.
    #[test]
    fn test_first_and_second_search_no_history() {
        let cs = MockCommonSystems::new().set_db(|| {
            MemDb::from_data(RerankerData {
                prev_documents: data_with_embedding((0..10).map(|id| (id, vec![id as f32; 128])))
                    .into(),
                ..Default::default()
            })
        });
        let mut reranker = Reranker::new(cs).unwrap();
        let documents = documents_from_ids(0..10);

        let rank = reranker.rerank(&[], &documents);

        assert_eq!(rank, expected_rerank_unchanged(&documents));
        assert_eq!(reranker.data.prev_documents.len(), 10);
        assert!(reranker.data.user_interests.positive.is_empty());
        assert!(reranker.data.user_interests.negative.is_empty());

        check_error!(reranker, CoiSystemError::NoMatchingDocuments);
    }

    /// Similar to `test_first_and_second_search_no_history` but this time
    /// the `Reranker` cannot find any matching documents. The `Reranker`
    /// should return the results/`Document`s in an unchanged order and
    /// create the previous documents from the current `Document`s.
    #[test]
    fn test_first_and_second_search_no_matching_documents() {
        let cs = MockCommonSystems::new().set_db(|| {
            MemDb::from_data(RerankerData {
                prev_documents: data_with_embedding((0..10).map(|id| (id, vec![id as f32; 128])))
                    .into(),
                ..Default::default()
            })
        });
        let mut reranker = Reranker::new(cs).unwrap();
        let documents = documents_from_ids(0..10);

        // creates a history with one document with the id 11
        let history = document_history(vec![(11, Relevance::Low, UserFeedback::Relevant)]);
        let rank = reranker.rerank(&history, &documents);

        assert_eq!(rank, expected_rerank_unchanged(&documents));
        assert_eq!(reranker.data.prev_documents.len(), 10);
        assert!(reranker.data.user_interests.positive.is_empty());
        assert!(reranker.data.user_interests.negative.is_empty());

        check_error!(reranker, CoiSystemError::NoMatchingDocuments);
        check_error!(reranker, CoiSystemError::NoCoi);
    }
}
