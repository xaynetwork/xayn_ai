pub(crate) mod database;
pub(crate) mod public;
pub(crate) mod sync;
pub(crate) mod systems;

#[cfg(test)]
use derive_more::From;
use serde::{Deserialize, Serialize};
use serde_repr::{Deserialize_repr, Serialize_repr};

use crate::{
    analytics::Analytics,
    coi::NeutralCoiSystem,
    data::{
        document::{Document, DocumentHistory, RerankingOutcomes},
        document_data::{
            make_documents,
            DocumentDataWithContext,
            DocumentDataWithRank,
            DocumentDataWithSMBert,
            RankComponent,
        },
    },
    embedding::{qambert::NeutralQAMBert, smbert::NeutralSMBert},
    error::Error,
    ltr::ConstLtr,
    reranker::{
        database::RerankerData,
        sync::SyncData,
        systems::{
            CoiSystem,
            CoiSystemData,
            CommonSystems,
            LtrSystem,
            QAMBertSystem,
            SMBertSystem,
        },
    },
    utils::{nan_safe_f32_cmp, to_vec_of_ref_of},
};

const CURRENT_SCHEMA_VERSION: u8 = 2;

/// The mode used to run reranking with.
///
/// This will influence how exactly the reranking
/// is done. E.g. using `News` will disable the
/// QA-mBert pipeline.
#[derive(Clone, Copy, Deserialize_repr, Serialize_repr)]
#[repr(u8)]
pub enum RerankMode {
    /// Run reranking for news.
    StandardNews = 0,
    /// Run reranking for news with personalization.
    PersonalizedNews = 1,
    /// Run reranking for search.
    StandardSearch = 2,
    /// Run reranking for search with personalization.
    PersonalizedSearch = 3,
}

impl RerankMode {
    pub(crate) fn is_personalized(&self) -> bool {
        matches!(self, Self::PersonalizedNews | Self::PersonalizedSearch)
    }

    pub(crate) fn is_search(&self) -> bool {
        matches!(self, Self::StandardSearch | Self::PersonalizedSearch)
    }
}

#[derive(Deserialize, Serialize)]
#[cfg_attr(test, derive(Clone, Debug, From, PartialEq))]
pub(super) enum PreviousDocuments {
    None,
    Embedding(Vec<DocumentDataWithSMBert>),
    Final(Vec<DocumentDataWithRank>),
}

impl Default for PreviousDocuments {
    fn default() -> Self {
        Self::None
    }
}

impl PreviousDocuments {
    pub(super) fn to_coi_system_data(&self) -> Option<Vec<&dyn CoiSystemData>> {
        match self {
            PreviousDocuments::None => None,
            PreviousDocuments::Embedding(documents) => {
                Some(to_vec_of_ref_of!(documents, &dyn CoiSystemData))
            }
            PreviousDocuments::Final(documents) => {
                Some(to_vec_of_ref_of!(documents, &dyn CoiSystemData))
            }
        }
    }

    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        match self {
            PreviousDocuments::None => 0,
            PreviousDocuments::Embedding(documents) => documents.len(),
            PreviousDocuments::Final(documents) => documents.len(),
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
        let data = common_systems.database().load_data()?.unwrap_or_default();

        Ok(Self {
            common_systems,
            data,
            errors: Vec::new(),
            analytics: None,
        })
    }

    pub(crate) fn errors(&self) -> &[Error] {
        self.errors.as_slice()
    }

    /// Returns the analytics for penultimate call to `rerank`.
    /// Analytics will be provided only if the penultimate call to `rerank` was able
    /// to run the full model without error, and the correct history is passed to the
    /// last call to `rerank`.
    pub(crate) fn analytics(&self) -> Option<&Analytics> {
        self.analytics.as_ref()
    }

    /// Create a byte representation of the internal state of the Reranker.
    pub(crate) fn serialize(&self) -> Result<Vec<u8>, Error> {
        self.common_systems.database().serialize(&self.data)
    }

    /// Create a byte representation of the synchronizable data.
    pub(crate) fn syncdata_bytes(&self) -> Result<Vec<u8>, Error> {
        self.data.sync_data.serialize()
    }

    /// Synchronizes internal data with the `SyncData` given in serialized form.
    pub(crate) fn synchronize(&mut self, bytes: &[u8]) -> Result<(), Error> {
        let remote_data = SyncData::deserialize(bytes)?;
        self.data.sync_data.synchronize(remote_data);
        Ok(())
    }

    /// Reranks the documents based on the chosen mode.
    pub(crate) fn rerank(
        &mut self,
        mode: RerankMode,
        history: &[DocumentHistory],
        documents: &[Document],
    ) -> RerankingOutcomes {
        // The number of errors it can contain is very limited. By using `clear` we avoid
        // re-allocating the vector on each method call.
        self.errors.clear();

        // feedback loop and analytics
        self.learn_user_interests(history);
        self.collect_analytics(history);

        match mode {
            RerankMode::StandardNews => self.rerank_standard_news(history, documents),
            RerankMode::PersonalizedNews => self.rerank_personalized_news(history, documents),
            RerankMode::StandardSearch => self.rerank_standard_search(history, documents),
            RerankMode::PersonalizedSearch => self.rerank_personalized_search(history, documents),
        }
        .map(|documents_with_rank| {
            let outcome = RerankingOutcomes::from_rank(mode, documents, &documents_with_rank);

            if mode.is_personalized() {
                self.data.prev_documents = PreviousDocuments::Final(documents_with_rank);
            } else {
                self.data.prev_documents = PreviousDocuments::None;
            }

            outcome
        })
        .unwrap_or_else(|e| {
            self.errors.push(e);

            if mode.is_personalized() {
                let prev_documents = make_documents(documents);
                let prev_documents = self
                    .common_systems
                    .smbert()
                    .compute_embedding(prev_documents)
                    .unwrap_or_default();
                self.data.prev_documents = PreviousDocuments::Embedding(prev_documents);
            } else {
                self.data.prev_documents = PreviousDocuments::None;
            }

            RerankingOutcomes::from_initial_ranking(documents)
        })
    }

    /// Updates cois from user feedback.
    fn learn_user_interests(&mut self, history: &[DocumentHistory]) {
        if let Some(prev_documents) = self.data.prev_documents.to_coi_system_data() {
            match self.common_systems.coi().update_user_interests(
                history,
                &prev_documents,
                self.data.sync_data.user_interests.clone(),
            ) {
                Ok(user_interests) => self.data.sync_data.user_interests = user_interests,
                Err(e) => self.errors.push(e),
            }
        }
    }

    /// Computes and saves analytics.
    fn collect_analytics(&mut self, history: &[DocumentHistory]) {
        if let PreviousDocuments::Final(ref prev_documents) = self.data.prev_documents {
            self.analytics = self
                .common_systems
                .analytics()
                .compute_analytics(history, prev_documents)
                .map_err(|e| self.errors.push(e))
                .ok();
        }
    }

    /// Reranks the documents without any systems.
    fn rerank_standard_news(
        &self,
        history: &[DocumentHistory],
        documents: &[Document],
    ) -> Result<Vec<DocumentDataWithRank>, Error> {
        let documents = make_documents(documents);
        let documents = NeutralSMBert.compute_embedding(documents)?;
        let documents = NeutralQAMBert.compute_similarity(documents)?;
        let documents =
            NeutralCoiSystem.compute_coi(documents, &self.data.sync_data.user_interests)?;
        let documents = ConstLtr.compute_ltr(history, documents)?;
        let documents = self.common_systems.context().compute_context(documents)?;
        let documents = rank_by_identity(documents); // stable order needed

        Ok(documents)
    }

    /// Reranks the documents with all systems except QAMbert.
    fn rerank_personalized_news(
        &self,
        history: &[DocumentHistory],
        documents: &[Document],
    ) -> Result<Vec<DocumentDataWithRank>, Error> {
        let documents = make_documents(documents);
        let documents = self.common_systems.smbert().compute_embedding(documents)?;
        let documents = NeutralQAMBert.compute_similarity(documents)?;
        let documents = self
            .common_systems
            .coi()
            .compute_coi(documents, &self.data.sync_data.user_interests)?;
        let documents = self.common_systems.ltr().compute_ltr(history, documents)?;
        let documents = self.common_systems.context().compute_context(documents)?;
        let documents = rank_by_context(documents);

        Ok(documents)
    }

    /// Reranks the documents just with the QAMbert system.
    fn rerank_standard_search(
        &self,
        history: &[DocumentHistory],
        documents: &[Document],
    ) -> Result<Vec<DocumentDataWithRank>, Error> {
        let documents = make_documents(documents);
        let documents = NeutralSMBert.compute_embedding(documents)?;
        let documents = self
            .common_systems
            .qambert()
            .compute_similarity(documents)?;
        let documents =
            NeutralCoiSystem.compute_coi(documents, &self.data.sync_data.user_interests)?;
        let documents = ConstLtr.compute_ltr(history, documents)?;
        let documents = self.common_systems.context().compute_context(documents)?;
        let documents = rank_by_context(documents);

        Ok(documents)
    }

    /// Reranks the documents with all systems.
    fn rerank_personalized_search(
        &self,
        history: &[DocumentHistory],
        documents: &[Document],
    ) -> Result<Vec<DocumentDataWithRank>, Error> {
        let documents = make_documents(documents);
        let documents = self.common_systems.smbert().compute_embedding(documents)?;
        let documents = self
            .common_systems
            .qambert()
            .compute_similarity(documents)?;
        let documents = self
            .common_systems
            .coi()
            .compute_coi(documents, &self.data.sync_data.user_interests)?;
        let documents = self.common_systems.ltr().compute_ltr(history, documents)?;
        let documents = self.common_systems.context().compute_context(documents)?;
        let documents = rank_by_context(documents);

        Ok(documents)
    }
}

pub(crate) fn rank_by_identity(docs: Vec<DocumentDataWithContext>) -> Vec<DocumentDataWithRank> {
    docs.into_iter()
        .enumerate()
        .map(|(rank, doc)| DocumentDataWithRank::from_document(doc, RankComponent { rank }))
        .collect()
}

pub(crate) fn rank_by_context(mut docs: Vec<DocumentDataWithContext>) -> Vec<DocumentDataWithRank> {
    docs.sort_unstable_by(|a, b| {
        nan_safe_f32_cmp(&b.context.context_value, &a.context.context_value)
    });
    rank_by_identity(docs)
}

#[cfg(test)]
mod tests {
    use anyhow::bail;
    use paste::paste;
    use rstest::rstest;
    use rstest_reuse::{apply, template};

    use super::*;
    use crate::{
        analytics::NoRelevantHistoricInfo,
        coi::CoiSystemError,
        data::document::{Relevance, UserFeedback},
        tests::{
            document_history,
            documents_from_ids,
            documents_with_embeddings_from_ids,
            expected_rerank_unchanged,
            history_for_prev_docs,
            mocked_smbert_system,
            MemDb,
            MockAnalyticsSystem,
            MockCoiSystem,
            MockCommonSystems,
            MockContextSystem,
            MockDatabase,
            MockLtrSystem,
            MockSMBertSystem,
        },
    };

    macro_rules! contains_error {
        ($reranker:expr, $error:pat $(,)?) => {
            $reranker
                .errors()
                .iter()
                .any(|e| matches!(e.downcast_ref(), Some($error)))
        };
    }

    macro_rules! assert_contains_error {
        ($reranker:expr, $error:pat $(,)?) => {
            assert!(contains_error!($reranker, $error))
        };
    }

    mod car_interest_example {
        use std::ops::Range;

        use crate::{
            coi::point::UserInterests,
            data::document::{Document, RerankingOutcomes},
            reranker::{sync::SyncData, PreviousDocuments, RerankerData},
            tests::{
                data_with_rank,
                documents_from_words,
                from_ids,
                mocked_smbert_system,
                pos_cois_from_words,
            },
        };

        pub(super) fn reranker_data_with_rank_from_ids(ids: Range<u32>) -> RerankerData {
            let docs = data_with_rank(from_ids(ids));
            reranker_data(docs)
        }

        pub(super) fn documents() -> Vec<Document> {
            documents_from_words(
                (0..6).zip(&["ship", "car", "auto", "flugzeug", "plane", "vehicle"]),
            )
        }

        pub(super) fn expected_rerank_outcome() -> RerankingOutcomes {
            // the (id, rank) mapping is [5, 3, 4, 0, 2, 1].zip(0..6)
            RerankingOutcomes {
                final_ranking: vec![3, 5, 4, 1, 2, 0],
                qambert_similarities: None,
                context_scores: None,
            }
        }

        fn reranker_data(docs: impl Into<PreviousDocuments>) -> RerankerData {
            RerankerData {
                prev_documents: docs.into(),
                sync_data: SyncData {
                    user_interests: UserInterests {
                        positive: pos_cois_from_words(&["vehicle"], mocked_smbert_system()),
                        ..UserInterests::default()
                    },
                },
            }
        }
    }

    /// Template to run a test with all the rerank mode we support.
    #[template]
    #[rstest(
        mode,
        case(RerankMode::StandardNews),
        case(RerankMode::PersonalizedNews),
        case(RerankMode::StandardSearch),
        case(RerankMode::PersonalizedSearch)
    )]
    fn tmpl_rerank_mode_cases(mode: RerankMode) {}

    /// A user performs the very first search that returns no results/`Documents`.
    /// In this case, the `Reranker` should return an empty `DocumentsRank`.
    #[apply(tmpl_rerank_mode_cases)]
    fn test_first_search_without_search_results(mode: RerankMode) {
        let cs = MockCommonSystems::default();
        let mut reranker = Reranker::new(cs).unwrap();

        let outcome = reranker.rerank(mode, &[], &[]);
        assert!(outcome.final_ranking.is_empty());
    }

    /// A user performs the very first search that returns results/`Document`s. The `Reranker` is
    /// not yet aware of any user interests and can therefore not perform any reranking. In this
    /// case, the `Reranker` should return the results/`Document`s in an unchanged order.
    /// Furthermore, in case of personalized queries, the `Reranker` should create
    /// `DocumentDataWithEmbedding` from the `Document`s, from which the first user interests will
    /// be learned in the next call to `rerank`.
    #[apply(tmpl_rerank_mode_cases)]
    fn test_first_search_with_search_results(mode: RerankMode) {
        let cs = MockCommonSystems::default();
        let mut reranker = Reranker::new(cs).unwrap();
        let documents = documents_from_ids(0..10);

        let outcome = reranker.rerank(mode, &[], &documents);

        assert_eq!(outcome.final_ranking, expected_rerank_unchanged(&documents));
        assert_eq!(
            reranker.data.prev_documents.len(),
            mode.is_personalized()
                .then(|| documents.len())
                .unwrap_or_default(),
        );
        assert!(reranker.data.sync_data.user_interests.positive.is_empty());
        assert!(reranker.data.sync_data.user_interests.negative.is_empty());

        if mode.is_personalized() {
            assert_contains_error!(reranker, CoiSystemError::NoCoi);
        } else {
            assert!(reranker.errors().is_empty());
        }
    }

    /// A user performed the very first search. The `Reranker` created the
    /// previous documents if that search was personalized. The next time the user
    /// searches, the user interests should be learned from the previous
    /// documents and the current results/`Document`s should be reranked
    /// based on the newly learned user interests.
    #[apply(tmpl_rerank_mode_cases)]
    fn test_first_and_second_search_learn_cois_and_rerank(
        mode: RerankMode,
        #[values(
            RerankMode::StandardNews,
            RerankMode::PersonalizedNews,
            RerankMode::StandardSearch,
            RerankMode::PersonalizedSearch
        )]
        second_mode: RerankMode,
    ) {
        let cs = MockCommonSystems::default();
        let mut reranker = Reranker::new(cs).unwrap();
        let documents = car_interest_example::documents();

        let _rank = reranker.rerank(mode, &[], &documents);

        let history = reranker
            .data
            .prev_documents
            .to_coi_system_data()
            .map(|prev_documents| {
                history_for_prev_docs(
                    &prev_documents,
                    vec![
                        (Relevance::Low, UserFeedback::Irrelevant),
                        (Relevance::Low, UserFeedback::Relevant),
                        (Relevance::Low, UserFeedback::Relevant),
                        (Relevance::Low, UserFeedback::Irrelevant),
                        (Relevance::Low, UserFeedback::Irrelevant),
                        (Relevance::Low, UserFeedback::Relevant),
                    ],
                )
            })
            .unwrap_or_default();

        let documents = documents_from_ids(10..20);

        let _rank = reranker.rerank(second_mode, &history, &documents);
        if !mode.is_personalized() && second_mode.is_personalized() {
            assert_contains_error!(reranker, CoiSystemError::NoCoi);
        } else {
            assert!(reranker.errors().is_empty());
        }
        assert_eq!(
            reranker.data.prev_documents.len(),
            second_mode
                .is_personalized()
                .then(|| documents.len())
                .unwrap_or_default(),
        );

        let coi_len = mode.is_personalized().then(|| 3).unwrap_or_default();
        assert_eq!(
            reranker.data.sync_data.user_interests.positive.len(),
            coi_len,
        );
        assert_eq!(
            reranker.data.sync_data.user_interests.negative.len(),
            coi_len,
        );
    }

    /// A user performed a couple of searches. The `Reranker` data holds the previous documents from
    /// the last search. The user decides to clear their history and then do a new search. The
    /// `Reranker` should skip the learning step and discard the previous documents. In case of a
    /// personalized query, it reranks the current `Document`s based on the current user interests
    /// and create the previous documents from the current `Document`s. Otherwise it shouldn't
    /// rerank and create no previous documents.
    #[apply(tmpl_rerank_mode_cases)]
    fn test_rerank_no_history(mode: RerankMode) {
        let cs = MockCommonSystems::new().set_db(|| {
            MemDb::from_data(car_interest_example::reranker_data_with_rank_from_ids(
                0..10,
            ))
        });
        let mut reranker = Reranker::new(cs).unwrap();

        let documents = car_interest_example::documents();
        let outcome = reranker.rerank(mode, &[], &documents);

        assert_eq!(
            outcome.final_ranking,
            mode.is_personalized()
                .then(|| car_interest_example::expected_rerank_outcome().final_ranking)
                .unwrap_or_else(|| expected_rerank_unchanged(&documents)),
        );
        assert_eq!(
            reranker.data.prev_documents.len(),
            mode.is_personalized()
                .then(|| car_interest_example::documents().len())
                .unwrap_or_default(),
        );
    }

    /// This case is unlikely because the app always sends the complete history. If the app decides
    /// to only send a subset of the history like "news" or "search" history (for example if the
    /// user switches from the search to the news screen), this case will be more likely. In case of
    /// a personalized query, the `Reranker` should fail in the learning step with a
    /// `NoMatchingDocuments` error, create the previous documents from the current `Document`s and
    /// rerank the current `Document`s based on the current user interests. Otherwise it should skip
    /// the learning step and fail in the analysis step with a `NoRelevantHistoricInfo` error, it
    /// shouldn't rerank and create no previous documents.
    #[apply(tmpl_rerank_mode_cases)]
    fn test_rerank_no_matching_documents(mode: RerankMode) {
        let cs = MockCommonSystems::new().set_db(|| {
            MemDb::from_data(car_interest_example::reranker_data_with_rank_from_ids(
                0..10,
            ))
        });
        let mut reranker = Reranker::new(cs).unwrap();

        // creates a history with one document with the id 11
        let history = document_history(vec![(11, Relevance::Low, UserFeedback::Relevant)]);
        let documents = car_interest_example::documents();
        let outcome = reranker.rerank(mode, &history, &documents);

        assert_eq!(
            outcome.final_ranking,
            mode.is_personalized()
                .then(|| car_interest_example::expected_rerank_outcome().final_ranking)
                .unwrap_or_else(|| expected_rerank_unchanged(&documents)),
        );
        assert_eq!(
            reranker.data.prev_documents.len(),
            mode.is_personalized()
                .then(|| documents.len())
                .unwrap_or_default(),
        );

        if mode.is_personalized() {
            assert_contains_error!(reranker, CoiSystemError::NoMatchingDocuments);
        } else {
            assert_contains_error!(reranker, NoRelevantHistoricInfo);
        }
    }

    /// A user performed the very first search. In case of a personalized query, the `Reranker`
    /// created the previous documents from that search. The user decides to clear their history and
    /// then do a new search. The `Reranker` should skip the learning step and discard the previous
    /// documents. Since the learning step was skipped, the `Reranker` is not yet aware of any user
    /// interests and can therefore not perform any reranking. The `Reranker` should return the
    /// results/`Document`s in an unchanged order. In case of a personalized query, those documents
    /// are set as the new previous documents.
    #[apply(tmpl_rerank_mode_cases)]
    fn test_first_and_second_search_no_history(mode: RerankMode) {
        let cs = MockCommonSystems::new().set_db(|| {
            MemDb::from_data(RerankerData {
                prev_documents: documents_with_embeddings_from_ids(0..10).into(),
                ..RerankerData::default()
            })
        });
        let mut reranker = Reranker::new(cs).unwrap();
        let documents = documents_from_ids(0..10);

        let outcome = reranker.rerank(mode, &[], &documents);

        assert_eq!(outcome.final_ranking, expected_rerank_unchanged(&documents));
        assert_eq!(
            reranker.data.prev_documents.len(),
            mode.is_personalized()
                .then(|| documents.len())
                .unwrap_or_default(),
        );
        assert!(reranker.data.sync_data.user_interests.positive.is_empty());
        assert!(reranker.data.sync_data.user_interests.negative.is_empty());

        assert_contains_error!(reranker, CoiSystemError::NoMatchingDocuments);
    }

    /// Similar to `test_first_and_second_search_no_history` but this time the `Reranker` cannot
    /// find any matching documents. The `Reranker` should return the results/`Document`s in an
    /// unchanged order and in case of a personalized query create the previous documents from the
    /// current `Document`s.
    #[apply(tmpl_rerank_mode_cases)]
    fn test_first_and_second_search_no_matching_documents(mode: RerankMode) {
        let cs = MockCommonSystems::new().set_db(|| {
            MemDb::from_data(RerankerData {
                prev_documents: documents_with_embeddings_from_ids(0..10).into(),
                ..Default::default()
            })
        });
        let mut reranker = Reranker::new(cs).unwrap();
        let documents = documents_from_ids(0..10);

        // creates a history with one document with the id 11
        let history = document_history(vec![(11, Relevance::Low, UserFeedback::Relevant)]);
        let outcome = reranker.rerank(mode, &history, &documents);

        assert_eq!(outcome.final_ranking, expected_rerank_unchanged(&documents));
        assert_eq!(
            reranker.data.prev_documents.len(),
            mode.is_personalized()
                .then(|| documents.len())
                .unwrap_or_default(),
        );
        assert!(reranker.data.sync_data.user_interests.positive.is_empty());
        assert!(reranker.data.sync_data.user_interests.negative.is_empty());

        assert_contains_error!(reranker, CoiSystemError::NoMatchingDocuments);
        if mode.is_personalized() {
            assert_contains_error!(reranker, CoiSystemError::NoCoi);
        }
    }

    #[derive(thiserror::Error, Debug)]
    enum MockError {
        #[error("fail")]
        Fail,
    }

    macro_rules! common_systems_with_fail {
        ($system:ident, $mock:ty, $method:ident, |$($args:tt),*|) => {
            paste! {{
                let mut mock_system = $mock::new();
                mock_system.[<expect_$method>]().returning(|$($args),*| bail!(MockError::Fail));

                let cs = MockCommonSystems::default()
                    .[<set_$system>](|| mock_system)
                    .set_db(|| {
                        // We need to set at least one positive coi, otherwise
                        // `rerank` will fail with `CoiSystemError::NoCoi` and
                        // the systems that come after the `CoiSystem` will never
                        // be executed.
                        MemDb::from_data(car_interest_example::reranker_data_with_rank_from_ids(0..1))
                    });
                cs
            }}
        }
    }

    /// If any of the systems fail in the `rerank` method, the `Reranker` should return the results/
    /// `Document`s in an unchanged order. In case of a personalized query, it should fail with a
    /// `NoMatchingDocuments` error and create the previous documents from the current `Document`s.
    /// Otherwise it might potentially just fail in the analysis step, because neutral reranker
    /// components are infallible. An exception is the smbert system which of course cannot create
    /// previous documents if it fails.
    fn test_system_failure(cs: impl CommonSystems, can_fill_prev_docs: bool, mode: RerankMode) {
        let mut reranker = Reranker::new(cs).unwrap();
        let documents = documents_from_ids(0..10);

        // We use an empty history in order to skip the learning step.
        let outcome = reranker.rerank(mode, &[], &documents);

        assert_eq!(outcome.final_ranking, expected_rerank_unchanged(&documents));
        if mode.is_personalized() {
            assert_contains_error!(reranker, MockError::Fail);
        } else {
            assert!(reranker.errors.is_empty() || !contains_error!(reranker, MockError::Fail));
        }
        assert_eq!(
            reranker.data.prev_documents.len(),
            (can_fill_prev_docs && mode.is_personalized())
                .then(|| documents.len())
                .unwrap_or_default(),
        );
    }

    macro_rules! test_system_failure {
        ($system:ident, $mock:ty, $method:ident, |$($args:tt),*|) => {
            test_system_failure!($system, $mock, $method, |$($args),*|, true);
        };
        ($system:ident, $mock:ty, $method:ident, |$($args:tt),*|, $can_fill_prev_docs: expr) => {
            paste! {
                #[apply(tmpl_rerank_mode_cases)]
                fn [<test_component_failure_ $system>](mode: RerankMode) {
                    let cs = common_systems_with_fail!($system, $mock, $method, |$($args),*|);
                    test_system_failure(cs, $can_fill_prev_docs, mode);
                }
            }
        };
    }

    test_system_failure!(smbert, MockSMBertSystem, compute_embedding, |_|, false);
    test_system_failure!(ltr, MockLtrSystem, compute_ltr, |_,_|);

    #[apply(tmpl_rerank_mode_cases)]
    fn test_component_failure_context(mode: RerankMode) {
        let cs = common_systems_with_fail!(context, MockContextSystem, compute_context, |_|);
        let mut reranker = Reranker::new(cs).unwrap();
        let documents = documents_from_ids(0..10);

        // We use an empty history in order to skip the learning step.
        let outcome = reranker.rerank(mode, &[], &documents);

        assert_eq!(outcome.final_ranking, expected_rerank_unchanged(&documents));
        if mode.is_personalized() {
            assert_contains_error!(reranker, CoiSystemError::NoMatchingDocuments);
        }
        assert_contains_error!(reranker, MockError::Fail);
        assert_eq!(
            reranker.data.prev_documents.len(),
            mode.is_personalized()
                .then(|| documents.len())
                .unwrap_or_default(),
        );
    }

    /// An analytics system error should not prevent the documents from
    /// being reranked using the learned user interests. However, the error
    /// should be stored and made available via `Reranker::error()`.
    #[apply(tmpl_rerank_mode_cases)]
    fn test_system_failure_analytics(mode: RerankMode) {
        let cs =
            common_systems_with_fail!(analytics, MockAnalyticsSystem, compute_analytics, |_,_|);
        let mut reranker = Reranker::new(cs).unwrap();
        reranker.analytics = Some(Analytics {
            ndcg_initial_ranking: 0.,
            ndcg_ltr: 0.,
            ndcg_context: 0.,
            ndcg_final_ranking: 0.,
        });
        let documents = car_interest_example::documents();
        let outcome = reranker.rerank(mode, &[], &documents);

        assert_eq!(
            outcome.final_ranking,
            mode.is_personalized()
                .then(|| car_interest_example::expected_rerank_outcome().final_ranking)
                .unwrap_or_else(|| expected_rerank_unchanged(&documents)),
        );
        assert_contains_error!(reranker, MockError::Fail);
        assert!(reranker.analytics.is_none())
    }

    #[apply(tmpl_rerank_mode_cases)]
    fn test_system_failure_coi_fails_in_rerank(mode: RerankMode) {
        let cs = MockCommonSystems::default().set_coi(|| {
            let mut coi = MockCoiSystem::new();
            // we need to set this otherwise it will panic when called
            coi.expect_update_user_interests()
                .returning(|_, _, _| bail!(CoiSystemError::NoMatchingDocuments));
            coi.expect_compute_coi()
                .returning(|_, _| bail!(MockError::Fail));
            coi
        });

        test_system_failure(cs, true, mode);
    }

    /// If the smbert system fails spontaneously in the `rerank` function, the `Reranker` should
    /// return the results/`Document`s in an unchanged order. In case of a personalized query, it
    /// should create the previous documents from the current `Document`s.
    #[apply(tmpl_rerank_mode_cases)]
    fn test_system_failure_smbert_fails_in_rerank(mode: RerankMode) {
        let mut called = 0;

        let cs = MockCommonSystems::default().set_smbert(|| {
            let mut smbert = MockSMBertSystem::new();
            smbert.expect_compute_embedding().returning(move |docs| {
                let res = match called {
                    0 => Err(MockError::Fail.into()),
                    1 => mocked_smbert_system().compute_embedding(docs),
                    _ => panic!("`compute_embedding` should only be called twice"),
                };
                called += 1;
                res
            });
            smbert
        });

        let mut reranker = Reranker::new(cs).unwrap();
        let documents = car_interest_example::documents();

        let outcome = reranker.rerank(mode, &[], &documents);

        assert_eq!(outcome.final_ranking, expected_rerank_unchanged(&documents));
        assert_eq!(
            reranker.data.prev_documents.len(),
            mode.is_personalized()
                .then(|| documents.len())
                .unwrap_or_default(),
        );
        if mode.is_personalized() {
            assert_contains_error!(reranker, MockError::Fail);
        } else {
            assert!(reranker.errors().is_empty());
        }
    }

    /// If the database fails to load the data, propagate the error to the caller.
    #[test]
    fn test_data_read_load_data_fails() {
        let cs = MockCommonSystems::default().set_db(|| {
            let mut db = MockDatabase::new();
            db.expect_load_data().returning(|| bail!(MockError::Fail));
            db
        });

        match Reranker::new(cs) {
            Ok(_) => panic!("an error is expected"),
            Err(error) => assert!(matches!(error.downcast_ref(), Some(MockError::Fail))),
        };
    }

    #[apply(tmpl_rerank_mode_cases)]
    fn test_analytics(mode: RerankMode) {
        let cs = MockCommonSystems::default();
        let mut reranker = Reranker::new(cs).unwrap();
        let documents = car_interest_example::documents();

        // first rerank will return api ranking
        let _rank = reranker.rerank(mode, &[], &documents);

        let history = reranker
            .data
            .prev_documents
            .to_coi_system_data()
            .map(|prev_documents| {
                history_for_prev_docs(
                    &prev_documents,
                    vec![
                        (Relevance::Low, UserFeedback::Irrelevant),
                        (Relevance::High, UserFeedback::Relevant),
                        (Relevance::High, UserFeedback::Relevant),
                        (Relevance::Low, UserFeedback::Irrelevant),
                        (Relevance::Low, UserFeedback::Irrelevant),
                        (Relevance::High, UserFeedback::Relevant),
                    ],
                )
            })
            .unwrap_or_default();

        // feedbackloop generate cois and can rerank
        let _rank = reranker.rerank(mode, &history, &documents);

        assert!(reranker.errors().is_empty());
        // the previous ranking was not able to run because
        // we don't have coi so the analytics is empty
        assert!(reranker.analytics().is_none());

        let _rank = reranker.rerank(mode, &history, &documents);
        assert!(reranker.errors().is_empty());
        assert_eq!(reranker.analytics().is_some(), mode.is_personalized());
    }
}
