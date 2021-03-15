#![allow(dead_code)]

use crate::{
    coi::UserInterestsStatus,
    data::{
        document::{Document, DocumentHistory},
        document_data::{
            DocumentContentComponent,
            DocumentDataWithDocument,
            DocumentDataWithEmbedding,
            DocumentDataWithMab,
            DocumentIdComponent,
        },
        UserInterests,
    },
    error::Error,
    reranker_systems::{BertSystem, CommonSystems},
};

pub type DocumentsRank = Vec<usize>;

/// In this state we have the documents from the previous query with which
/// we initialize the user interests.
pub struct InitUserInterestsFeedbackLoop {
    prev_documents: Vec<DocumentDataWithEmbedding>,
    user_interests: UserInterests,
}

/// This state will be the initial state with an empty `UserInterests` or it will be
/// the next state `InitUserInterestsFeedbackLoop` if we don't have enough cois.
/// It will just return the same rank of the source.
pub struct InitUserInterestsRerank {
    user_interests: UserInterests,
}

/// In this state we have all the data to run the feedback loop.
pub struct MainFeedbackLoop {
    prev_documents: Vec<DocumentDataWithMab>,
    user_interests: UserInterests,
}

/// In this state we can run our model to rerank documents.
pub struct MainRerank {
    user_interests: UserInterests,
}

pub struct PhaseState<S> {
    inner: S,
}

impl PhaseState<InitUserInterestsFeedbackLoop> {
    fn rerank<CS>(
        self,
        common_systems: &CS,
        history: &[DocumentHistory],
        _documents: &[Document],
    ) -> Result<(RerankerState, Option<DocumentsRank>), Error>
    where
        CS: CommonSystems,
    {
        let user_interests_orig = self.inner.user_interests.clone();
        let status = common_systems
            .coi()
            .make_user_interests(
                history,
                &self.inner.prev_documents,
                self.inner.user_interests,
            )
            .unwrap_or(UserInterestsStatus::NotEnough(user_interests_orig));

        match status {
            UserInterestsStatus::NotEnough(user_interests) => {
                let state = RerankerState::InitUserInterestsRerank(PhaseState {
                    inner: InitUserInterestsRerank { user_interests },
                });

                Ok((state, None))
            }
            UserInterestsStatus::Ready(user_interests) => {
                let state = RerankerState::MainRerank(PhaseState {
                    inner: MainRerank { user_interests },
                });

                Ok((state, None))
            }
        }
    }
}

impl PhaseState<InitUserInterestsRerank> {
    fn rerank<CS>(
        self,
        common_systems: &CS,
        _history: &[DocumentHistory],
        documents: &[Document],
    ) -> Result<(RerankerState, Option<DocumentsRank>), Error>
    where
        CS: CommonSystems,
    {
        let prev_documents = make_documents_with_embedding(common_systems.bert(), &documents)?;

        let state = RerankerState::InitUserInterestsFeedbackLoop(PhaseState {
            inner: InitUserInterestsFeedbackLoop {
                prev_documents,
                user_interests: self.inner.user_interests,
            },
        });

        let rank = documents.iter().map(|document| document.rank).collect();
        Ok((state, Some(rank)))
    }
}

impl PhaseState<MainFeedbackLoop> {
    fn rerank<CS>(
        self,
        common_systems: &CS,
        history: &[DocumentHistory],
        _documents: &[Document],
    ) -> Result<(RerankerState, Option<DocumentsRank>), Error>
    where
        CS: CommonSystems,
    {
        let user_interests_orig = self.inner.user_interests.clone();
        let user_interests = common_systems
            .coi()
            .update_user_interests(
                history,
                &self.inner.prev_documents,
                self.inner.user_interests,
            )
            .unwrap_or(user_interests_orig);

        // try to compute and save analytics
        common_systems
            .analytics()
            .compute_analytics(history, &self.inner.prev_documents)
            .and_then(|analytics| common_systems.database().save_analytics(&analytics))
            .unwrap_or_default();

        let state = RerankerState::MainRerank(PhaseState {
            inner: MainRerank { user_interests },
        });

        Ok((state, None))
    }
}

impl PhaseState<MainRerank> {
    fn rerank<CS>(
        self,
        common_systems: &CS,
        history: &[DocumentHistory],
        documents: &[Document],
    ) -> Result<(RerankerState, Option<DocumentsRank>), Error>
    where
        CS: CommonSystems,
    {
        let documents = make_documents_with_embedding(common_systems.bert(), &documents)?;
        let documents = common_systems
            .coi()
            .compute_coi(documents, &self.inner.user_interests)?;
        let documents = common_systems.ltr().compute_ltr(history, documents)?;
        let documents = common_systems.context().compute_context(documents)?;
        let (documents, user_interests) = common_systems
            .mab()
            .compute_mab(documents, self.inner.user_interests)?;

        let ranks = documents.iter().map(|document| document.mab.rank).collect();

        let inner = RerankerState::MainFeedbackLoop(PhaseState {
            inner: MainFeedbackLoop {
                prev_documents: documents,
                user_interests,
            },
        });

        Ok((inner, Some(ranks)))
    }
}

pub enum RerankerState {
    InitUserInterestsFeedbackLoop(PhaseState<InitUserInterestsFeedbackLoop>),
    InitUserInterestsRerank(PhaseState<InitUserInterestsRerank>),
    MainFeedbackLoop(PhaseState<MainFeedbackLoop>),
    MainRerank(PhaseState<MainRerank>),
}

impl RerankerState {
    fn rerank<CS>(
        self,
        common_systems: &CS,
        history: &[DocumentHistory],
        documents: &[Document],
    ) -> Result<(RerankerState, Option<DocumentsRank>), Error>
    where
        CS: CommonSystems,
    {
        match self {
            RerankerState::InitUserInterestsFeedbackLoop(state) => {
                state.rerank(common_systems, history, documents)
            }
            RerankerState::InitUserInterestsRerank(state) => {
                state.rerank(common_systems, history, documents)
            }
            RerankerState::MainFeedbackLoop(state) => {
                state.rerank(common_systems, history, documents)
            }
            RerankerState::MainRerank(state) => state.rerank(common_systems, history, documents),
        }
    }
}

pub struct Reranker<CS> {
    common_systems: CS,
    state: RerankerState,
}

impl<CS> Reranker<CS>
where
    CS: CommonSystems,
{
    fn new(common_systems: CS) -> Result<Self, Error> {
        // load the correct state from the database
        let state = common_systems.database().load_state().map(|inner| {
            inner.unwrap_or_else(|| {
                RerankerState::InitUserInterestsRerank(PhaseState {
                    inner: InitUserInterestsRerank {
                        user_interests: UserInterests::default(),
                    },
                })
            })
        })?;

        Ok(Self {
            common_systems,
            state,
        })
    }

    fn rerank(
        mut self,
        history: &[DocumentHistory],
        documents: &[Document],
    ) -> Result<DocumentsRank, Error> {
        loop {
            let (state, rank) = self
                .state
                .rerank(&self.common_systems, history, documents)?;
            self.state = state;

            if let Some(rank) = rank {
                self.common_systems.database().save_state(&self.state)?;
                return Ok(rank);
            }
        }
    }
}

fn make_documents_with_embedding<BS>(
    bert_system: &BS,
    documents: &[Document],
) -> Result<Vec<DocumentDataWithEmbedding>, Error>
where
    BS: BertSystem + ?Sized,
{
    // TODO: we should probably consume documents by value to avoid unnecessary cloning
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
    bert_system.compute_embedding(documents)
}

fn rerank_documents<CS>(
    common_systems: &CS,
    history: &[DocumentHistory],
    documents: &[Document],
    user_interests: UserInterests,
) -> Result<(RerankerState, Option<DocumentsRank>), Error>
where
    CS: CommonSystems,
{
    let documents = make_documents_with_embedding(common_systems.bert(), &documents)?;
    let documents = common_systems
        .coi()
        .compute_coi(documents, &user_interests)?;
    let documents = common_systems.ltr().compute_ltr(history, documents)?;
    let documents = common_systems.context().compute_context(documents)?;
    let (documents, user_interests) = common_systems
        .mab()
        .compute_mab(documents, user_interests)?;

    let ranks = documents.iter().map(|document| document.mab.rank).collect();

    let inner = RerankerState::MainFeedbackLoop(PhaseState {
        inner: MainFeedbackLoop {
            prev_documents: documents,
            user_interests,
        },
    });

    Ok((inner, Some(ranks)))
}
