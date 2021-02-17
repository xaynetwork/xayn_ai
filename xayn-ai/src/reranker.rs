#![allow(dead_code)]

use crate::{
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

/// Empty state
enum Empty {}

/// In this state we have the documents from the previous query but we do not have user interests
struct InitUserInterests {
    prev_documents: Vec<DocumentDataWithEmbedding>,
}

/// In this state we have all the data we need to do reranking and run the feedback loop
struct Nominal {
    prev_documents: Vec<DocumentDataWithMab>,
    user_interests: UserInterests,
}

struct RerankerState<S> {
    inner: S,
}

enum RerankerInner {
    Empty,
    InitUserInterests(RerankerState<InitUserInterests>),
    Nominal(RerankerState<Nominal>),
}

impl RerankerInner {
    fn rerank<CS>(
        self,
        common_systems: &CS,
        history: &[DocumentHistory],
        documents: &[Document],
    ) -> Result<(RerankerInner, DocumentsRank), Error>
    where
        CS: CommonSystems,
    {
        match self {
            RerankerInner::Empty => {
                RerankerState::<Empty>::rerank(common_systems, history, documents)
            }
            RerankerInner::InitUserInterests(state) => {
                state.rerank(common_systems, history, documents)
            }
            RerankerInner::Nominal(state) => state.rerank(common_systems, history, documents),
        }
    }
}

pub struct Reranker<CS> {
    common_systems: CS,
    inner: RerankerInner,
}

impl RerankerState<Empty> {
    fn rerank<CS>(
        common_systems: &CS,
        _history: &[DocumentHistory],
        documents: &[Document],
    ) -> Result<(RerankerInner, DocumentsRank), Error>
    where
        CS: CommonSystems,
    {
        Ok((
            to_init_user_interests(common_systems, documents)?,
            rank_from_source(documents),
        ))
    }
}

impl RerankerState<InitUserInterests> {
    fn rerank<CS>(
        self,
        common_systems: &CS,
        history: &[DocumentHistory],
        documents: &[Document],
    ) -> Result<(RerankerInner, DocumentsRank), Error>
    where
        CS: CommonSystems,
    {
        let user_interests = common_systems
            .coi()
            .make_user_interests(history, &self.inner.prev_documents)?;

        match user_interests {
            None => Ok((
                to_init_user_interests(common_systems, documents)?,
                rank_from_source(documents),
            )),
            Some(user_interests) => {
                rerank_documents(common_systems, history, documents, &user_interests)
            }
        }
    }
}

impl RerankerState<Nominal> {
    fn rerank<CS>(
        self,
        common_systems: &CS,
        history: &[DocumentHistory],
        documents: &[Document],
    ) -> Result<(RerankerInner, DocumentsRank), Error>
    where
        CS: CommonSystems,
    {
        let user_interests = common_systems
            .coi()
            .update_user_interests(history, documents, &self.inner.user_interests)?;

        common_systems
            .database()
            .save_user_interests(&user_interests)?;

        // We probably do not want to fail if analytics fails
        let analytics = common_systems
            .analytics()
            .compute_analytics(history, &self.inner.prev_documents)?;
        common_systems.database().save_analytics(&analytics)?;

        rerank_documents(common_systems, history, documents, &user_interests)
    }
}

impl<CS> Reranker<CS>
where
    CS: CommonSystems,
{
    fn new(common_systems: CS) -> Result<Self, Error> {
        // load the correct state from the database
        let inner = common_systems
            .database()
            .load_prev_documents_full()
            .and_then(|prev_documents| {
                if let Some(prev_documents) = prev_documents {
                    // if we have documents with all the data
                    // we need to have the user interests
                    common_systems
                        .database()
                        .load_user_interests()?
                        .map_or_else(
                            // We could go to InitUserInterests instead and try to recover from that
                            || Err(Error {}),
                            |user_interests| {
                                Ok(RerankerInner::Nominal(RerankerState {
                                    inner: Nominal {
                                        prev_documents,
                                        user_interests,
                                    },
                                }))
                            },
                        )
                } else {
                    // if we have document with the embedding we need to init the  interest
                    Ok(common_systems
                        .database()
                        .load_prev_documents()?
                        .map_or_else(
                            || RerankerInner::Empty,
                            |prev_documents| {
                                RerankerInner::InitUserInterests(RerankerState {
                                    inner: InitUserInterests { prev_documents },
                                })
                            },
                        ))
                }
            })?;

        Ok(Self {
            common_systems,
            inner,
        })
    }
}

fn make_documents_with_embedding<BS>(
    bert_system: &BS,
    documents: &[Document],
) -> Result<Vec<DocumentDataWithEmbedding>, Error>
where
    BS: BertSystem + ?Sized,
{
    let prev_documents: Vec<_> = documents
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
    bert_system.compute_embedding(&prev_documents)
}

fn to_init_user_interests<CS>(
    common_systems: &CS,
    documents: &[Document],
) -> Result<RerankerInner, Error>
where
    CS: CommonSystems,
{
    let prev_documents = make_documents_with_embedding(common_systems.bert(), &documents)?;

    common_systems
        .database()
        .save_prev_documents(&prev_documents)?;

    let inner = RerankerState::<InitUserInterests> {
        inner: InitUserInterests { prev_documents },
    };

    Ok(RerankerInner::InitUserInterests(inner))
}

fn rank_from_source(documents: &[Document]) -> DocumentsRank {
    documents.iter().map(|document| document.rank).collect()
}

fn rerank_documents<CS>(
    common_systems: &CS,
    history: &[DocumentHistory],
    documents: &[Document],
    user_interests: &UserInterests,
) -> Result<(RerankerInner, DocumentsRank), Error>
where
    CS: CommonSystems,
{
    let documents = make_documents_with_embedding(common_systems.bert(), &documents)?;
    let documents = common_systems
        .coi()
        .compute_coi(&documents, user_interests)?;
    let documents = common_systems.ltr().compute_ltr(history, &documents)?;
    let documents = common_systems.context().compute_context(&documents)?;
    let (documents, user_interests) = common_systems
        .mab()
        .compute_mab(&documents, user_interests)?;

    let database = common_systems.database();
    // What should we do if we can save one but not the other?
    database.save_user_interests(&user_interests)?;
    database.save_prev_documents_full(&documents)?;

    let ranks = documents.iter().map(|document| document.mab.rank).collect();

    let inner = RerankerInner::Nominal(RerankerState {
        inner: Nominal {
            prev_documents: documents,
            user_interests,
        },
    });

    Ok((inner, ranks))
}
