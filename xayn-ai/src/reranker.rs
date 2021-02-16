#![allow(dead_code)]

use crate::{
    data::{
        document::{Document, DocumentHistory},
        document_data::{
            DocumentContentComponent, DocumentDataWithDocument, DocumentDataWithEmbedding,
            DocumentDataWithMab, DocumentIdComponent,
        },
        CentersOfInterest,
    },
    error::Error,
    reranker_systems::{BertSystem, CommonSystems},
};

pub type DocumentsRank = Vec<usize>;

/// Empty state
enum Empty {}

/// In this state we have the documents from the previous query but we do not have center of interest
struct InitCentersOfInterest {
    prev_documents: Vec<DocumentDataWithEmbedding>,
}

/// In this state we have all the data we need to do reranking and run the feedback loop
struct Nominal {
    prev_documents: Vec<DocumentDataWithMab>,
    centers_of_interest: CentersOfInterest,
}

struct RerankerState<S> {
    inner: S,
}

enum RerankerInner {
    Empty,
    InitCentersOfInterest(RerankerState<InitCentersOfInterest>),
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
            RerankerInner::InitCentersOfInterest(state) => {
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
            to_init_centers_of_interest(common_systems, documents)?,
            rank_from_source(documents),
        ))
    }
}

impl RerankerState<InitCentersOfInterest> {
    fn rerank<CS>(
        self,
        common_systems: &CS,
        history: &[DocumentHistory],
        documents: &[Document],
    ) -> Result<(RerankerInner, DocumentsRank), Error>
    where
        CS: CommonSystems,
    {
        let centers_of_interest = common_systems
            .centers_of_interest()
            .make_centers_of_interest(history, &self.inner.prev_documents)?;

        match centers_of_interest {
            None => Ok((
                to_init_centers_of_interest(common_systems, documents)?,
                rank_from_source(documents),
            )),
            Some(centers_of_interest) => {
                rerank(common_systems, history, documents, &centers_of_interest)
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
        let centers_of_interest = common_systems
            .centers_of_interest()
            .update_centers_of_interest(history, documents, &self.inner.centers_of_interest)?;

        common_systems
            .database()
            .save_centers_of_interest(&centers_of_interest)?;

        // We probably do not want to fail if analytics fails
        let analytics = common_systems
            .analytics()
            .gen_analytics(history, &self.inner.prev_documents)?;
        common_systems.database().save_analytics(&analytics)?;

        rerank(common_systems, history, documents, &centers_of_interest)
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
                    // we need to have the center of interest
                    common_systems
                        .database()
                        .load_centers_of_interest()?
                        .map_or_else(
                            // We could go to InitCentersOfInterest instead and try to recover from that
                            || Err(Error {}),
                            |centers_of_interest| {
                                Ok(RerankerInner::Nominal(RerankerState {
                                    inner: Nominal {
                                        prev_documents,
                                        centers_of_interest,
                                    },
                                }))
                            },
                        )
                } else {
                    // if we have document with the embedding we need to init the center of interest
                    Ok(common_systems
                        .database()
                        .load_prev_documents()?
                        .map_or_else(
                            || RerankerInner::Empty,
                            |prev_documents| {
                                RerankerInner::InitCentersOfInterest(RerankerState {
                                    inner: InitCentersOfInterest { prev_documents },
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
    bert_system.add_embedding(&prev_documents)
}

fn to_init_centers_of_interest<CS>(
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

    let inner = RerankerState::<InitCentersOfInterest> {
        inner: InitCentersOfInterest { prev_documents },
    };

    Ok(RerankerInner::InitCentersOfInterest(inner))
}

fn rank_from_source(documents: &[Document]) -> DocumentsRank {
    documents.iter().map(|document| document.rank).collect()
}

fn rerank<CS>(
    common_systems: &CS,
    history: &[DocumentHistory],
    documents: &[Document],
    centers_of_interest: &CentersOfInterest,
) -> Result<(RerankerInner, DocumentsRank), Error>
where
    CS: CommonSystems,
{
    let documents = make_documents_with_embedding(common_systems.bert(), &documents)?;
    let documents = common_systems
        .centers_of_interest()
        .add_center_of_interest(&documents, centers_of_interest)?;
    let documents = common_systems.ltr().add_ltr(history, &documents)?;
    let documents = common_systems.context().add_context(&documents)?;
    let (documents, centers_of_interest) = common_systems
        .mab()
        .add_mab(&documents, centers_of_interest)?;

    let database = common_systems.database();
    // What should we do if we can save one but not the other?
    database.save_centers_of_interest(&centers_of_interest)?;
    database.save_prev_documents_full(&documents)?;

    let ranks = documents.iter().map(|document| document.mab.rank).collect();

    let inner = RerankerInner::Nominal(RerankerState {
        inner: Nominal {
            prev_documents: documents,
            centers_of_interest,
        },
    });

    Ok((inner, ranks))
}
