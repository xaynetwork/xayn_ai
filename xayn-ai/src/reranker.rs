#![allow(dead_code)]

use crate::{
    database::Database,
    document::Document,
    document_data::{DocumentComponent, DocumentDataState, WithDocument, WithMab},
};

type DocumentsRank = Vec<usize>;

// place holders to move in their respective module
enum CenterOfInterest {}
pub enum DocumentHistory {}
pub struct Error {}
pub struct Analytics {}
pub struct CentersOfInterest {
    positive: Vec<CenterOfInterest>,
    negative: Vec<CenterOfInterest>,
}

/// Common systems that we need in the reranker
pub trait CommonSystems {
    fn database(&self) -> &dyn Database;
    // bert
    // center of intereset
    // ltr
    // context
    // mab
}

/// Empty state
struct Empty {}

/// In this state we have the documents from the previous query but we do not have center of interest
struct InitCentersOfInterest {
    prev_documents: Vec<DocumentDataState<WithDocument>>,
}

/// In this state we have all the data we need to do reranking and run the feedback loop
struct Nominal {
    prev_documents: Vec<DocumentDataState<WithMab>>,
    centers_of_interest: CentersOfInterest,
}

struct RerankerState<S> {
    inner: S,
}

enum RerankerInner {
    Empty(RerankerState<Empty>),
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
            RerankerInner::Empty(state) => state.rerank(common_systems, history, documents),
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
    fn new() -> Self {
        Self { inner: Empty {} }
    }

    fn rerank<CS>(
        self,
        common_systems: &CS,
        _history: &[DocumentHistory],
        documents: &[Document],
    ) -> Result<(RerankerInner, DocumentsRank), Error>
    where
        CS: CommonSystems,
    {
        Ok((
            to_init_centers_of_interest(common_systems, documents)?,
            rank_from_source(documents)?,
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
        // try to create new coi from center of interest system
        let centers_of_interest = None;

        match centers_of_interest {
            None => Ok((
                to_init_centers_of_interest(common_systems, documents)?,
                rank_from_source(documents)?,
            )),
            Some(centers_of_interest) => {
                rerank(common_systems, history, documents, centers_of_interest)
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
        // update centers of interest
        let centers_of_interest = CentersOfInterest {
            positive: vec![],
            negative: vec![],
        };

        common_systems
            .database()
            .save_centers_of_interest(&centers_of_interest)?;

        let analytics = Analytics {};

        common_systems.database().save_analytics(&analytics)?;

        rerank(common_systems, history, documents, &centers_of_interest)
    }
}

impl<CS> Reranker<CS>
where
    CS: CommonSystems,
{
    fn new(common_systems: CS) -> Result<Self, Error> {
        let inner = common_systems
            .database()
            .load_prev_documents_full()
            .and_then(|prev_documents| {
                if let Some(prev_documents) = prev_documents {
                    common_systems
                        .database()
                        .load_centers_of_interest()?
                        .map(|centers_of_interest| {
                            Ok(RerankerInner::Nominal(RerankerState {
                                inner: Nominal {
                                    prev_documents,
                                    centers_of_interest,
                                },
                            }))
                        })
                        .unwrap_or_else(|| Err(Error {}))
                } else {
                    Ok(common_systems
                        .database()
                        .load_prev_documents()?
                        .map(|prev_documents| {
                            RerankerInner::InitCentersOfInterest(RerankerState {
                                inner: InitCentersOfInterest { prev_documents },
                            })
                        })
                        .unwrap_or_else(|| RerankerInner::Empty(RerankerState { inner: Empty {} })))
                }
            })?;

        Ok(Self {
            common_systems,
            inner,
        })
    }
}

fn to_init_centers_of_interest<CS>(
    common_systems: &CS,
    documents: &[Document],
) -> Result<RerankerInner, Error>
where
    CS: CommonSystems,
{
    let prev_documents: Vec<_> = documents
        .iter()
        .map(|document| DocumentComponent {
            id: document.id.clone(),
            snippet: document.snippet.clone(),
        })
        .map(DocumentDataState::<WithDocument>::new)
        .collect();

    // persist prev_documents
    common_systems
        .database()
        .save_prev_documents(&prev_documents)?;

    let inner = RerankerState::<InitCentersOfInterest> {
        inner: InitCentersOfInterest { prev_documents },
    };

    Ok(RerankerInner::InitCentersOfInterest(inner))
}

fn rank_from_source(documents: &[Document]) -> Result<DocumentsRank, Error> {
    Ok(documents.iter().map(|document| document.rank).collect())
}

fn rerank<CS>(
    _common_systems: &CS,
    _history: &[DocumentHistory],
    _documents: &[Document],
    _centers_of_interest: &CentersOfInterest,
) -> Result<(RerankerInner, DocumentsRank), Error>
where
    CS: CommonSystems,
{
    // Thiss will return a RerankerInner::Nominal
    unimplemented!()
}
