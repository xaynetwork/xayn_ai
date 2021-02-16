#![allow(dead_code)]

use crate::{
    data::{
        document::{Document, DocumentHistory},
        document_data::{
            DocumentDataWithCenterOfInterest, DocumentDataWithContext,
            DocumentDataWithDocument, DocumentDataWithEmbedding, DocumentDataWithLtr,
            DocumentDataWithMab
        },
        Analytics, CentersOfInterest,
    },
    database::Database,
    error::Error,
};

pub trait BertSystem {
    fn add_embedding(
        &self,
        documents: &[DocumentDataWithDocument],
    ) -> Result<Vec<DocumentDataWithEmbedding>, Error>;
}

pub trait CenterOfInterestSystem {
    /// Add center of interest information to a document
    fn add_center_of_interest(
        &self,
        documents: &[DocumentDataWithEmbedding],
        centers_of_interest: &CentersOfInterest,
    ) -> Result<Vec<DocumentDataWithCenterOfInterest>, Error>;

    /// Make new centers of interest from history and documents
    fn make_centers_of_interest(
        &self,
        history: &[DocumentHistory],
        documents: &[DocumentDataWithEmbedding],
    ) -> Result<Option<CentersOfInterest>, Error>;

    /// Update centers of interest from history and documents
    fn update_centers_of_interest(
        &self,
        history: &[DocumentHistory],
        documents: &[Document],
        centers_of_interest: &CentersOfInterest,
    ) -> Result<CentersOfInterest, Error>;
}

pub trait LtrSystem {
    fn add_ltr(
        &self,
        history: &[DocumentHistory],
        documents: &[DocumentDataWithCenterOfInterest],
    ) -> Result<Vec<DocumentDataWithLtr>, Error>;
}

pub trait ContextSystem {
    fn add_context(
        &self,
        documents: &[DocumentDataWithLtr],
    ) -> Result<Vec<DocumentDataWithContext>, Error>;
}

pub trait MabSystem {
    fn add_mab(
        &self,
        documents: &[DocumentDataWithContext],
        centers_of_interest: &CentersOfInterest,
    ) -> Result<(Vec<DocumentDataWithMab>, CentersOfInterest), Error>;
}

pub trait AnalyticsSystem {
    fn gen_analytics(
        &self,
        history: &[DocumentHistory],
        documents: &[DocumentDataWithMab],
    ) -> Result<Analytics, Error>;
}

/// Common systems that we need in the reranker
/// At the moment this exists only to avoid to have 7+ generics around
pub trait CommonSystems {
    fn database(&self) -> &dyn Database;
    fn bert(&self) -> &dyn BertSystem;
    fn centers_of_interest(&self) -> &dyn CenterOfInterestSystem;
    fn ltr(&self) -> &dyn LtrSystem;
    fn context(&self) -> &dyn ContextSystem;
    fn mab(&self) -> &dyn MabSystem;
    fn analytics(&self) -> &dyn AnalyticsSystem;
}
