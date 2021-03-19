use crate::{
    analytics::Analytics,
    data::{
        document::DocumentHistory,
        document_data::{
            DocumentDataWithCoi,
            DocumentDataWithContext,
            DocumentDataWithDocument,
            DocumentDataWithEmbedding,
            DocumentDataWithLtr,
            DocumentDataWithMab,
        },
        UserInterests,
    },
    database::Database,
    error::Error,
};

pub trait BertSystem {
    fn compute_embedding(
        &self,
        documents: Vec<DocumentDataWithDocument>,
    ) -> Result<Vec<DocumentDataWithEmbedding>, Error>;
}

pub trait CoiSystem {
    /// Add centre of interest information to a document
    fn compute_coi(
        &self,
        documents: Vec<DocumentDataWithEmbedding>,
        user_interests: &UserInterests,
    ) -> Result<Vec<DocumentDataWithCoi>, Error>;

    /// Update cois from history and documents
    fn update_user_interests(
        &self,
        history: &[DocumentHistory],
        documents: &[DocumentDataWithMab],
        user_interests: UserInterests,
    ) -> Result<UserInterests, Error>;
}

pub trait LtrSystem {
    fn compute_ltr(
        &self,
        history: &[DocumentHistory],
        documents: Vec<DocumentDataWithCoi>,
    ) -> Result<Vec<DocumentDataWithLtr>, Error>;
}

pub trait ContextSystem {
    fn compute_context(
        &self,
        documents: Vec<DocumentDataWithLtr>,
    ) -> Result<Vec<DocumentDataWithContext>, Error>;
}

pub trait MabSystem {
    fn compute_mab(
        &self,
        documents: Vec<DocumentDataWithContext>,
        user_interests: UserInterests,
    ) -> Result<(Vec<DocumentDataWithMab>, UserInterests), Error>;
}

pub trait AnalyticsSystem {
    fn compute_analytics(
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
    fn coi(&self) -> &dyn CoiSystem;
    fn ltr(&self) -> &dyn LtrSystem;
    fn context(&self) -> &dyn ContextSystem;
    fn mab(&self) -> &dyn MabSystem;
    fn analytics(&self) -> &dyn AnalyticsSystem;
}
