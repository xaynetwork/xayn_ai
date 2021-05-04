use crate::{
    analytics::Analytics,
    data::{
        document::{DocumentHistory, DocumentId},
        document_data::{
            CoiComponent,
            DocumentDataWithCoi,
            DocumentDataWithContext,
            DocumentDataWithDocument,
            DocumentDataWithSMBert,
            DocumentDataWithLtr,
            DocumentDataWithMab,
            SMBertEmbeddingComponent,
        },
        UserInterests,
    },
    error::Error,
    reranker::database::Database,
};

#[cfg(test)]
use mockall::automock;

#[cfg_attr(test, automock)]
pub(crate) trait SMBertSystem {
    fn compute_embedding(
        &self,
        documents: Vec<DocumentDataWithDocument>,
    ) -> Result<Vec<DocumentDataWithSMBert>, Error>;
}

pub(crate) trait CoiSystemData {
    fn id(&self) -> &DocumentId;
    fn embedding(&self) -> &SMBertEmbeddingComponent;
    fn coi(&self) -> Option<&CoiComponent>;
}

#[cfg_attr(test, automock)]
pub(crate) trait CoiSystem {
    /// Add centre of interest information to a document
    fn compute_coi(
        &self,
        documents: Vec<DocumentDataWithSMBert>,
        user_interests: &UserInterests,
    ) -> Result<Vec<DocumentDataWithCoi>, Error>;

    /// Update cois from history and documents
    fn update_user_interests<'a>(
        &self,
        history: &[DocumentHistory],
        documents: &[&'a dyn CoiSystemData],
        user_interests: UserInterests,
    ) -> Result<UserInterests, Error>;
}

#[cfg_attr(test, automock)]
pub(crate) trait LtrSystem {
    fn compute_ltr(
        &self,
        history: &[DocumentHistory],
        documents: Vec<DocumentDataWithCoi>,
    ) -> Result<Vec<DocumentDataWithLtr>, Error>;
}

#[cfg_attr(test, automock)]
pub(crate) trait ContextSystem {
    fn compute_context(
        &self,
        documents: Vec<DocumentDataWithLtr>,
    ) -> Result<Vec<DocumentDataWithContext>, Error>;
}

#[cfg_attr(test, automock)]
pub(crate) trait MabSystem {
    fn compute_mab(
        &self,
        documents: Vec<DocumentDataWithContext>,
        user_interests: UserInterests,
    ) -> Result<(Vec<DocumentDataWithMab>, UserInterests), Error>;
}

#[cfg_attr(test, automock)]
pub(crate) trait AnalyticsSystem {
    fn compute_analytics(
        &self,
        history: &[DocumentHistory],
        documents: &[DocumentDataWithMab],
    ) -> Result<Analytics, Error>;
}

/// Common systems that we need in the reranker
/// At the moment this exists only to avoid to have 7+ generics around
pub(crate) trait CommonSystems {
    fn database(&self) -> &dyn Database;
    fn smbert(&self) -> &dyn SMBertSystem;
    fn coi(&self) -> &dyn CoiSystem;
    fn ltr(&self) -> &dyn LtrSystem;
    fn context(&self) -> &dyn ContextSystem;
    fn mab(&self) -> &dyn MabSystem;
    fn analytics(&self) -> &dyn AnalyticsSystem;
}
