use std::time::Duration;

use crate::{
    analytics::Analytics,
    coi::point::UserInterests,
    data::{
        document::{DocumentHistory, DocumentId},
        document_data::{
            CoiComponent,
            DocumentDataWithCoi,
            DocumentDataWithContext,
            DocumentDataWithDocument,
            DocumentDataWithLtr,
            DocumentDataWithQAMBert,
            DocumentDataWithRank,
            DocumentDataWithSMBert,
            SMBertComponent,
        },
    },
    error::Error,
    reranker::database::Database,
};

#[cfg(test)]
use mockall::automock;

#[cfg_attr(test, automock)]
#[allow(clippy::upper_case_acronyms)]
pub(crate) trait SMBertSystem {
    fn compute_embedding(
        &self,
        documents: &[DocumentDataWithDocument],
    ) -> Result<Vec<DocumentDataWithSMBert>, Error>;
}

#[cfg_attr(test, automock)]
#[allow(clippy::upper_case_acronyms)]
pub(crate) trait QAMBertSystem {
    fn compute_similarity(
        &self,
        documents: &[DocumentDataWithCoi],
    ) -> Result<Vec<DocumentDataWithQAMBert>, Error>;
}

pub(crate) trait CoiSystemData {
    fn id(&self) -> DocumentId;
    fn smbert(&self) -> &SMBertComponent;
    fn coi(&self) -> Option<&CoiComponent>;
    fn viewed(&self) -> Duration;
}

#[cfg_attr(test, automock)]
pub(crate) trait CoiSystem {
    /// Add centre of interest information to a document
    fn compute_coi(
        &self,
        documents: &[DocumentDataWithSMBert],
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
        documents: &[DocumentDataWithQAMBert],
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
pub(crate) trait AnalyticsSystem {
    fn compute_analytics(
        &self,
        history: &[DocumentHistory],
        documents: &[DocumentDataWithRank],
    ) -> Result<Analytics, Error>;
}

/// Common systems that we need in the reranker
/// At the moment this exists only to avoid to have 7+ generics around
pub(crate) trait CommonSystems {
    fn database(&self) -> &dyn Database;
    fn smbert(&self) -> &dyn SMBertSystem;
    fn qambert(&self) -> &dyn QAMBertSystem;
    fn coi(&self) -> &dyn CoiSystem;
    fn ltr(&self) -> &dyn LtrSystem;
    fn context(&self) -> &dyn ContextSystem;
    fn analytics(&self) -> &dyn AnalyticsSystem;
}
