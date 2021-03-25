use crate::{
    data::{
        document::{Document, DocumentHistory},
        document_data::{
            DocumentContentComponent, DocumentDataWithDocument, DocumentDataWithEmbedding,
            DocumentDataWithMab, DocumentIdComponent,
        },
        UserInterests,
    },
    error::Error,
    reranker_systems::{CoiSystemData, CommonSystems},
    to_vec_of_ref_of,
};

pub type DocumentsRank = Vec<usize>;

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
) -> Result<(), Error>
where
    CS: CommonSystems,
{
    common_systems
        .analytics()
        .compute_analytics(history, prev_documents)
        .and_then(|analytics| common_systems.database().save_analytics(&analytics))
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

    let rank = documents.iter().map(|document| document.mab.rank).collect();

    Ok((documents, user_interests, rank))
}

pub enum PreviousDocuments {
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

    fn is_empty(&self) -> bool {
        match self {
            PreviousDocuments::Embedding(documents) => documents.is_empty(),
            PreviousDocuments::Mab(documents) => documents.is_empty(),
        }
    }
}

#[derive(Default)]
pub struct RerankerData {
    user_interests: UserInterests,
    prev_documents: PreviousDocuments,
}

pub struct Reranker<CS> {
    common_systems: CS,
    data: RerankerData,
    errors: Vec<Error>,
}

impl<CS> Reranker<CS>
where
    CS: CommonSystems,
{
    pub fn new(common_systems: CS) -> Result<Self, Error> {
        // load the last valid state from the database
        let data = common_systems
            .database()
            .load_data()?
            .unwrap_or_else(RerankerData::default);

        Ok(Self {
            common_systems,
            data,
            errors: Vec::new(),
        })
    }

    pub fn errors(&self) -> &Vec<Error> {
        &self.errors
    }

    pub fn rerank(&mut self, history: &[DocumentHistory], documents: &[Document]) -> DocumentsRank {
        // The number of errors it can contain is very limited. By using `clear` we avoid
        // re-allocating the vector on each method call.
        self.errors.clear();

        if !history.is_empty() && !self.data.prev_documents.is_empty() {
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
                if let Err(e) = collect_analytics(&self.common_systems, history, &prev_documents) {
                    self.errors.push(e);
                }
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

                documents.iter().map(|document| document.rank).collect()
            })
    }
}
