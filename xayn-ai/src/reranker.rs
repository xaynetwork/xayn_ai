use crate::{
    data::{
        document::{Document, DocumentHistory},
        document_data::{
            DocumentContentComponent,
            DocumentDataWithDocument,
            DocumentDataWithMab,
            DocumentIdComponent,
        },
        UserInterests,
    },
    error::Error,
    reranker_systems::CommonSystems,
};

pub type DocumentsRank = Vec<usize>;

/// Update AI
fn feedback_loop<CS>(
    common_systems: &CS,
    history: &[DocumentHistory],
    prev_documents: &[DocumentDataWithMab],
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
fn analytics<CS>(
    common_systems: &CS,
    history: &[DocumentHistory],
    prev_documents: &[DocumentDataWithMab],
) -> Result<(), Error>
where
    CS: CommonSystems,
{
    common_systems
        .analytics()
        .compute_analytics(history, &prev_documents)
        .and_then(|analytics| common_systems.database().save_analytics(&analytics))
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

    let documents = common_systems.bert().compute_embedding(documents)?;
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

#[derive(Default)]
pub struct RerankerData {
    user_interests: UserInterests,
    prev_documents: Vec<DocumentDataWithMab>,
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
        // load the correct state from the database
        let data = common_systems
            .database()
            .load_state()?
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

    pub fn rerank(mut self, history: &[DocumentHistory], documents: &[Document]) -> DocumentsRank {
        // The number of errors it can contains is very limited, using clear we avoid
        // to reallocate the vector each time.
        self.errors.clear();

        if !history.is_empty() && !self.data.prev_documents.is_empty() {
            let user_interests = self.data.user_interests.clone();

            match feedback_loop(
                &self.common_systems,
                history,
                &self.data.prev_documents,
                user_interests,
            ) {
                Ok(user_interests) => self.data.user_interests = user_interests,
                Err(e) => self.errors.push(e),
            };

            if let Err(e) = analytics(&self.common_systems, history, &self.data.prev_documents) {
                self.errors.push(e);
            }
        }

        let user_interests = self.data.user_interests.clone();

        rerank(&self.common_systems, history, documents, user_interests)
            .map(|(prev_documents, user_interests, rank)| {
                self.data.prev_documents = prev_documents;
                self.data.user_interests = user_interests;

                rank
            })
            .unwrap_or_else(|e| {
                self.errors.push(e);

                // in case of error we don't have documents
                self.data.prev_documents.clear();

                documents.iter().map(|document| document.rank).collect()
            })
    }
}
