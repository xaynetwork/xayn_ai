use ndarray::arr1;
#[cfg(feature = "multithreaded")]
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::{
    data::document_data::{DocumentDataWithDocument, DocumentDataWithSMBert, SMBertComponent},
    error::Error,
    reranker::systems::SMBertSystem,
};
use rubert::SMBert;

impl SMBertSystem for SMBert {
    fn compute_embedding(
        &self,
        documents: &[DocumentDataWithDocument],
    ) -> Result<Vec<DocumentDataWithSMBert>, Error> {
        #[cfg(not(feature = "multithreaded"))]
        let documents = documents.iter();
        #[cfg(feature = "multithreaded")]
        let documents = documents.into_par_iter();

        documents
            .map(|document| {
                let embedding = self.run(document.document_content.title.as_str());
                embedding
                    .map(|embedding| {
                        DocumentDataWithSMBert::from_document(
                            document,
                            SMBertComponent { embedding },
                        )
                    })
                    .map_err(Into::into)
            })
            .collect()
    }
}

/// SMBert system to run when SMBert is disabled
#[allow(clippy::upper_case_acronyms)]
pub(crate) struct NeutralSMBert;

impl SMBertSystem for NeutralSMBert {
    fn compute_embedding(
        &self,
        documents: &[DocumentDataWithDocument],
    ) -> Result<Vec<DocumentDataWithSMBert>, Error> {
        Ok(documents
            .iter()
            .map(|document| {
                DocumentDataWithSMBert::from_document(
                    document,
                    SMBertComponent {
                        embedding: arr1(&[]).into(),
                    },
                )
            })
            .collect())
    }
}
