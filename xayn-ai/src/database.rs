use crate::{
    document_data::{DocumentDataWithEmbedding, DocumentDataWithMab},
    error::Error,
    reranker::{Analytics, CentersOfInterest},
};

pub trait Database {
    fn save_centers_of_interest(
        &self,
        centers_of_interest: &CentersOfInterest,
    ) -> Result<(), Error>;

    fn load_centers_of_interest(&self) -> Result<Option<CentersOfInterest>, Error>;

    fn save_prev_documents(
        &self,
        prev_documents: &[DocumentDataWithEmbedding],
    ) -> Result<(), Error>;

    fn load_prev_documents(&self) -> Result<Option<Vec<DocumentDataWithEmbedding>>, Error>;

    fn save_prev_documents_full(&self, prev_documents: &[DocumentDataWithMab])
        -> Result<(), Error>;

    fn load_prev_documents_full(&self) -> Result<Option<Vec<DocumentDataWithMab>>, Error>;

    fn save_analytics(&self, analytics: &Analytics) -> Result<(), Error>;
}
