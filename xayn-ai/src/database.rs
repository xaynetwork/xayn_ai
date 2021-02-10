use crate::{
    document_data::{DocumentDataState, WithDocument, WithMab},
    reranker::{CentersOfInterest, Error, Analytics},
};

pub trait Database {
    fn save_centers_of_interest(
        &self,
        centers_of_interest: &CentersOfInterest,
    ) -> Result<(), Error>;

    fn load_centers_of_interest(
        &self,
    ) -> Result<Option<CentersOfInterest>, Error>;

    fn save_prev_documents(
        &self,
        prev_documents: &Vec<DocumentDataState<WithDocument>>,
    ) -> Result<(), Error>;

    fn load_prev_documents(
        &self,
    ) -> Result<Option<Vec<DocumentDataState<WithDocument>>>, Error>;

    fn save_prev_documents_full(
        &self,
        prev_documents: &Vec<DocumentDataState<WithMab>>,
    ) -> Result<(), Error>;

    fn load_prev_documents_full(
        &self,
    ) -> Result<Option<Vec<DocumentDataState<WithMab>>>, Error>;

    fn save_analytics(&self, analytics: &Analytics) -> Result<(), Error>;
}
