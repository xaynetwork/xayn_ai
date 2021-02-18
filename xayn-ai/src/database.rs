use crate::{
    data::{
        document_data::{DocumentDataWithEmbedding, DocumentDataWithMab},
        Analytics,
        UserInterests,
    },
    error::Error,
};

pub trait Database {
    fn save_user_interests(&self, user_interests: &UserInterests) -> Result<(), Error>;

    fn load_user_interests(&self) -> Result<Option<UserInterests>, Error>;

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
