use crate::{analytics::Analytics, error::Error, reranker::RerankerData};

#[cfg_attr(test, mockall::automock)]
pub trait Database {
    fn save_data(&self, state: &RerankerData) -> Result<(), Error>;

    fn load_data(&self) -> Result<Option<RerankerData>, Error>;

    fn save_analytics(&self, analytics: &Analytics) -> Result<(), Error>;
}
