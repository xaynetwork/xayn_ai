use crate::{data::Analytics, error::Error, reranker::RerankerData};

pub trait Database {
    fn save_state(&self, state: &RerankerData) -> Result<(), Error>;
    fn load_state(&self) -> Result<Option<RerankerData>, Error>;

    fn save_analytics(&self, analytics: &Analytics) -> Result<(), Error>;
}
