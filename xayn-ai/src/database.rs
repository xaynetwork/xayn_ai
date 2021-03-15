use crate::{data::Analytics, error::Error, reranker::RerankerState};

pub trait Database {
    fn save_state(&self, state: &RerankerState) -> Result<(), Error>;
    fn load_state(&self) -> Result<Option<RerankerState>, Error>;

    fn save_analytics(&self, analytics: &Analytics) -> Result<(), Error>;
}
