use std::cell::RefCell;

use crate::{analytics::Analytics, database::Database, reranker::RerankerData, Error};

pub struct MemDb {
    data: RefCell<Option<RerankerData>>,
}

impl MemDb {
    pub fn new() -> Self {
        Self {
            data: RefCell::new(None),
        }
    }

    pub fn from_data(data: RerankerData) -> Self {
        Self {
            data: RefCell::new(data.into()),
        }
    }
}

impl Database for MemDb {
    fn save_data(&self, state: &RerankerData) -> Result<(), Error> {
        *self.data.borrow_mut() = Some(state.clone());
        Ok(())
    }

    fn load_data(&self) -> Result<Option<RerankerData>, Error> {
        Ok(self.data.borrow().clone())
    }

    fn save_analytics(&self, _analytics: &Analytics) -> Result<(), Error> {
        Ok(())
    }
}
