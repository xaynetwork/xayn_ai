use std::cell::RefCell;

use crate::{
    reranker::{database::Database, RerankerData},
    Error,
};

pub(crate) struct MemDb {
    data: RefCell<Option<RerankerData>>,
}

impl MemDb {
    pub(crate) fn new() -> Self {
        Self {
            data: RefCell::new(None),
        }
    }

    pub(crate) fn from_data(data: RerankerData) -> Self {
        Self {
            data: RefCell::new(data.into()),
        }
    }
}

impl Database for MemDb {
    fn serialize_data(&self, _data: &RerankerData) -> Result<Vec<u8>, Error> {
        unimplemented!("mocked database does not have a serialized representation")
    }

    fn load_data(&self) -> Result<Option<RerankerData>, Error> {
        Ok(self.data.borrow().clone())
    }
}
