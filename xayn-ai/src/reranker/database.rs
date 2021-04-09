use std::{cell::RefCell, collections::HashMap};

use crate::error::Error;

use super::RerankerData;

pub trait DatabaseRaw {
    fn get(&self, key: impl AsRef<[u8]>) -> Result<Option<Vec<u8>>, Error>;

    fn insert(&self, key: impl AsRef<[u8]>, value: impl AsRef<[u8]>) -> Result<(), Error>;

    fn delete(&self, key: impl AsRef<[u8]>) -> Result<(), Error>;
}

#[cfg_attr(test, mockall::automock)]
pub(crate) trait Database {
    fn save_data(&self, state: &RerankerData) -> Result<(), Error>;

    fn load_data(&self) -> Result<Option<RerankerData>, Error>;
}

pub(super) struct Db<DbRaw>(DbRaw);

impl<DbRaw> Db<DbRaw> {
    pub fn new(db_raw: DbRaw) -> Self {
        Self(db_raw)
    }
}

const CURRENT_SCHEMA_VERSION: u8 = 0;

impl<DbRaw> Database for Db<DbRaw>
where
    DbRaw: DatabaseRaw,
{
    fn save_data(&self, state: &RerankerData) -> Result<(), Error> {
        let key = reranker_data_key(CURRENT_SCHEMA_VERSION);
        let value = bincode::serialize(state)?;

        self.0.insert(key, value)
    }

    fn load_data(&self) -> Result<Option<RerankerData>, Error> {
        let key = reranker_data_key(CURRENT_SCHEMA_VERSION);

        self.0
            .get(key)?
            .map(|bs| bincode::deserialize(&bs))
            .transpose()
            .map_err(|e| e.into())
    }
}

fn reranker_data_key(version: u8) -> Vec<u8> {
    let key_prefix: &[u8] = b"reranker_data";

    [key_prefix, &[version]].concat()
}

/// A temporary dummy database.
#[derive(Default)]
pub struct InMemoryDatabaseRaw(RefCell<HashMap<Vec<u8>, Vec<u8>>>);

impl DatabaseRaw for InMemoryDatabaseRaw {
    fn get(&self, key: impl AsRef<[u8]>) -> Result<Option<Vec<u8>>, Error> {
        Ok(self.0.borrow().get(&key.as_ref().to_vec()).cloned())
    }

    fn insert(&self, key: impl AsRef<[u8]>, value: impl AsRef<[u8]>) -> Result<(), Error> {
        self.0
            .borrow_mut()
            .insert(key.as_ref().to_vec(), value.as_ref().to_vec());

        Ok(())
    }

    fn delete(&self, key: impl AsRef<[u8]>) -> Result<(), Error> {
        self.0.borrow_mut().remove(&key.as_ref().to_vec());

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        data::UserInterests,
        tests::{cois_from_words, data_with_mab, mocked_bert_system},
    };

    #[test]
    fn test_database_save_load() {
        let cois = cois_from_words(&["a", "b", "c"], mocked_bert_system());
        let user_interests = UserInterests {
            positive: cois.clone(),
            negative: cois,
        };
        let docs = data_with_mab(vec![(0, vec![1.; 128])].into_iter());
        let data = RerankerData::new_with_mab(user_interests, docs);

        let database = Db::new(InMemoryDatabaseRaw::default());
        database.save_data(&data).expect("saving data");
        let loaded_data = database
            .load_data()
            .expect("load data")
            .expect("loaded data");

        assert_eq!(data, loaded_data);
    }
}
