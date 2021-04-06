use crate::{error::Error, reranker::RerankerData};

#[cfg_attr(test, mockall::automock)]
pub trait Database {
    fn save_data(&self, state: &RerankerData) -> Result<(), Error>;

    fn load_data(&self) -> Result<Option<RerankerData>, Error>;
}

pub trait DatabaseRaw {
    fn get<K>(&self, key: K) -> Result<Option<Vec<u8>>, Error>
    where
        K: AsRef<[u8]>;

    fn put<K, V>(&self, key: K, value: V) -> Result<(), Error>
    where
        K: AsRef<[u8]>,
        V: AsRef<[u8]>;

    fn delete<K>(&self, key: K) -> Result<(), Error>
    where
        K: AsRef<[u8]>;
}

pub struct Db<DbRaw> {
    db_raw: DbRaw,
}

impl<DbRaw> Db<DbRaw>
where
    DbRaw: DatabaseRaw,
{
    pub fn new(db_raw: DbRaw) -> Self {
        Self { db_raw }
    }
}

impl<DbRaw> Database for Db<DbRaw>
where
    DbRaw: DatabaseRaw,
{
    fn save_data(&self, state: &RerankerData) -> Result<(), Error> {
        let key = reranker_data_key(0);
        let value = bincode::serialize(state)?;

        self.db_raw.put(key, value)
    }

    fn load_data(&self) -> Result<Option<RerankerData>, Error> {
        let key = reranker_data_key(0);

        self.db_raw
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        data::UserInterests,
        reranker::PreviousDocuments,
        tests::{cois_from_words, data_with_mab, mocked_bert_system},
    };

    use std::{cell::RefCell, collections::HashMap};

    impl DatabaseRaw for RefCell<HashMap<Vec<u8>, Vec<u8>>> {
        fn get<K>(&self, key: K) -> Result<Option<Vec<u8>>, Error>
        where
            K: AsRef<[u8]>,
        {
            Ok(self.borrow().get(&key.as_ref().to_vec()).cloned())
        }

        fn put<K, V>(&self, key: K, value: V) -> Result<(), Error>
        where
            K: AsRef<[u8]>,
            V: AsRef<[u8]>,
        {
            self.borrow_mut()
                .insert(key.as_ref().to_vec(), value.as_ref().to_vec());

            Ok(())
        }

        fn delete<K>(&self, key: K) -> Result<(), Error>
        where
            K: AsRef<[u8]>,
        {
            self.borrow_mut().remove(&key.as_ref().to_vec());

            Ok(())
        }
    }

    #[test]
    fn test_database_save_load() {
        let cois = cois_from_words(&["a", "b", "c"], mocked_bert_system());
        let user_interests = UserInterests {
            positive: cois.clone(),
            negative: cois,
        };
        let docs = data_with_mab(vec![(0, vec![1.; 128])].into_iter());
        let docs = PreviousDocuments::Mab(docs);
        let data = RerankerData::new(user_interests, docs);

        let database = Db::new(RefCell::new(HashMap::new()));
        database.save_data(&data).expect("saving data");
        let loaded_data = database.load_data().expect("load data").expect("loaded data");

        assert_eq!(data, loaded_data);
    }
}

/// A temporary dummy database.
pub struct DummyDatabase;

impl Database for DummyDatabase {
    fn save_data(&self, _state: &RerankerData) -> Result<(), Error> {
        Ok(())
    }

    fn load_data(&self) -> Result<Option<RerankerData>, Error> {
        Ok(None)
    }

    fn save_analytics(&self, _analytics: &Analytics) -> Result<(), Error> {
        Ok(())
    }
}
