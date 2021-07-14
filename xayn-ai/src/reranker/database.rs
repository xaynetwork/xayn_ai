use anyhow::bail;
use std::cell::RefCell;

use crate::error::Error;

use super::RerankerData;

#[cfg_attr(test, mockall::automock)]
pub(crate) trait Database {
    fn load_data(&self) -> Result<Option<RerankerData>, Error>;

    fn serialize(&self, data: &RerankerData) -> Result<Vec<u8>, Error>;
}

const CURRENT_SCHEMA_VERSION: u8 = 1;

#[derive(Default)]
pub(super) struct Db(RefCell<Option<RerankerData>>);

impl Db {
    /// If `bytes` is empty it will return an empty database,
    /// otherwise it will try to deserialize the bytes and
    /// return a database with that data.
    pub fn deserialize(bytes: &[u8]) -> Result<Self, Error> {
        if bytes.is_empty() {
            return Ok(Self::default());
        }

        // version is encoded in the first byte
        let version = bytes[0];
        if version != CURRENT_SCHEMA_VERSION && version != 0 {
            bail!(
                "Unsupported serialized data. Found version {} expected {}",
                version,
                CURRENT_SCHEMA_VERSION
            );
        }
        // TODO migration from version 0 to current
        let data = bincode::deserialize(&bytes[1..])?;

        Ok(Self(RefCell::new(Some(data))))
    }

    fn serialize(data: &RerankerData) -> Result<Vec<u8>, Error> {
        let size = bincode::serialized_size(data)? + 1;
        let mut serialized = Vec::with_capacity(size as usize);
        // version is encoded in the first byte
        serialized.push(CURRENT_SCHEMA_VERSION);
        bincode::serialize_into(&mut serialized, data)?;

        Ok(serialized)
    }
}

impl Database for Db {
    fn load_data(&self) -> Result<Option<RerankerData>, Error> {
        Ok(self.0.borrow_mut().take())
    }

    fn serialize(&self, data: &RerankerData) -> Result<Vec<u8>, Error> {
        Db::serialize(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        data::UserInterests,
        tests::{
            data_with_mab,
            from_ids,
            mocked_smbert_system,
            neg_cois_from_words,
            pos_cois_from_words,
        },
    };

    #[test]
    fn test_database_serialize_load() {
        let words = &["a", "b", "c"];
        let pos_cois = pos_cois_from_words(words, mocked_smbert_system());
        let neg_cois = neg_cois_from_words(words, mocked_smbert_system());
        let user_interests = UserInterests {
            positive: pos_cois,
            negative: neg_cois,
        };
        let docs = data_with_mab(from_ids(0..1));
        let data = RerankerData::new_with_mab(user_interests, docs);

        let serialized = Db::serialize(&data).expect("serialized data");
        let database = Db::deserialize(&serialized).expect("load data from serialized");
        let loaded_data = database
            .load_data()
            .expect("load data")
            .expect("loaded data");

        assert_eq!(data, loaded_data);
    }
}
