use anyhow::bail;
use serde::Deserialize;
use std::cell::RefCell;

use crate::{data::UserInterests, error::Error, utils::serialize_with_version};

use super::{sync::SyncData, PreviousDocuments, RerankerData};

#[cfg_attr(test, mockall::automock)]
pub(crate) trait Database {
    fn load_data(&self) -> Result<Option<RerankerData>, Error>;

    fn serialize(&self, data: &RerankerData) -> Result<Vec<u8>, Error>;
}

const CURRENT_SCHEMA_VERSION: u8 = 1;

#[derive(Default)]
pub(super) struct Db(RefCell<Option<RerankerData>>);

/// Version 0 of `RerankerData` for backwards compatibility.
#[cfg_attr(test, derive(serde::Serialize))]
#[derive(Deserialize)]
struct RerankerData0 {
    user_interests: UserInterests,
    prev_documents: PreviousDocuments,
}

impl From<RerankerData0> for RerankerData {
    fn from(data0: RerankerData0) -> Self {
        let user_interests = data0.user_interests;
        let sync_data = SyncData { user_interests };
        Self {
            sync_data,
            prev_documents: data0.prev_documents,
        }
    }
}

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
        if version > CURRENT_SCHEMA_VERSION {
            bail!(
                "Unsupported serialized data. Found version {} expected {}",
                version,
                CURRENT_SCHEMA_VERSION,
            );
        }

        // migration from version 0 to current
        let data = if version == 0 {
            bincode::deserialize::<RerankerData0>(&bytes[1..])?.into()
        } else {
            bincode::deserialize(&bytes[1..])?
        };

        Ok(Self(RefCell::new(Some(data))))
    }
}

impl Database for Db {
    fn load_data(&self) -> Result<Option<RerankerData>, Error> {
        Ok(self.0.borrow_mut().take())
    }

    fn serialize(&self, data: &RerankerData) -> Result<Vec<u8>, Error> {
        serialize_with_version(data, CURRENT_SCHEMA_VERSION)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        data::{UserInterests, UserInterests_v0_0_0},
        reranker::{RerankerData, RerankerData_v0_0_0},
        tests::{
            data_with_rank,
            from_ids,
            mocked_smbert_system,
            neg_cois_from_words,
            pos_cois_from_words,
            pos_cois_from_words_v0,
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
        let docs = data_with_rank(from_ids(0..1));
        let data = RerankerData::new_with_rank(user_interests, docs);
        let serialized =
            serialize_with_version(&data, CURRENT_SCHEMA_VERSION).expect("serialized data");

        let database = Db::deserialize(&serialized).expect("load data from serialized");
        let loaded_data = database
            .load_data()
            .expect("load data")
            .expect("loaded data");

        assert_eq!(data, loaded_data);
    }

    #[test]
    fn test_database_migration_from_v0() {
        let words = &["a", "b", "c"];
        let positive = pos_cois_from_words_v0(words, mocked_smbert_system());
        let negative = neg_cois_from_words(words, mocked_smbert_system());
        let user_interests = UserInterests_v0_0_0 { positive, negative };
        let docs = data_with_rank(from_ids(0..1));
        let data = RerankerData_v0_0_0::new_with_rank(user_interests, docs);
        let serialized = serialize_with_version(&data, 0).expect("serialized data");

        let database = Db::deserialize(&serialized).expect("load data from serialized");
        let loaded_data = database
            .load_data()
            .expect("load data")
            .expect("loaded data");

        assert_eq!(RerankerData::from(data), loaded_data);
    }
}
