use std::cell::RefCell;

use anyhow::bail;
use serde::{Deserialize, Serialize};

use crate::{
    coi::point::UserInterests_v0_0_0,
    error::Error,
    reranker::{
        sync::{SyncData_v0_1_0, SyncData_v0_2_0, SyncData_v0_3_0},
        PreviousDocuments,
        CURRENT_SCHEMA_VERSION,
    },
    utils::serialize_with_version,
};

#[obake::versioned]
#[obake(version("0.0.0"))]
#[obake(version("0.1.0"))]
#[obake(version("0.2.0"))]
#[obake(version("0.3.0"))]
#[derive(Default, Deserialize, Serialize)]
#[cfg_attr(test, derive(Clone, Debug, PartialEq))]
pub(crate) struct RerankerData {
    #[obake(inherit)]
    #[obake(cfg(">=0.1"))]
    pub(super) sync_data: SyncData,
    #[obake(cfg(">=0.0"))]
    pub(super) prev_documents: PreviousDocuments,

    // removed fields go below this line
    #[obake(inherit)]
    #[obake(cfg(">=0.0, <0.1"))]
    pub(super) user_interests: UserInterests,
}

impl From<RerankerData_v0_0_0> for RerankerData_v0_1_0 {
    fn from(data: RerankerData_v0_0_0) -> Self {
        Self {
            sync_data: SyncData_v0_1_0 {
                user_interests: data.user_interests.into(),
            },
            prev_documents: data.prev_documents,
        }
    }
}

impl From<RerankerData_v0_1_0> for RerankerData_v0_2_0 {
    fn from(data: RerankerData_v0_1_0) -> Self {
        Self {
            sync_data: data.sync_data.into(),
            prev_documents: data.prev_documents,
        }
    }
}

impl From<RerankerData_v0_2_0> for RerankerData {
    fn from(data: RerankerData_v0_2_0) -> Self {
        Self {
            sync_data: data.sync_data.into(),
            prev_documents: data.prev_documents,
        }
    }
}

impl From<RerankerData_v0_1_0> for RerankerData {
    fn from(data: RerankerData_v0_1_0) -> Self {
        RerankerData_v0_2_0::from(data).into()
    }
}

impl From<RerankerData_v0_0_0> for RerankerData {
    fn from(data: RerankerData_v0_0_0) -> Self {
        RerankerData_v0_1_0::from(data).into()
    }
}

#[cfg_attr(test, mockall::automock)]
pub(crate) trait Database {
    fn load_data(&self) -> Result<Option<RerankerData>, Error>;

    fn serialize(&self, data: &RerankerData) -> Result<Vec<u8>, Error>;
}

#[derive(Default)]
pub(super) struct Db(RefCell<Option<RerankerData>>);

impl Db {
    /// Deserializes the bytes into a database.
    ///
    /// If `bytes` is empty, then an empty database is returned. If `bytes` represents an older
    /// version, then the database is migrated.
    pub fn deserialize(bytes: &[u8]) -> Result<Self, Error> {
        if bytes.is_empty() {
            return Ok(Self::default());
        }

        // version is encoded in the first byte
        let data = match bytes[0] {
            0 => bincode::deserialize::<RerankerData_v0_0_0>(&bytes[1..])?.into(),
            1 => bincode::deserialize::<RerankerData_v0_1_0>(&bytes[1..])?.into(),
            2 => bincode::deserialize::<RerankerData_v0_2_0>(&bytes[1..])?.into(),
            CURRENT_SCHEMA_VERSION => bincode::deserialize(&bytes[1..])?,
            version => bail!(
                "Unsupported serialized data. Found version {} expected {}",
                version,
                CURRENT_SCHEMA_VERSION,
            ),
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
        coi::point::{UserInterests, UserInterests_v0_1_0, UserInterests_v0_2_0},
        data::document_data::DocumentDataWithRank,
        reranker::sync::SyncData,
        tests::{
            data_with_rank,
            from_ids,
            mocked_smbert_system,
            neg_cois_from_words,
            pos_cois_from_words,
            pos_cois_from_words_v0,
            pos_cois_from_words_v1,
            pos_cois_from_words_v2,
        },
    };

    impl RerankerData_v0_0_0 {
        pub(crate) fn new_with_rank(
            user_interests: UserInterests_v0_0_0,
            prev_documents: Vec<DocumentDataWithRank>,
        ) -> Self {
            Self {
                user_interests,
                prev_documents: PreviousDocuments::Final(prev_documents),
            }
        }
    }

    impl RerankerData_v0_1_0 {
        pub(crate) fn new_with_rank(
            user_interests: UserInterests_v0_1_0,
            prev_documents: Vec<DocumentDataWithRank>,
        ) -> Self {
            Self {
                sync_data: SyncData_v0_1_0 { user_interests },
                prev_documents: PreviousDocuments::Final(prev_documents),
            }
        }
    }

    impl RerankerData_v0_2_0 {
        pub(crate) fn new_with_rank(
            user_interests: UserInterests_v0_2_0,
            prev_documents: Vec<DocumentDataWithRank>,
        ) -> Self {
            Self {
                sync_data: SyncData_v0_2_0 { user_interests },
                prev_documents: PreviousDocuments::Final(prev_documents),
            }
        }
    }

    impl RerankerData {
        pub(crate) fn new_with_rank(
            user_interests: UserInterests,
            prev_documents: Vec<DocumentDataWithRank>,
        ) -> Self {
            Self {
                sync_data: SyncData { user_interests },
                prev_documents: PreviousDocuments::Final(prev_documents),
            }
        }
    }

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

    #[test]
    fn test_database_migration_from_v1() {
        let words = &["a", "b", "c"];
        let positive = pos_cois_from_words_v1(words, mocked_smbert_system());
        let negative = neg_cois_from_words(words, mocked_smbert_system());
        let user_interests = UserInterests_v0_1_0 { positive, negative };
        let docs = data_with_rank(from_ids(0..1));
        let data = RerankerData_v0_1_0::new_with_rank(user_interests, docs);
        let serialized = serialize_with_version(&data, 1).expect("serialized data");

        let database = Db::deserialize(&serialized).expect("load data from serialized");
        let loaded_data = database
            .load_data()
            .expect("load data")
            .expect("loaded data");

        assert_eq!(RerankerData::from(data), loaded_data);
    }

    #[test]
    fn test_database_migration_from_v2() {
        let words = &["a", "b", "c"];
        let positive = pos_cois_from_words_v2(words, mocked_smbert_system());
        let negative = neg_cois_from_words(words, mocked_smbert_system());
        let user_interests = UserInterests_v0_2_0 { positive, negative };
        let docs = data_with_rank(from_ids(0..1));
        let data = RerankerData_v0_2_0::new_with_rank(user_interests, docs);
        let serialized = serialize_with_version(&data, 2).expect("serialized data");

        let database = Db::deserialize(&serialized).expect("load data from serialized");
        let loaded_data = database
            .load_data()
            .expect("load data")
            .expect("loaded data");

        assert_eq!(RerankerData::from(data), loaded_data);
    }
}
