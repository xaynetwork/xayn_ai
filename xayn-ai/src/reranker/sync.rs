use crate::{
    coi::reduce_cois,
    data::{CoiPoint, UserInterests},
    error::Error,
    utils::serialize_with_version,
};
use anyhow::bail;
use serde::{Deserialize, Serialize};

const CURRENT_SCHEMA_VERSION: u8 = 0;

/// Synchronizable data of the reranker.
#[cfg_attr(test, derive(Clone, PartialEq, Debug))]
#[derive(Default, Serialize, Deserialize)]
pub(crate) struct SyncData {
    pub(crate) user_interests: UserInterests,
}

impl SyncData {
    /// Deserializes a `SyncData` from `bytes`.
    pub(crate) fn deserialize(bytes: &[u8]) -> Result<Self, Error> {
        if bytes.is_empty() {
            bail!("Empty serialized data.");
        }

        // version encoded in first byte
        let version = bytes[0];
        if version != CURRENT_SCHEMA_VERSION {
            bail!(
                "Unsupported serialized data. Found version {} expected {}.",
                version,
                CURRENT_SCHEMA_VERSION,
            );
        }

        let data = bincode::deserialize(&bytes[1..])?;
        Ok(data)
    }

    /// Serializes a `SyncData` to a byte representation.
    pub(crate) fn serialize(&self) -> Result<Vec<u8>, Error> {
        serialize_with_version(self, CURRENT_SCHEMA_VERSION)
    }

    /// Synchronizes with another `SyncData`.
    pub(crate) fn synchronize(&mut self, other: SyncData) {
        let Self { user_interests } = other;
        self.user_interests.append(user_interests);

        reduce_cois(&mut self.user_interests.positive);
        reduce_cois(&mut self.user_interests.negative);
    }
}

impl UserInterests {
    /// Moves all user interests of `other` into `Self`.
    pub(crate) fn append(&mut self, mut other: Self) {
        append_cois(&mut self.positive, &mut other.positive);
        append_cois(&mut self.negative, &mut other.negative);
    }
}

/// Appends `remotes` to `locals`.
///
/// Remote CoIs with ids clashing with any of the local CoIs are removed.
fn append_cois<C: CoiPoint>(locals: &mut Vec<C>, remotes: &mut Vec<C>) {
    remotes.retain(|rem| !locals.iter().any(|loc| loc.id() == rem.id()));
    locals.append(remotes);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{
        mocked_smbert_system,
        neg_cois_from_words,
        neg_cois_from_words_with_ids,
        pos_cois_from_words,
        pos_cois_from_words_with_ids,
    };

    impl SyncData {
        fn from_words(words: &[&str], start_id: usize) -> Self {
            let positive = pos_cois_from_words_with_ids(words, mocked_smbert_system(), start_id);
            let negative = neg_cois_from_words_with_ids(words, mocked_smbert_system(), start_id);
            let user_interests = UserInterests { positive, negative };
            Self { user_interests }
        }

        /// True if the `SyncData` contains all elements of `other`.
        fn contains(&self, other: &SyncData) -> bool {
            let cois = &self.user_interests;
            let other_cois = &other.user_interests;
            other_cois
                .positive
                .iter()
                .all(|coi| cois.positive.contains(coi))
                && other_cois
                    .negative
                    .iter()
                    .all(|coi| cois.negative.contains(coi))
        }

        /// True if the `SyncData` equals `other` up to reordering of cois.
        fn eq_up_to_reordering(&self, other: &SyncData) -> bool {
            self.contains(other) && other.contains(self)
        }
    }

    #[test]
    fn test_syncdata_serialize_deserialize() {
        let syncdata = SyncData::from_words(&["a", "b", "c"], 0);
        let serialized = syncdata.serialize().expect("serialized data");
        let deserialized = SyncData::deserialize(&serialized).expect("deserialized data");

        assert_eq!(deserialized, syncdata);
    }

    #[test]
    fn test_synchronize_nonempty_empty() {
        // NOTE words deliberately chosen to be "far enough apart" to avoid merges
        let mut syncdata = SyncData::from_words(&["a", "m", "z"], 0);
        let syncdata_before = syncdata.clone();
        let empty = SyncData::default();

        syncdata.synchronize(empty);
        // check no change after sync
        assert_eq!(syncdata, syncdata_before);
    }

    #[test]
    fn test_synchronize_empty_nonempty() {
        let nonempty = SyncData::from_words(&["a", "m", "z"], 0);
        let mut empty = SyncData::default();

        empty.synchronize(nonempty.clone());
        // check empty becomes nonempty
        assert_eq!(empty, nonempty);
    }

    #[test]
    fn test_synchronize_commutative_distinct() {
        let words = &["a", "g", "m", "s", "z"];
        let mut pos_cois = pos_cois_from_words(words, mocked_smbert_system());
        let mut neg_cois = neg_cois_from_words(words, mocked_smbert_system());
        let pos_cois_union = pos_cois.clone();
        let neg_cois_union = neg_cois.clone();

        // split [a, g, m, s, z] -> [a, g, m], [s, z]
        let pos_cois2 = pos_cois.split_off(3);
        let neg_cois2 = neg_cois.split_off(3);

        let user_interests = UserInterests {
            positive: pos_cois,
            negative: neg_cois,
        };
        let mut data1 = SyncData { user_interests };
        let data1_before = data1.clone();

        let user_interests = UserInterests {
            positive: pos_cois2,
            negative: neg_cois2,
        };
        let mut data2 = SyncData { user_interests };

        data1.synchronize(data2.clone());
        assert_eq!(data1.user_interests.positive, pos_cois_union);
        assert_eq!(data1.user_interests.negative, neg_cois_union);

        data2.synchronize(data1_before);
        assert!(data2.eq_up_to_reordering(&data1));
    }

    #[test]
    fn test_synchronize_commutative_dupes() {
        let mut data1 = SyncData::from_words(&["a", "g", "m"], 0); // ids 0, 1, 2
        let data1_before = data1.clone();
        let mut data2 = SyncData::from_words(&["g", "m", "s"], 1); // ids 1, 2, 3

        // data1 becomes: { a, g, m, s }
        data1.synchronize(data2.clone());
        assert!(data1.contains(&data1_before));
        assert!(data1.contains(&data2));
        assert_eq!(data1.user_interests.positive.len(), 4);
        assert_eq!(data1.user_interests.negative.len(), 4);

        data2.synchronize(data1_before);
        assert!(data2.eq_up_to_reordering(&data1));
    }

    #[test]
    fn test_synchronize_commutative_merge() {
        let mut data1 = SyncData::from_words(&["a", "g", "m"], 0); // ids 0, 1, 2
        let data1_before = data1.clone();
        let mut data2 = SyncData::from_words(&["s", "f", "b"], 3); // ids 3, 4, 5

        // data1 becomes: { merge(a, b), merge(g, f), m, s }
        data1.synchronize(data2.clone());
        assert_eq!(data1.user_interests.positive.len(), 4);
        assert_eq!(data1.user_interests.negative.len(), 4);
        assert!(!data1.contains(&data1_before)); // no longer contains a, g
        assert!(!data1.contains(&data2)); // doesn't contain f, b

        // a is close to b, g is close to f, leaving m and s intact
        let data_unmerged = SyncData::from_words(&["m", "s"], 2); // ids 2, 3
        assert!(data1.contains(&data_unmerged)); // does contain m, s

        data2.synchronize(data1_before);
        assert!(data2.eq_up_to_reordering(&data1));
    }
}
