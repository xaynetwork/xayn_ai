use anyhow::bail;
use serde::{Deserialize, Serialize};

use crate::{
    coi::{
        merge::reduce_cois,
        point::{CoiPoint, UserInterests, UserInterests_v0_1_0, UserInterests_v0_2_0},
    },
    error::Error,
    reranker::CURRENT_SCHEMA_VERSION,
    utils::serialize_with_version,
};

/// Synchronizable data of the reranker.
#[obake::versioned]
#[obake(version("0.1.0"))]
#[obake(version("0.2.0"))]
#[derive(Default, Deserialize, Serialize)]
#[cfg_attr(test, derive(Clone, Debug, PartialEq))]
pub(crate) struct SyncData {
    #[obake(inherit)]
    #[obake(cfg(">=0.1"))]
    pub(crate) user_interests: UserInterests,
}

impl From<SyncData_v0_1_0> for SyncData {
    fn from(data: SyncData_v0_1_0) -> Self {
        Self {
            user_interests: data.user_interests.into(),
        }
    }
}

impl SyncData {
    /// Deserializes a `SyncData` from `bytes`.
    pub(crate) fn deserialize(bytes: &[u8]) -> Result<Self, Error> {
        if bytes.is_empty() {
            bail!("Empty serialized data.");
        }

        // version encoded in first byte
        let data = match bytes[0] {
            0 | 1 => bincode::deserialize::<SyncData_v0_1_0>(&bytes[1..])?.into(),
            CURRENT_SCHEMA_VERSION => bincode::deserialize(&bytes[1..])?,
            version => bail!(
                "Unsupported serialized data. Found version {} expected {}.",
                version,
                CURRENT_SCHEMA_VERSION,
            ),
        };

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

    impl UserInterests {
        fn from_words(words: &[&str], start_id: usize) -> Self {
            let positive = pos_cois_from_words_with_ids(words, mocked_smbert_system(), start_id);
            let negative = neg_cois_from_words_with_ids(words, mocked_smbert_system(), start_id);
            Self { positive, negative }
        }
    }

    impl SyncData {
        fn from_words(words: &[&str], start_id: usize) -> Self {
            let user_interests = UserInterests::from_words(words, start_id);
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
    fn test_append_user_interests() {
        let mut cois = UserInterests::from_words(&["a", "b", "c"], 0); // ids 0, 1, 2
        let extra = UserInterests::from_words(&["b", "c", "d"], 1); // ids 1, 2, 3

        cois.append(extra);

        let expected = UserInterests::from_words(&["a", "b", "c", "d"], 0);
        assert_eq!(cois, expected);
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
        let mut data1 = SyncData::from_words(&["g", "a", "m"], 0); // ids 0, 1, 2
        let data1_before = data1.clone();
        let mut data2 = SyncData::from_words(&["s", "f", "a"], 3); // ids 3, 4, 5

        // data1.a & data2.a are close; data1.g & data2.f are close
        // data1 should become: { merge(a, a), merge(g, f), m, s }
        data1.synchronize(data2.clone());
        assert_eq!(data1.user_interests.positive.len(), 4);
        assert_eq!(data1.user_interests.negative.len(), 4);

        // check data1 contains merge(a1, a5) = a1
        // and leftovers m, s
        let expected_123 = SyncData::from_words(&["a", "m", "s"], 1); // ids 1, 2, 3
        assert!(data1.contains(&expected_123));

        // check data1 contains merge(g, f)
        let mut expected_0 = SyncData::from_words(&["g"], 0);
        expected_0.synchronize(SyncData::from_words(&["f"], 4));
        assert_eq!(expected_0.user_interests.positive.len(), 1);
        assert_eq!(expected_0.user_interests.negative.len(), 1);
        assert!(data1.contains(&expected_0));

        data2.synchronize(data1_before);
        assert!(data2.eq_up_to_reordering(&data1));
    }

    #[test]
    fn test_synchronize_noncommutative_collision() {
        let mut data1 = SyncData::from_words(&["a", "g"], 0); // ids 0, 1
        let data1_before = data1.clone();
        let mut data2 = SyncData::from_words(&["m", "s"], 1); // ids 1, 2

        data1.synchronize(data2.clone());
        // m is removed: id clashes with g's
        let expected = SyncData::from_words(&["a", "g", "s"], 0);
        assert_eq!(data1, expected);

        data2.synchronize(data1_before);
        assert!(!data2.eq_up_to_reordering(&data1));
        // g is removed: id clashes with m's
        let expected = SyncData::from_words(&["a", "m", "s"], 0);
        assert!(data2.eq_up_to_reordering(&expected));
    }
}
