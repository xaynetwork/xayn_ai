use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet, HashMap},
    ops::Index,
};

use derive_more::Into;

use crate::coi::{key_phrase::KeyPhrase, CoiError, CoiId};

/// A finite relevance score.
#[derive(Clone, Copy, Debug, Default, Into, PartialEq, PartialOrd)]
pub(crate) struct Relevance(f32);

impl Relevance {
    /// Creates a relevance score.
    ///
    /// # Errors
    /// Fails if the relevance isn't finite.
    pub fn new(relevance: f32) -> Result<Self, CoiError> {
        if relevance.is_finite() {
            Ok(Relevance(relevance))
        } else {
            Err(CoiError::NonFiniteRelevance)
        }
    }
}

impl Eq for Relevance {
    // never nan by construction
}

#[allow(clippy::derive_ord_xor_partial_ord)]
impl Ord for Relevance {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap(/* never nan by construction */)
    }
}

#[derive(Default)]
pub(crate) struct RelevanceMaps {
    coi_to_relevance: HashMap<CoiId, BTreeSet<Relevance>>,
    relevance_to_key_phrase: BTreeMap<(Relevance, CoiId), Vec<KeyPhrase>>,
}

impl RelevanceMaps {
    pub fn get_mut_relevances(&mut self, coi_id: CoiId) -> Option<&mut BTreeSet<Relevance>> {
        self.coi_to_relevance.get_mut(&coi_id)
    }

    pub fn insert_relevance(&mut self, coi_id: CoiId, relevance: Relevance) {
        self.coi_to_relevance
            .entry(coi_id)
            .or_default()
            .insert(relevance);
    }

    pub fn insert_relevances(
        &mut self,
        coi_id: CoiId,
        relevances: BTreeSet<Relevance>,
    ) -> Option<BTreeSet<Relevance>> {
        self.coi_to_relevance.insert(coi_id, relevances)
    }

    pub fn remove_relevances(&mut self, coi_id: CoiId) -> Option<BTreeSet<Relevance>> {
        self.coi_to_relevance.remove(&coi_id)
    }

    pub fn get_mut_key_phrases(
        &mut self,
        relevance: Relevance,
        coi_id: CoiId,
    ) -> Option<&mut Vec<KeyPhrase>> {
        self.relevance_to_key_phrase.get_mut(&(relevance, coi_id))
    }

    pub fn insert_key_phrase(
        &mut self,
        relevance: Relevance,
        coi_id: CoiId,
        key_phrase: KeyPhrase,
    ) {
        self.relevance_to_key_phrase
            .entry((relevance, coi_id))
            .or_default()
            .push(key_phrase);
    }

    pub fn remove_key_phrases(
        &mut self,
        relevance: Relevance,
        coi_id: CoiId,
    ) -> Option<Vec<KeyPhrase>> {
        self.relevance_to_key_phrase.remove(&(relevance, coi_id))
    }

    pub fn iter_key_phrases(
        &self,
    ) -> impl Iterator<Item = (Relevance, CoiId, &[KeyPhrase])> + DoubleEndedIterator {
        self.relevance_to_key_phrase
            .iter()
            .map(|(&(relevance, coi_id), key_phrases)| (relevance, coi_id, key_phrases.as_slice()))
    }

    pub fn insert(&mut self, coi_id: CoiId, relevance: Relevance, key_phrase: KeyPhrase) {
        self.insert_relevance(coi_id, relevance);
        self.insert_key_phrase(relevance, coi_id, key_phrase);
    }

    pub fn remove(&mut self, coi_id: CoiId) -> Option<BTreeSet<KeyPhrase>> {
        self.remove_relevances(coi_id)
            .map(|relevances| {
                let key_phrases = relevances
                    .into_iter()
                    .map(|relevance| {
                        self.remove_key_phrases(relevance, coi_id)
                            .unwrap_or_default()
                    })
                    .flatten()
                    .collect::<BTreeSet<_>>();
                (!key_phrases.is_empty()).then(|| key_phrases)
            })
            .flatten()
    }
}

impl Index<CoiId> for RelevanceMaps {
    type Output = BTreeSet<Relevance>;

    fn index(&self, coi_id: CoiId) -> &Self::Output {
        &self.coi_to_relevance[&coi_id]
    }
}

impl Index<(Relevance, CoiId)> for RelevanceMaps {
    type Output = [KeyPhrase];

    fn index(&self, (relevance, coi_id): (Relevance, CoiId)) -> &Self::Output {
        &self.relevance_to_key_phrase[&(relevance, coi_id)]
    }
}

#[cfg(test)]
mod tests {
    use std::iter::once;

    use ndarray::Ix;

    use test_utils::ApproxEqIter;

    use super::*;

    impl RelevanceMaps {
        pub fn relevances_len(&self) -> usize {
            self.coi_to_relevance.len()
        }

        pub fn relevances_is_empty(&self) -> bool {
            self.coi_to_relevance.is_empty()
        }

        pub fn key_phrases_len(&self) -> usize {
            self.relevance_to_key_phrase.len()
        }

        pub fn key_phrases_is_empty(&self) -> bool {
            self.relevance_to_key_phrase.is_empty()
        }
    }

    impl<'a> ApproxEqIter<'a, f32> for &'a Relevance {
        fn indexed_iter_logical_order(
            self,
            index_prefix: Vec<Ix>,
        ) -> Box<dyn Iterator<Item = (Vec<Ix>, f32)> + 'a> {
            Box::new(once((index_prefix, self.0)))
        }
    }
}
