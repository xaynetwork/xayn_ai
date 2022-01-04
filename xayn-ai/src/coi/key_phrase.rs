use std::{borrow::Borrow, collections::BTreeSet, mem::swap};

use derivative::Derivative;
use lazy_static::lazy_static;
use serde::{Deserialize, Serialize};

use crate::{
    coi::{
        point::{NegativeCoi, PositiveCoi},
        CoiError,
    },
    embedding::utils::ArcEmbedding,
};

#[derive(Clone, Debug, Derivative, Deserialize, Serialize)]
#[derivative(Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct KeyPhrase {
    words: String,
    #[derivative(Ord = "ignore", PartialEq = "ignore", PartialOrd = "ignore")]
    point: ArcEmbedding,
    #[derivative(Ord = "ignore", PartialEq = "ignore", PartialOrd = "ignore")]
    relevance: f32,
}

lazy_static! {
    // TODO: temporary workaround, remove once positive and negative cois have been split properly
    static ref EMPTY_KEY_PHRASES: BTreeSet<KeyPhrase> = BTreeSet::new();
}

impl KeyPhrase {
    pub(crate) fn new(
        words: impl Into<String>,
        point: impl Into<ArcEmbedding>,
    ) -> Result<Self, CoiError> {
        let words = words.into();
        let point = point.into();
        let relevance = 0.;

        if words.is_empty() || point.is_empty() {
            return Err(CoiError::EmptyKeyPhrase);
        }
        if !point.iter().copied().all(f32::is_finite) {
            return Err(CoiError::NonFiniteKeyPhrase(point));
        }

        Ok(Self {
            words,
            point,
            relevance,
        })
    }

    pub(crate) fn with_relevance(self, relevance: f32) -> Result<Self, CoiError> {
        if (0. ..=1.).contains(&relevance) {
            Ok(Self { relevance, ..self })
        } else {
            Err(CoiError::NonNormalizedKeyPhrase(relevance))
        }
    }

    #[cfg(test)]
    pub(crate) fn words(&self) -> &str {
        &self.words
    }

    pub(crate) fn point(&self) -> &ArcEmbedding {
        &self.point
    }

    #[cfg(test)]
    pub(crate) fn relevance(&self) -> f32 {
        self.relevance
    }
}

impl Borrow<String> for KeyPhrase {
    fn borrow(&self) -> &String {
        &self.words
    }
}

impl Borrow<str> for KeyPhrase {
    fn borrow(&self) -> &str {
        self.words.as_str()
    }
}

pub(crate) trait CoiPointKeyPhrases {
    fn key_phrases(&self) -> &BTreeSet<KeyPhrase>;

    fn swap_key_phrases(&mut self, candidates: BTreeSet<KeyPhrase>) -> BTreeSet<KeyPhrase>;
}

impl CoiPointKeyPhrases for PositiveCoi {
    fn key_phrases(&self) -> &BTreeSet<KeyPhrase> {
        &self.key_phrases
    }

    fn swap_key_phrases(&mut self, mut candidates: BTreeSet<KeyPhrase>) -> BTreeSet<KeyPhrase> {
        swap(&mut self.key_phrases, &mut candidates);
        candidates
    }
}

impl CoiPointKeyPhrases for NegativeCoi {
    fn key_phrases(&self) -> &BTreeSet<KeyPhrase> {
        &EMPTY_KEY_PHRASES
    }

    fn swap_key_phrases(&mut self, _candidates: BTreeSet<KeyPhrase>) -> BTreeSet<KeyPhrase> {
        BTreeSet::default()
    }
}
