use std::mem::swap;

use derive_more::Deref;
use serde::{Deserialize, Serialize};

use crate::{
    coi::{
        point::{NegativeCoi, PositiveCoi},
        CoiError,
    },
    embedding::utils::Embedding,
};

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub(crate) struct KeyPhrase {
    words: String,
    point: Embedding,
}

impl KeyPhrase {
    #[allow(dead_code)]
    pub(crate) fn new(
        words: impl Into<String>,
        point: impl Into<Embedding>,
    ) -> Result<Self, CoiError> {
        let words = words.into();
        let point = point.into();
        if !words.is_empty() && point.iter().copied().all(f32::is_finite) {
            Ok(Self { words, point })
        } else {
            Err(CoiError::InvalidKeyPhrase)
        }
    }

    #[allow(dead_code)]
    pub(crate) fn words(&self) -> &str {
        &self.words
    }

    #[allow(dead_code)]
    pub(crate) fn point(&self) -> &Embedding {
        &self.point
    }
}

// invariant: must be unique
// note: can't use neither HashMap nor sort & dedup because the underlying ArrayBase doesn't
// implement any of Eq, Hash, PartialOrd and Ord
#[derive(Clone, Debug, Default, Deref, Deserialize, PartialEq, Serialize)]
pub(crate) struct KeyPhrases(Vec<KeyPhrase>);

impl KeyPhrases {
    #[allow(dead_code)]
    pub(crate) fn new(key_phrases: Vec<KeyPhrase>) -> Result<Self, CoiError> {
        for (i, this) in key_phrases.iter().enumerate() {
            for other in key_phrases[i + 1..].iter() {
                if this == other {
                    return Err(CoiError::DuplicateKeyPhrases);
                }
            }
        }

        Ok(Self(key_phrases))
    }
}

impl IntoIterator for KeyPhrases {
    type Item = KeyPhrase;
    type IntoIter = <Vec<KeyPhrase> as IntoIterator>::IntoIter;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

pub(crate) trait CoiPointKeyPhrases {
    fn key_phrases(&self) -> &[KeyPhrase];

    fn swap_key_phrases(&mut self, candidates: KeyPhrases) -> KeyPhrases;
}

impl CoiPointKeyPhrases for PositiveCoi {
    fn key_phrases(&self) -> &[KeyPhrase] {
        self.key_phrases.as_slice()
    }

    fn swap_key_phrases(&mut self, mut candidates: KeyPhrases) -> KeyPhrases {
        swap(&mut self.key_phrases, &mut candidates);
        candidates
    }
}

impl CoiPointKeyPhrases for NegativeCoi {
    fn key_phrases(&self) -> &[KeyPhrase] {
        &[]
    }

    fn swap_key_phrases(&mut self, _candidates: KeyPhrases) -> KeyPhrases {
        KeyPhrases::default()
    }
}
