use std::{collections::HashSet, mem::swap};

use derivative::Derivative;
use lazy_static::lazy_static;
use serde::{Deserialize, Serialize};

use crate::{
    coi::{
        point::{NegativeCoi, PositiveCoi},
        CoiError,
    },
    embedding::utils::Embedding,
};

#[derive(Clone, Debug, Derivative, Deserialize, Serialize)]
#[derivative(Eq, Hash, PartialEq)]
pub(crate) struct KeyPhrase {
    words: String,
    #[derivative(Hash = "ignore", PartialEq = "ignore")]
    point: Embedding,
}

lazy_static! {
    // TODO: temporary workaround, remove once positive and negative cois have been split properly
    static ref EMPTY_KEY_PHRASES: HashSet<KeyPhrase> = HashSet::new();
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

pub(crate) trait CoiPointKeyPhrases {
    fn key_phrases(&self) -> &HashSet<KeyPhrase>;

    fn swap_key_phrases(&mut self, candidates: HashSet<KeyPhrase>) -> HashSet<KeyPhrase>;
}

impl CoiPointKeyPhrases for PositiveCoi {
    fn key_phrases(&self) -> &HashSet<KeyPhrase> {
        &self.key_phrases
    }

    fn swap_key_phrases(&mut self, mut candidates: HashSet<KeyPhrase>) -> HashSet<KeyPhrase> {
        swap(&mut self.key_phrases, &mut candidates);
        candidates
    }
}

impl CoiPointKeyPhrases for NegativeCoi {
    fn key_phrases(&self) -> &HashSet<KeyPhrase> {
        &EMPTY_KEY_PHRASES
    }

    fn swap_key_phrases(&mut self, _candidates: HashSet<KeyPhrase>) -> HashSet<KeyPhrase> {
        HashSet::default()
    }
}
