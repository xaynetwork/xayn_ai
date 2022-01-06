use std::{
    io::{BufRead, Read},
    path::Path,
};

use kpe::Builder as KPEBuilder;
use layer::io::BinParams;
use rubert::{AveragePooler, SMBertBuilder};

use crate::{
    coi::{point::UserInterests, CoiSystem, Configuration as CoiSystemConfiguration},
    embedding::utils::Embedding,
    error::Error,
};

pub struct Ranker(super::Ranker);

impl Ranker {
    /// Creates a byte representation of the internal state of the ranker.
    pub fn serialize(&self) -> Result<Vec<u8>, Error> {
        self.0.serialize()
    }

    /// Computes the SMBert embedding of the given `sequence`.
    pub fn compute_smbert(&self, sequence: &str) -> Result<Embedding, Error> {
        self.0.compute_smbert(sequence)
    }
}

pub struct Builder<SV, SM, KV, KM> {
    smbert: SMBertBuilder<SV, SM>,
    kpe: KPEBuilder<KV, KM>,
    user_interests: UserInterests,
}

impl Default for Builder<(), (), (), ()> {
    fn default() -> Self {
        Self {
            smbert: SMBertBuilder::new((), ()),
            kpe: KPEBuilder::new((), (), BinParams::default(), BinParams::default()),
            user_interests: UserInterests::default(),
        }
    }
}

impl<SV, SM, KV, KM> Builder<SV, SM, KV, KM> {
    pub fn with_serialized_state(mut self, bytes: Option<impl AsRef<[u8]>>) -> Result<Self, Error> {
        if let Some(user_interests) = bytes {
            self.user_interests = bincode::deserialize(user_interests.as_ref())?;
        }
        Ok(self)
    }

    pub fn with_smbert_from_reader<W, N>(self, vocab: W, bert: N) -> Builder<W, N, KV, KM> {
        Builder {
            smbert: SMBertBuilder::new(vocab, bert),
            kpe: self.kpe,
            user_interests: self.user_interests,
        }
    }

    pub fn with_smbert_from_file(
        self,
        vocab: impl AsRef<Path>,
        bert: impl AsRef<Path>,
    ) -> Result<Builder<impl BufRead, impl Read, KV, KM>, Error> {
        Ok(Builder {
            smbert: SMBertBuilder::from_files(vocab, bert)?,
            kpe: self.kpe,
            user_interests: self.user_interests,
        })
    }

    pub fn with_kpe_from_reader<W, N>(
        self,
        vocab: W,
        bert: N,
        cnn: impl Read,
        classifier: impl Read,
    ) -> Result<Builder<SV, SM, W, N>, Error> {
        Ok(Builder {
            smbert: self.smbert,
            kpe: KPEBuilder::new(
                vocab,
                bert,
                BinParams::deserialize_from(cnn)?,
                BinParams::deserialize_from(classifier)?,
            ),
            user_interests: self.user_interests,
        })
    }

    pub fn with_kpe_from_file(
        self,
        vocab: impl AsRef<Path>,
        bert: impl AsRef<Path>,
        cnn: impl AsRef<Path>,
        classifier: impl AsRef<Path>,
    ) -> Result<Builder<SV, SM, impl BufRead, impl Read>, Error> {
        Ok(Builder {
            smbert: self.smbert,
            kpe: KPEBuilder::from_files(vocab, bert, cnn, classifier)?,
            user_interests: self.user_interests,
        })
    }

    pub fn build(self) -> Result<Ranker, Error>
    where
        SV: BufRead,
        SM: Read,
        KV: BufRead,
        KM: Read,
    {
        let smbert = self
            .smbert
            .with_token_size(52)?
            .with_accents(false)
            .with_lowercase(true)
            .with_pooling(AveragePooler)
            .build()?;
        let coi = CoiSystem::new(CoiSystemConfiguration::default());
        let kpe = self
            .kpe
            .with_token_size(52)?
            .with_accents(false)
            .with_lowercase(true)
            .build()?;

        Ok(Ranker(super::Ranker::new(
            smbert,
            coi,
            kpe,
            self.user_interests,
        )))
    }
}
