mod context;
pub(crate) mod public;
mod utils;

use std::collections::HashMap;

use displaydoc::Display;
use kpe::Pipeline as KPE;
use thiserror::Error;

use crate::{
    coi::{compute_coi_for_embedding, point::UserInterests, Configuration},
    data::document_data::CoiComponent,
    embedding::{smbert::SMBert, utils::Embedding},
    error::Error,
    ranker::{
        context::Context,
        utils::{Document, Id as DocumentId},
    },
    utils::nan_safe_f32_cmp,
};

#[derive(Error, Debug, Display)]
pub(crate) enum RankerError {
    /// No user interests are known.
    NoUserInterests,
}

/// The Ranker.
pub(crate) struct Ranker {
    /// SMBert system.
    smbert: SMBert,
    /// CoI configuration.
    coi_config: Configuration,
    #[allow(dead_code)]
    /// Key phrase extraction system.
    kpe: KPE,
    /// The learned user interests.
    user_interests: UserInterests,
}

impl Ranker {
    /// Creates a new `Ranker`.
    pub(crate) fn new(
        smbert: SMBert,
        coi_config: Configuration,
        kpe: KPE,
        user_interests: UserInterests,
    ) -> Self {
        Self {
            smbert,
            coi_config,
            kpe,
            user_interests,
        }
    }

    /// Creates a byte representation of the internal state of the ranker.
    pub(crate) fn serialize(&self) -> Result<Vec<u8>, Error> {
        bincode::serialize(&self.user_interests).map_err(Into::into)
    }

    /// Computes the SMBert embedding of the given `sequence`.
    pub(crate) fn compute_smbert(&self, sequence: &str) -> Result<Embedding, Error> {
        self.smbert.run(sequence).map_err(Into::into)
    }

    /// Ranks the given documents based on the learned user interests.
    ///
    /// # Errors
    ///
    /// Fails if no user interests are known.
    pub(crate) fn rank(&self, items: &mut [Document]) -> Result<(), Error> {
        let cois_for_docs =
            compute_cois_for_docs(items, &self.user_interests, self.coi_config.neighbors.get())?;
        let context_for_docs = compute_score_for_docs(cois_for_docs.as_slice());

        items.sort_unstable_by(|a, b| {
            nan_safe_f32_cmp(
                context_for_docs.get(&b.id).unwrap(),
                context_for_docs.get(&a.id).unwrap(),
            )
        });

        Ok(())
    }
}

fn compute_cois_for_docs(
    documents: &[Document],
    user_interests: &UserInterests,
    neighbors: usize,
) -> Result<Vec<(DocumentId, CoiComponent)>, Error> {
    documents
        .iter()
        .map(|document| {
            let coi =
                compute_coi_for_embedding(&document.smbert_embedding, user_interests, neighbors)
                    .ok_or(RankerError::NoUserInterests)?;
            Ok::<(DocumentId, CoiComponent), Error>((document.id, coi))
        })
        .collect()
}

fn compute_score_for_docs(
    cois_for_docs: &[(DocumentId, CoiComponent)],
) -> HashMap<&DocumentId, f32> {
    let context = Context::from_cois(cois_for_docs);
    let mut scores = HashMap::new();
    cois_for_docs.iter().for_each(|(id, coi)| {
        scores.insert(
            id,
            context.calculate_score(coi.pos_distance, coi.neg_distance),
        );
    });
    scores
}
