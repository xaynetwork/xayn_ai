mod context;
pub(crate) mod public;
mod util;

use std::collections::HashMap;

use kpe::Pipeline as KPE;
use rubert::SMBert;

use crate::{
    coi::{point::UserInterests, CoiSystem},
    data::document_data::CoiComponent,
    embedding::utils::Embedding,
    error::Error,
    ranker::{
        context::Context,
        util::{Document, Id},
    },
    utils::nan_safe_f32_cmp,
};

/// The Ranker.
pub(crate) struct Ranker {
    coi: CoiSystem,
    smbert: SMBert,
    #[allow(dead_code)]
    kpe: KPE,
    user_interests: UserInterests,
}

impl Ranker {
    /// Creates a new `Ranker`.
    pub(crate) fn new(
        smbert: SMBert,
        coi: CoiSystem,
        kpe: KPE,
        user_interests: UserInterests,
    ) -> Self {
        Self {
            smbert,
            coi,
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

    pub(crate) fn rank(&self, items: &mut [Document]) -> Result<(), Error> {
        let mut cois = items
            .iter()
            .map(|item| {
                match self
                    .coi
                    .compute_coi_for_embedding(&item.smbert_embedding, &self.user_interests)
                    .ok_or(anyhow::anyhow!("No Coi"))
                {
                    Ok(coi) => Ok((item.id, coi)),
                    Err(err) => return Err(err),
                }
            })
            .collect::<Result<Vec<(Id, CoiComponent)>, Error>>()?;

        let context = Context::from_cois(cois.as_slice());

        let mut context_for_docs = HashMap::new();

        cois.iter().for_each(|(id, coi)| {
            context_for_docs.insert(
                id.clone(),
                context.calculate(coi.pos_distance, coi.neg_distance),
            );
        });

        items.sort_unstable_by(|a, b| {
            nan_safe_f32_cmp(
                context_for_docs.get(&a.0).unwrap(),
                context_for_docs.get(&b.0).unwrap(),
            )
        });

        Ok(())
    }
}
