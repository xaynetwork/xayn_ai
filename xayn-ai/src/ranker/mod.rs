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
    pub(crate) fn rank(&self, documents: &mut [Document]) -> Result<(), Error> {
        rank(
            documents,
            &self.user_interests,
            self.coi_config.neighbors.get(),
        )
    }
}

fn rank(
    documents: &mut [Document],
    user_interests: &UserInterests,
    neighbors: usize,
) -> Result<(), Error> {
    let cois_for_docs = compute_cois_for_docs(documents, user_interests, neighbors)?;
    let context_for_docs = compute_score_for_docs(cois_for_docs.as_slice());

    documents.sort_unstable_by(|a, b| {
        nan_safe_f32_cmp(
            context_for_docs.get(&b.id).unwrap(),
            context_for_docs.get(&a.id).unwrap(),
        )
    });

    Ok(())
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
            Ok((document.id, coi))
        })
        .collect()
}

fn compute_score_for_docs(
    cois_for_docs: &[(DocumentId, CoiComponent)],
) -> HashMap<&DocumentId, f32> {
    let context = Context::from_cois(cois_for_docs);
    cois_for_docs
        .iter()
        .map(|(id, coi)| {
            (
                id,
                context.calculate_score(coi.pos_distance, coi.neg_distance),
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use ndarray::arr1;

    use crate::coi::create_pos_cois;

    use super::*;

    #[test]
    fn test_rank() {
        let mut documents = vec![
            Document {
                id: DocumentId::from_u128(0),
                smbert_embedding: arr1(&[3., 0., 0.]).into(),
            },
            Document {
                id: DocumentId::from_u128(1),
                smbert_embedding: arr1(&[1., 1., 0.]).into(),
            },
            Document {
                id: DocumentId::from_u128(2),
                smbert_embedding: arr1(&[1., 0., 0.]).into(),
            },
            Document {
                id: DocumentId::from_u128(3),
                smbert_embedding: arr1(&[5., 0., 0.]).into(),
            },
        ];

        let positive = create_pos_cois(&[[1., 0., 0.]]);
        let user_interests = UserInterests {
            positive,
            ..Default::default()
        };

        let res = rank(
            &mut documents,
            &user_interests,
            Configuration::default().neighbors.get(),
        );

        assert!(res.is_ok());
        assert_eq!(documents[0].id, DocumentId::from_u128(2));
        assert_eq!(documents[1].id, DocumentId::from_u128(1));
        assert_eq!(documents[2].id, DocumentId::from_u128(0));
        assert_eq!(documents[3].id, DocumentId::from_u128(3));
    }

    #[test]
    fn test_rank_no_user_interests() {
        let mut documents = vec![Document {
            id: DocumentId::from_u128(0),
            smbert_embedding: arr1(&[0., 0., 0.]).into(),
        }];

        let res = rank(
            &mut documents,
            &UserInterests::default(),
            Configuration::default().neighbors.get(),
        );

        assert!(matches!(
            res.unwrap_err().downcast_ref(),
            Some(RankerError::NoUserInterests)
        ));
    }

    #[test]
    fn test_rank_no_documents() {
        let res = rank(
            &mut [],
            &UserInterests::default(),
            Configuration::default().neighbors.get(),
        );
        assert!(res.is_ok())
    }
}
