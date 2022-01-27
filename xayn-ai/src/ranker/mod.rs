pub(crate) mod config;
mod context;
mod document;
pub(crate) mod public;

use std::time::Duration;

use displaydoc::Display;
use kpe::Pipeline as KPE;
use thiserror::Error;

use crate::{
    coi::{
        key_phrase::KeyPhrase,
        point::UserInterests,
        CoiSystem,
        DocumentRelevance,
        RelevanceMap,
    },
    data::document::{Relevance, UserFeedback},
    embedding::{smbert::SMBert, utils::Embedding},
    error::Error,
    ranker::{
        config::Configuration,
        context::{compute_score_for_docs, Error as ContextError},
        document::Document,
    },
    utils::nan_safe_f32_cmp,
};

#[derive(Error, Debug, Display)]
pub(crate) enum RankerError {
    /// No user interests are known.
    Context(#[from] ContextError),
}

/// The Ranker.
pub(crate) struct Ranker {
    /// Ranker configuration.
    config: Configuration,
    /// SMBert system.
    smbert: SMBert,
    /// CoI system.
    coi: CoiSystem,
    /// Key phrase extraction system.
    kpe: KPE,
    /// The learned user interests.
    user_interests: UserInterests,
}

impl Ranker {
    /// Creates a new `Ranker`.
    pub(crate) fn new(
        config: Configuration,
        smbert: SMBert,
        coi: CoiSystem,
        kpe: KPE,
        user_interests: UserInterests,
    ) -> Self {
        Self {
            config,
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

    /// Ranks the given documents based on the learned user interests.
    ///
    /// # Errors
    ///
    /// Fails if the scores of the documents cannot be computed.
    pub(crate) fn rank(&mut self, documents: &mut [impl Document]) -> Result<(), Error> {
        rank(
            documents,
            &self.user_interests,
            self.coi.relevances_mut(),
            &self.config,
        )
    }

    /// Updates the user interests based on the given information.
    #[allow(dead_code)]
    fn update_user_interests(
        &mut self,
        relevance: Relevance,
        user_feedback: UserFeedback,
        snippet: &str,
        embedding: &Embedding,
        viewed: Duration,
    ) {
        match (relevance, user_feedback).into() {
            DocumentRelevance::Positive => {
                let smbert = &self.smbert;
                let key_phrases = self.kpe.run(snippet).unwrap_or_default();
                self.coi.update_positive_coi(
                    &mut self.user_interests.positive,
                    embedding,
                    &self.config,
                    |words| smbert.run(words).map_err(Into::into),
                    key_phrases.as_slice(),
                    viewed,
                )
            }
            DocumentRelevance::Negative => self.coi.update_negative_coi(
                &mut self.user_interests.negative,
                embedding,
                &self.config,
            ),
        }
    }

    /// Selects the top key phrases from the positive cois, sorted in descending relevance.
    pub(crate) fn select_top_key_phrases(&mut self, top: usize) -> Vec<KeyPhrase> {
        self.coi
            .select_top_key_phrases(&self.user_interests.positive, &self.config, top)
    }
}

fn rank(
    documents: &mut [impl Document],
    user_interests: &UserInterests,
    relevances: &mut RelevanceMap,
    config: &Configuration,
) -> Result<(), Error> {
    if documents.len() < 2 {
        return Ok(());
    }

    let score_for_docs = compute_score_for_docs(documents, user_interests, relevances, config)?;

    documents.sort_unstable_by(|a, b| {
        nan_safe_f32_cmp(
            score_for_docs.get(&a.id()).unwrap(),
            score_for_docs.get(&b.id()).unwrap(),
        )
    });

    Ok(())
}

#[cfg(test)]
mod tests {
    use ndarray::arr1;

    use crate::{
        coi::{create_neg_cois, create_pos_cois},
        ranker::document::TestDocument,
        DocumentId,
    };

    use super::*;

    #[test]
    fn test_rank() {
        let mut documents = vec![
            TestDocument::new(0, arr1(&[3., 0., 0.])),
            TestDocument::new(1, arr1(&[1., 1., 0.])),
            TestDocument::new(2, arr1(&[1., 0., 0.])),
            TestDocument::new(3, arr1(&[5., 0., 0.])),
        ];

        let config = Configuration::default()
            .with_min_positive_cois(1)
            .unwrap()
            .with_min_negative_cois(1)
            .unwrap();
        let positive = create_pos_cois(&[[1., 0., 0.]]);
        let negative = create_neg_cois(&[[100., 0., 0.]]);

        let user_interests = UserInterests { positive, negative };

        let res = rank(
            &mut documents,
            &user_interests,
            &mut RelevanceMap::default(),
            &config,
        );

        assert!(res.is_ok());
        assert_eq!(documents[0].id(), DocumentId::from_u128(2));
        assert_eq!(documents[1].id(), DocumentId::from_u128(1));
        assert_eq!(documents[2].id(), DocumentId::from_u128(0));
        assert_eq!(documents[3].id(), DocumentId::from_u128(3));
    }

    #[test]
    fn test_rank_no_user_interests() {
        let mut documents = vec![
            TestDocument::new(0, arr1(&[0., 0., 0.])),
            TestDocument::new(1, arr1(&[0., 0., 0.])),
        ];

        let config = Configuration::default().with_min_positive_cois(1).unwrap();

        let res = rank(
            &mut documents,
            &UserInterests::default(),
            &mut RelevanceMap::default(),
            &config,
        );

        assert!(matches!(
            res.unwrap_err().downcast_ref(),
            Some(ContextError::NotEnoughCois)
        ));
    }

    #[test]
    fn test_rank_no_documents() {
        let res = rank(
            &mut [] as &mut [TestDocument],
            &UserInterests::default(),
            &mut RelevanceMap::default(),
            &Configuration::default(),
        );
        assert!(res.is_ok())
    }
}
