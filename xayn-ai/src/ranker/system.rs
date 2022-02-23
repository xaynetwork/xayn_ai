use std::time::Duration;

use displaydoc::Display;

use kpe::Pipeline as KPE;
use thiserror::Error;

use crate::{
    coi::{config::Config, key_phrase::KeyPhrase, point::UserInterests, CoiSystem, RelevanceMap},
    data::document::UserFeedback,
    embedding::{smbert::SMBert, utils::Embedding},
    error::Error,
    ranker::{
        context::{compute_score_for_docs, Error as ContextError},
        Document,
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

    /// Ranks the given documents based on the learned user interests.
    ///
    /// # Errors
    ///
    /// Fails if the scores of the documents cannot be computed.
    pub(crate) fn rank(&mut self, documents: &mut [impl Document]) -> Result<(), Error> {
        rank(
            documents,
            &self.user_interests,
            &mut self.coi.relevances,
            &self.coi.config,
        )
    }

    /// Logs the document view time and updates the user interests based on the given information.
    pub(crate) fn log_document_view_time(
        &mut self,
        user_feedback: UserFeedback,
        embedding: &Embedding,
        viewed: Duration,
    ) {
        if let UserFeedback::Relevant | UserFeedback::NotGiven = user_feedback {
            self.coi
                .log_document_view_time(&mut self.user_interests.positive, embedding, viewed)
        }
    }

    /// Logs the user reaction and updates the user interests based on the given information.
    pub(crate) fn log_user_reaction(
        &mut self,
        user_feedback: UserFeedback,
        snippet: &str,
        embedding: &Embedding,
    ) {
        match user_feedback {
            UserFeedback::Relevant => {
                let smbert = &self.smbert;
                let key_phrases = self.kpe.run(snippet).unwrap_or_default();
                self.coi.log_positive_user_reaction(
                    &mut self.user_interests.positive,
                    embedding,
                    |words| smbert.run(words).map_err(Into::into),
                    key_phrases.as_slice(),
                )
            }
            UserFeedback::Irrelevant => self
                .coi
                .log_negative_user_reaction(&mut self.user_interests.negative, embedding),
            _ => (),
        }
    }

    /// Selects the top key phrases from the positive cois, sorted in descending relevance.
    pub(crate) fn select_top_key_phrases(&mut self, top: usize) -> Vec<KeyPhrase> {
        self.coi
            .select_top_key_phrases(&self.user_interests.positive, top)
    }
}

fn rank(
    documents: &mut [impl Document],
    user_interests: &UserInterests,
    relevances: &mut RelevanceMap,
    config: &Config,
) -> Result<(), Error> {
    if documents.len() < 2 {
        return Ok(());
    }

    if let Ok(score_for_docs) =
        compute_score_for_docs(documents, user_interests, relevances, config)
    {
        documents.sort_unstable_by(|a, b| {
            nan_safe_f32_cmp(
                score_for_docs.get(&a.id()).unwrap(),
                score_for_docs.get(&b.id()).unwrap(),
            )
        });
    } else {
        documents.sort_unstable_by(|a, b| {
            a.score()
                .and_then(|a_score| {
                    b.score()
                        .map(|b_score| nan_safe_f32_cmp(&b_score, &a_score))
                })
                .unwrap_or_else(|| a.rank().cmp(&b.rank()))
        });
    }

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
            TestDocument::new(0, arr1(&[3., 0., 0.]), None, 0),
            TestDocument::new(1, arr1(&[1., 1., 0.]), None, 1),
            TestDocument::new(2, arr1(&[1., 0., 0.]), None, 2),
            TestDocument::new(3, arr1(&[5., 0., 0.]), None, 3),
        ];

        let config = Config::default()
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
        assert_eq!(documents[0].id(), DocumentId::from_u128(0));
        assert_eq!(documents[1].id(), DocumentId::from_u128(1));
        assert_eq!(documents[2].id(), DocumentId::from_u128(2));
        assert_eq!(documents[3].id(), DocumentId::from_u128(3));
    }

    #[test]
    fn test_rank_no_user_interests_no_score() {
        let mut documents = vec![
            TestDocument::new(0, arr1(&[0., 0., 0.]), None, 1),
            TestDocument::new(1, arr1(&[0., 0., 0.]), None, 0),
        ];

        let config = Config::default().with_min_positive_cois(1).unwrap();

        let res = rank(
            &mut documents,
            &UserInterests::default(),
            &mut RelevanceMap::default(),
            &config,
        );

        assert!(res.is_ok());
        assert_eq!(documents[0].id(), DocumentId::from_u128(1));
        assert_eq!(documents[1].id(), DocumentId::from_u128(0));
    }

    #[test]
    fn test_rank_no_user_interests_one_score() {
        let mut documents = vec![
            TestDocument::new(0, arr1(&[0., 0., 0.]), Some(1.), 1),
            TestDocument::new(1, arr1(&[0., 0., 0.]), None, 0),
        ];

        let config = Config::default().with_min_positive_cois(1).unwrap();

        let res = rank(
            &mut documents,
            &UserInterests::default(),
            &mut RelevanceMap::default(),
            &config,
        );

        assert!(res.is_ok());
        assert_eq!(documents[0].id(), DocumentId::from_u128(1));
        assert_eq!(documents[1].id(), DocumentId::from_u128(0));

        let mut documents = vec![
            TestDocument::new(0, arr1(&[0., 0., 0.]), None, 1),
            TestDocument::new(1, arr1(&[0., 0., 0.]), Some(1.), 0),
        ];

        let res = rank(
            &mut documents,
            &UserInterests::default(),
            &mut RelevanceMap::default(),
            &config,
        );

        assert!(res.is_ok());
        assert_eq!(documents[0].id(), DocumentId::from_u128(1));
        assert_eq!(documents[1].id(), DocumentId::from_u128(0));
    }

    #[test]
    fn test_rank_no_user_interests_with_score() {
        let mut documents = vec![
            TestDocument::new(0, arr1(&[0., 0., 0.]), Some(4.), 1),
            TestDocument::new(1, arr1(&[0., 0., 0.]), Some(3.), 0),
        ];

        let config = Config::default().with_min_positive_cois(1).unwrap();

        let res = rank(
            &mut documents,
            &UserInterests::default(),
            &mut RelevanceMap::default(),
            &config,
        );

        assert!(res.is_ok());
        assert_eq!(documents[0].id(), DocumentId::from_u128(0));
        assert_eq!(documents[1].id(), DocumentId::from_u128(1));
    }

    #[test]
    fn test_rank_no_documents() {
        let res = rank(
            &mut [] as &mut [TestDocument],
            &UserInterests::default(),
            &mut RelevanceMap::default(),
            &Config::default(),
        );
        assert!(res.is_ok());
    }
}
