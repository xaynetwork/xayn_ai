use std::time::Duration;

use displaydoc::Display;

use kpe::Pipeline as KPE;
use serde::{Deserialize, Serialize};
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
    utils::{nan_safe_f32_cmp, serialize_with_version},
};

#[derive(Error, Debug, Display)]
pub(crate) enum RankerError {
    /// No user interests are known.
    Context(#[from] ContextError),
}

pub(super) const STATE_VERSION: u8 = 0;

#[derive(Debug, Default, Serialize, Deserialize)]
pub(super) struct State {
    /// The learned user interests.
    pub(super) user_interests: UserInterests,

    /// Key phrases.
    pub(super) relevances: RelevanceMap,
}

/// The Ranker.
pub(crate) struct Ranker {
    /// SMBert system.
    smbert: SMBert,
    /// CoI system.
    coi: CoiSystem,
    /// Key phrase extraction system.
    kpe: KPE,
    state: State,
}

impl Ranker {
    /// Creates a new `Ranker`.
    pub(super) fn new(smbert: SMBert, coi: CoiSystem, kpe: KPE, state: State) -> Self {
        Self {
            smbert,
            coi,
            kpe,
            state,
        }
    }

    /// Creates a byte representation of the internal state of the ranker.
    pub(crate) fn serialize(&self) -> Result<Vec<u8>, Error> {
        serialize_with_version(&self.state, 0)
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
            &self.state.user_interests,
            &mut self.state.relevances,
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
            self.coi.log_document_view_time(
                &mut self.state.user_interests.positive,
                embedding,
                viewed,
            )
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
                    &mut self.state.user_interests.positive,
                    &mut self.state.relevances,
                    embedding,
                    |words| smbert.run(words).map_err(Into::into),
                    key_phrases.as_slice(),
                )
            }
            UserFeedback::Irrelevant => self
                .coi
                .log_negative_user_reaction(&mut self.state.user_interests.negative, embedding),
            _ => (),
        }
    }

    /// Selects the top key phrases from the positive cois, sorted in descending relevance.
    pub(crate) fn select_top_key_phrases(&mut self, top: usize) -> Vec<KeyPhrase> {
        self.coi.select_top_key_phrases(
            &self.state.user_interests.positive,
            &mut self.state.relevances,
            top,
        )
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
        documents.sort_unstable_by(|this, other| {
            nan_safe_f32_cmp(
                score_for_docs.get(&this.id()).unwrap(),
                score_for_docs.get(&other.id()).unwrap(),
            )
        });
    } else {
        documents.sort_unstable_by(|this, other| {
            this.date_published().cmp(&other.date_published()).reverse()
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
            TestDocument::new(0, arr1(&[3., 0., 0.]), "2000-01-01 00:00:03"),
            TestDocument::new(1, arr1(&[1., 1., 0.]), "2000-01-01 00:00:02"),
            TestDocument::new(2, arr1(&[1., 0., 0.]), "2000-01-01 00:00:01"),
            TestDocument::new(3, arr1(&[5., 0., 0.]), "2000-01-01 00:00:00"),
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
    fn test_rank_no_user_interests() {
        let mut documents = vec![
            TestDocument::new(0, arr1(&[0., 0., 0.]), "2000-01-01 00:00:00"),
            TestDocument::new(1, arr1(&[0., 0., 0.]), "2000-01-01 00:00:01"),
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
