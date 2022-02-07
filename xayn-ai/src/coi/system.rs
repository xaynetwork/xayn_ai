use std::time::Duration;

use displaydoc::Display;
use thiserror::Error;
use uuid::Uuid;

use crate::{
    coi::{
        point::{
            find_closest_coi,
            find_closest_coi_mut,
            CoiPoint,
            NegativeCoi,
            PositiveCoi,
            UserInterests,
        },
        relevance::RelevanceMap,
        utils::{classify_documents_based_on_user_feedback, collect_matching_documents},
        CoiId,
    },
    data::document_data::{CoiComponent, DocumentDataWithCoi, DocumentDataWithSMBert},
    embedding::{
        smbert::SMBert,
        utils::{Embedding, MINIMUM_COSINE_SIMILARITY},
    },
    ranker::Config,
    reranker::systems::{self, CoiSystemData},
    DocumentHistory,
    Error,
};

use super::key_phrase::KeyPhrase;

#[derive(Error, Debug, Display)]
pub(crate) enum CoiSystemError {
    /// No CoI could be found for the given embedding
    NoCoi,
    /// No matching documents could be found
    NoMatchingDocuments,
}

pub(crate) struct CoiSystem {
    config: Config,
    smbert: SMBert,
    relevances: RelevanceMap,
}

impl CoiSystem {
    /// Creates a new centre of interest system.
    pub(crate) fn new(config: Config, smbert: SMBert) -> Self {
        Self {
            config,
            smbert,
            relevances: RelevanceMap::default(),
        }
    }

    /// Updates the view time of the positive coi closest to the embedding.
    pub(crate) fn log_document_view_time(
        &mut self,
        cois: &mut Vec<PositiveCoi>,
        embedding: &Embedding,
        viewed: Duration,
    ) {
        log_document_view_time(cois, embedding, viewed);
    }

    /// Updates the positive coi closest to the embedding or creates a new one if it's too far away.
    pub(crate) fn log_positive_user_reaction(
        &mut self,
        cois: &mut Vec<PositiveCoi>,
        embedding: &Embedding,
        config: &Config,
        smbert: impl Fn(&str) -> Result<Embedding, Error>,
        candidates: &[String],
    ) {
        log_positive_user_reaction(
            cois,
            embedding,
            config,
            &mut self.relevances,
            smbert,
            candidates,
        );
    }

    /// Updates the negative coi closest to the embedding or creates a new one if it's too far away.
    pub(crate) fn log_negative_user_reaction(
        &self,
        cois: &mut Vec<NegativeCoi>,
        embedding: &Embedding,
        config: &Config,
    ) {
        log_negative_user_reaction(cois, embedding, config);
    }

    /// Selects the top key phrases from the positive cois, sorted in descending relevance.
    pub(crate) fn select_top_key_phrases(
        &mut self,
        cois: &[PositiveCoi],
        config: &Config,
        top: usize,
    ) -> Vec<KeyPhrase> {
        self.relevances
            .select_top_key_phrases(cois, top, config.horizon(), config.penalty())
    }

    /// Returns a mutable reference to the inner [`RelevanceMap`].
    pub(crate) fn relevances_mut(&mut self) -> &mut RelevanceMap {
        &mut self.relevances
    }
}

impl systems::CoiSystem for CoiSystem {
    fn compute_coi(
        &self,
        documents: &[DocumentDataWithSMBert],
        user_interests: &UserInterests,
    ) -> Result<Vec<DocumentDataWithCoi>, Error> {
        compute_coi(documents, user_interests)
    }

    fn update_user_interests(
        &mut self,
        history: &[DocumentHistory],
        documents: &[&dyn CoiSystemData],
        user_interests: UserInterests,
    ) -> Result<UserInterests, Error> {
        let smbert = &self.smbert;
        update_user_interests(
            user_interests,
            &mut self.relevances,
            history,
            documents,
            |key_phrase| smbert.run(key_phrase).map_err(Into::into),
            &self.config,
        )
    }
}

/// Assigns a CoI for the given embedding.
///
/// Returns `None` if no CoI could be found otherwise it returns the Id of
/// the CoI along with the positive and negative similarity.
pub(crate) fn compute_coi_for_embedding(
    embedding: &Embedding,
    user_interests: &UserInterests,
) -> Option<CoiComponent> {
    let (coi, pos_similarity) = find_closest_coi(&user_interests.positive, embedding)?;
    let neg_similarity = match find_closest_coi(&user_interests.negative, embedding) {
        Some((_, similarity)) => similarity,
        None => MINIMUM_COSINE_SIMILARITY,
    };

    Some(CoiComponent {
        id: coi.id,
        pos_similarity,
        neg_similarity,
    })
}

pub(crate) fn compute_coi(
    documents: &[DocumentDataWithSMBert],
    user_interests: &UserInterests,
) -> Result<Vec<DocumentDataWithCoi>, Error> {
    documents
        .iter()
        .map(|document| {
            compute_coi_for_embedding(&document.smbert.embedding, user_interests)
                .map(|coi| DocumentDataWithCoi::from_document(document, coi))
                .ok_or_else(|| CoiSystemError::NoCoi.into())
        })
        .collect()
}

/// Updates the positive coi closest to the embedding or creates a new one if it's too far away.
fn log_positive_user_reaction(
    cois: &mut Vec<PositiveCoi>,
    embedding: &Embedding,
    config: &Config,
    relevances: &mut RelevanceMap,
    smbert: impl Fn(&str) -> Result<Embedding, Error>,
    candidates: &[String],
) {
    match find_closest_coi_mut(cois, embedding) {
        // If the given embedding's similarity to the CoI is above the threshold,
        // we adjust the position of the nearest CoI
        Some((coi, similarity)) if similarity >= config.threshold() => {
            coi.shift_point(embedding, config.shift_factor());
            coi.select_key_phrases(
                relevances,
                candidates,
                smbert,
                config.max_key_phrases(),
                config.gamma(),
            );
            coi.log_reaction();
        }

        // If the embedding is too dissimilar, we create a new CoI instead
        _ => {
            let coi = PositiveCoi::new(Uuid::new_v4(), embedding.clone());
            coi.select_key_phrases(
                relevances,
                candidates,
                smbert,
                config.max_key_phrases(),
                config.gamma(),
            );
            cois.push(coi);
        }
    }
}

/// Updates the positive cois based on the documents data.
fn update_positive_cois(
    cois: &mut Vec<PositiveCoi>,
    docs: &[&dyn CoiSystemData],
    config: &Config,
    relevances: &mut RelevanceMap,
    smbert: impl Copy + Fn(&str) -> Result<Embedding, Error>,
) {
    docs.iter().fold(cois, |cois, doc| {
        log_positive_user_reaction(
            cois,
            &doc.smbert().embedding,
            config,
            relevances,
            smbert,
            &[/* TODO: run KPE on doc */],
        );
        cois
    });
}

/// Updates the negative coi closest to the embedding or creates a new one if it's too far away.
fn log_negative_user_reaction(cois: &mut Vec<NegativeCoi>, embedding: &Embedding, config: &Config) {
    match find_closest_coi_mut(cois, embedding) {
        Some((coi, similarity)) if similarity >= config.threshold() => {
            coi.shift_point(embedding, config.shift_factor());
            coi.log_reaction();
        }
        _ => cois.push(NegativeCoi::new(Uuid::new_v4(), embedding.clone())),
    }
}

/// Updates the negative cois based on the documents data.
fn update_negative_cois(cois: &mut Vec<NegativeCoi>, docs: &[&dyn CoiSystemData], config: &Config) {
    docs.iter().fold(cois, |cois, doc| {
        log_negative_user_reaction(cois, &doc.smbert().embedding, config);
        cois
    });
}

pub(crate) fn update_user_interests(
    mut user_interests: UserInterests,
    relevances: &mut RelevanceMap,
    history: &[DocumentHistory],
    documents: &[&dyn CoiSystemData],
    smbert: impl Copy + Fn(&str) -> Result<Embedding, Error>,
    config: &Config,
) -> Result<UserInterests, Error> {
    let matching_documents = collect_matching_documents(history, documents);

    if matching_documents.is_empty() {
        return Err(CoiSystemError::NoMatchingDocuments.into());
    }

    let (positive_docs, negative_docs) =
        classify_documents_based_on_user_feedback(matching_documents);

    update_positive_cois(
        &mut user_interests.positive,
        &positive_docs,
        config,
        relevances,
        smbert,
    );
    update_negative_cois(&mut user_interests.negative, &negative_docs, config);

    Ok(user_interests)
}

fn log_document_view_time(cois: &mut Vec<PositiveCoi>, embedding: &Embedding, viewed: Duration) {
    if let Some((coi, _)) = find_closest_coi_mut(cois, embedding) {
        coi.log_time(viewed);
    }
}

/// Coi system to run when Coi is disabled
pub struct NeutralCoiSystem;

impl NeutralCoiSystem {
    pub(crate) const COI: CoiComponent = CoiComponent {
        id: CoiId(Uuid::nil()),
        pos_similarity: 0.,
        neg_similarity: 0.,
    };
}

impl systems::CoiSystem for NeutralCoiSystem {
    fn compute_coi(
        &self,
        documents: &[DocumentDataWithSMBert],
        _user_interests: &UserInterests,
    ) -> Result<Vec<DocumentDataWithCoi>, Error> {
        Ok(documents
            .iter()
            .map(|document| DocumentDataWithCoi::from_document(document, Self::COI))
            .collect())
    }

    fn update_user_interests(
        &mut self,
        _history: &[DocumentHistory],
        _documents: &[&dyn CoiSystemData],
        _user_interests: UserInterests,
    ) -> Result<UserInterests, Error> {
        unreachable!(/* should never be called on this system */)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{arr1, FixedInitializer};
    use std::f32::NAN;

    use super::*;
    use crate::{
        coi::{
            utils::tests::{
                create_data_with_embeddings,
                create_document_history,
                create_neg_cois,
                create_pos_cois,
            },
            CoiId,
        },
        data::{
            document::{DocumentId, Relevance, UserFeedback},
            document_data::{
                ContextComponent,
                DocumentBaseComponent,
                DocumentContentComponent,
                DocumentDataWithRank,
                LtrComponent,
                QAMBertComponent,
                RankComponent,
                SMBertComponent,
            },
        },
        utils::to_vec_of_ref_of,
    };
    use test_utils::assert_approx_eq;

    pub(crate) fn create_data_with_rank(
        embeddings: &[impl FixedInitializer<Elem = f32>],
    ) -> Vec<DocumentDataWithRank> {
        embeddings
            .iter()
            .enumerate()
            .map(|(id, embedding)| DocumentDataWithRank {
                document_base: DocumentBaseComponent {
                    id: DocumentId::from_u128(id as u128),
                    initial_ranking: id,
                },
                document_content: DocumentContentComponent {
                    title: id.to_string(),
                    ..DocumentContentComponent::default()
                },
                smbert: SMBertComponent {
                    embedding: arr1(embedding.as_init_slice()).into(),
                },
                qambert: QAMBertComponent { similarity: 0.5 },
                coi: CoiComponent {
                    id: CoiId::mocked(1),
                    pos_similarity: 0.1,
                    neg_similarity: 0.1,
                },
                ltr: LtrComponent { ltr_score: 0.5 },
                context: ContextComponent { context_value: 0.5 },
                rank: RankComponent { rank: 0 },
            })
            .collect()
    }

    #[test]
    fn test_update_coi_add_point() {
        let mut cois = create_pos_cois(&[[1., 0., 0.], [1., 0.2, 1.], [0.5, 0.5, 0.1]]);
        let mut relevances = RelevanceMap::default();
        let embedding = arr1(&[1.91, 73.78, 72.35]).into();
        let config = Config::default();

        let (closest, similarity) = find_closest_coi(&cois, &embedding).unwrap();

        assert_eq!(closest.point, arr1(&[0.5, 0.5, 0.1]));
        assert_approx_eq!(f32, similarity, 0.610_772_5);
        assert!(config.threshold() >= similarity);

        log_positive_user_reaction(
            &mut cois,
            &embedding,
            &config,
            &mut relevances,
            |_| unreachable!(),
            &[],
        );
        assert_eq!(cois.len(), 4);
    }

    #[test]
    fn test_update_coi_update_point() {
        let mut cois = create_pos_cois(&[[1., 1., 1.], [10., 10., 10.], [20., 20., 20.]]);
        let mut relevances = RelevanceMap::default();
        let embedding = arr1(&[2., 3., 4.]).into();
        let config = Config::default();

        let last_view_before = cois[0].stats.last_view;

        log_positive_user_reaction(
            &mut cois,
            &embedding,
            &config,
            &mut relevances,
            |_| unreachable!(),
            &[],
        );

        assert_eq!(cois.len(), 3);
        assert_eq!(cois[0].point, arr1(&[1.1, 1.2, 1.3]));
        assert_eq!(cois[1].point, arr1(&[10., 10., 10.]));
        assert_eq!(cois[2].point, arr1(&[20., 20., 20.]));

        assert_eq!(cois[0].stats.view_count, 2);
        assert!(cois[0].stats.last_view > last_view_before);
    }

    #[test]
    fn test_update_coi_under_similarity_threshold_adds_new_coi() {
        let mut cois = create_pos_cois(&[[0., 1.]]);
        let mut relevances = RelevanceMap::default();
        let embedding = arr1(&[1., 0.]).into();
        let config = Config::default();

        log_positive_user_reaction(
            &mut cois,
            &embedding,
            &config,
            &mut relevances,
            |_| unreachable!(),
            &[],
        );

        assert_eq!(cois.len(), 2);
        assert_eq!(cois[0].point, arr1(&[0., 1.,]));
        assert_eq!(cois[1].point, arr1(&[1., 0.]));
    }

    #[test]
    fn test_update_cois_update_the_same_point_twice() {
        // checks that an updated coi is used in the next iteration
        let mut cois = create_pos_cois(&[[0., 0., 0.]]);
        let mut relevances = RelevanceMap::default();
        let documents = create_data_with_rank(&[[0., 0., 4.9], [0., 0., 5.]]);
        let documents = to_vec_of_ref_of!(documents, &dyn CoiSystemData);
        let config = Config::default();

        update_positive_cois(
            &mut cois,
            &documents,
            &config,
            &mut relevances,
            |_| unreachable!(),
        );

        assert_eq!(cois.len(), 1);
        // updated coi after first embedding = [0., 0., 0.49]
        // updated coi after second embedding = [0., 0., 0.941]
        assert_eq!(cois[0].point, arr1(&[0., 0., 0.941]));
    }

    #[test]
    fn test_compute_coi_for_embedding() {
        let positive = create_pos_cois(&[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]);
        let negative = create_neg_cois(&[[10., 10., 0.], [0., 10., 10.], [10., 0., 10.]]);
        let user_interests = UserInterests { positive, negative };
        let embedding = arr1(&[2., 3., 4.]).into();

        let coi_comp = compute_coi_for_embedding(&embedding, &user_interests).unwrap();

        assert_eq!(coi_comp.id, CoiId::mocked(2));
        assert_approx_eq!(f32, coi_comp.pos_similarity, 0.742_781_34);
        assert_approx_eq!(f32, coi_comp.neg_similarity, 0.919_145_05);
    }

    #[test]
    fn test_compute_coi_for_embedding_empty_negative_cois() {
        let positive_cois = create_pos_cois(&[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]);
        let user_interests = UserInterests {
            positive: positive_cois,
            negative: Vec::new(),
        };
        let embedding = arr1(&[2., 3., 4.]).into();

        let coi_comp = compute_coi_for_embedding(&embedding, &user_interests).unwrap();

        assert_eq!(coi_comp.id, CoiId::mocked(2));
        assert_approx_eq!(f32, coi_comp.pos_similarity, 0.742_781_34);
        assert_approx_eq!(f32, coi_comp.neg_similarity, -1.);
    }

    #[test]
    fn test_compute_coi() {
        let positive = create_pos_cois(&[[3., 2., 1.], [1., 2., 3.]]);
        let negative = create_neg_cois(&[[4., 5., 6.]]);
        let user_interests = UserInterests { positive, negative };
        let documents = create_data_with_embeddings(&[[1., 4., 4.], [3., 6., 6.]]);

        let documents_coi = compute_coi(&documents, &user_interests).unwrap();

        assert_eq!(documents_coi[0].coi.id, CoiId::mocked(1));
        assert_approx_eq!(f32, documents_coi[0].coi.pos_similarity, 0.977_008_4);
        assert_approx_eq!(f32, documents_coi[0].coi.neg_similarity, 0.952_223_54);

        assert_eq!(documents_coi[1].coi.id, CoiId::mocked(1));
        assert_approx_eq!(f32, documents_coi[1].coi.pos_similarity, 0.979_957_9);
        assert_approx_eq!(f32, documents_coi[1].coi.neg_similarity, 0.987_658_3);
    }

    #[test]
    #[should_panic(expected = "vector must consist of real values only")]
    fn test_compute_coi_all_nan() {
        let positive = create_pos_cois(&[[3., 2., 1.], [1., 2., 3.]]);
        let negative = create_neg_cois(&[[4., 5., 6.]]);
        let user_interests = UserInterests { positive, negative };
        let documents = create_data_with_embeddings(&[[NAN, NAN, NAN]]);
        let _ = compute_coi(&documents, &user_interests);
    }

    #[test]
    #[should_panic(expected = "vector must consist of real values only")]
    fn test_compute_coi_single_nan() {
        let positive = create_pos_cois(&[[3., 2., 1.], [1., 2., 3.]]);
        let negative = create_neg_cois(&[[4., 5., 6.]]);
        let user_interests = UserInterests { positive, negative };
        let documents = create_data_with_embeddings(&[[1., NAN, 2.]]);
        let _ = compute_coi(&documents, &user_interests);
    }

    #[test]
    fn test_update_user_interests() {
        let positive = create_pos_cois(&[[3., 2., 1.], [1., 2., 3.]]);
        let negative = create_neg_cois(&[[4., 5., 6.]]);
        let user_interests = UserInterests { positive, negative };
        let mut relevances = RelevanceMap::default();
        let history = create_document_history(vec![
            (Relevance::Low, UserFeedback::Irrelevant),
            (Relevance::Low, UserFeedback::Relevant),
            (Relevance::Low, UserFeedback::Relevant),
        ]);
        let documents = create_data_with_rank(&[[1., 4., 4.], [4., 47., 4.], [1., 1., 1.]]);
        let documents = to_vec_of_ref_of!(documents, &dyn CoiSystemData);
        let config = Config::default();

        let UserInterests { positive, negative } = update_user_interests(
            user_interests,
            &mut relevances,
            &history,
            &documents,
            |_| todo!(/* mock once KPE is used */),
            &config,
        )
        .unwrap();

        assert_eq!(positive.len(), 3);
        assert_eq!(positive[0].point, arr1(&[2.7999997, 1.9, 1.]));
        assert_eq!(positive[1].point, arr1(&[1., 2., 3.]));
        assert_eq!(positive[2].point, arr1(&[4., 47., 4.]));

        assert_eq!(negative.len(), 1);
        assert_eq!(negative[0].point, arr1(&[3.6999998, 4.9, 5.7999997]));
    }

    #[test]
    fn test_update_user_interests_no_matches() {
        let error = update_user_interests(
            UserInterests::default(),
            &mut RelevanceMap::default(),
            &[],
            &[],
            |_| unreachable!(),
            &Config::default(),
        )
        .err()
        .unwrap();
        let error = error.downcast::<CoiSystemError>().unwrap();

        assert!(matches!(error, CoiSystemError::NoMatchingDocuments));
    }

    #[test]
    fn test_log_negative_user_reaction_last_view() {
        let mut cois = create_neg_cois(&[[1., 2., 3.]]);
        let config = Config::default();
        let before = cois[0].last_view;
        log_negative_user_reaction(&mut cois, &arr1(&[1., 2., 4.]).into(), &config);
        assert!(cois[0].last_view > before);
        assert_eq!(cois.len(), 1);
    }

    #[test]
    fn test_log_document_view_time() {
        let mut cois = create_pos_cois(&[[1., 2., 3.]]);

        log_document_view_time(
            &mut cois,
            &arr1(&[1., 2., 4.]).into(),
            Duration::from_secs(10),
        );
        assert_eq!(Duration::from_secs(10), cois[0].stats.view_time);

        log_document_view_time(
            &mut cois,
            &arr1(&[1., 2., 4.]).into(),
            Duration::from_secs(10),
        );
        assert_eq!(Duration::from_secs(20), cois[0].stats.view_time);
    }
}
