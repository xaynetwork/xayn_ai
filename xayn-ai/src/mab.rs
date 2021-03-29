#![allow(dead_code)]

use crate::{
    data::{
        document_data::{DocumentDataWithContext, DocumentDataWithMab, MabComponent},
        Coi,
        CoiId,
        UserInterests,
    },
    reranker_systems::MabSystem,
    Error,
};

use std::{
    cmp::Ordering,
    collections::{hash_map::Entry, BinaryHeap, HashMap},
};

use displaydoc::Display;
use rand_distr::{Beta, BetaError, Distribution};
use thiserror::Error;

#[cfg(test)]
use mockall::{automock, Sequence};

#[derive(Error, Debug, Display)]
pub enum MabError {
    /// The coi id assigned to a document does not exist
    DocumentCoiDoesNotExist,
    /// No documents to pull
    NoDocumentsToPull,
    /// Extracted coi does not have documents
    ExtractedCoiNoDocuments,
    /// Error while sampling
    Sampling(#[from] BetaError),
    /// Context value must be [0, 1]
    InvalidContext,
}

#[cfg_attr(test, automock)]
pub trait BetaSample {
    fn sample(&self, alpha: f32, beta: f32) -> Result<f32, MabError>;
}

/// Sample a value from a beta distribution
pub struct BetaSampler;

impl BetaSample for BetaSampler {
    fn sample(&self, alpha: f32, beta: f32) -> Result<f32, MabError> {
        Ok(Beta::new(alpha, beta)?.sample(&mut rand::thread_rng()))
    }
}

/// Pretend that comparing two f32 is total. The function will rank `NaN`
/// as the lowest value, similar to what [`f32::max`] does.
fn f32_total_cmp(a: &f32, b: &f32) -> Ordering {
    a.partial_cmp(&b).unwrap_or_else(|| {
        // if `partial_cmp` returns None we have at least one `NaN`,
        // we treat it as the lowest value
        match (a.is_nan(), b.is_nan()) {
            (true, true) => Ordering::Equal,
            (true, _) => Ordering::Less,
            (_, true) => Ordering::Greater,
            _ => unreachable!("partial_cmp returned None but both numbers are not NaN"),
        }
    })
}

/// Wrapper to order documents by `context_value`.
/// We need to implement `Ord` to use it in the `BinaryHeap`.
#[cfg_attr(test, derive(Debug, Clone))]
struct DocumentByContext(DocumentDataWithContext);

impl PartialEq for DocumentByContext {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for DocumentByContext {}

impl PartialOrd for DocumentByContext {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DocumentByContext {
    fn cmp(&self, other: &Self) -> Ordering {
        f32_total_cmp(
            &self.0.context.context_value,
            &other.0.context.context_value,
        )
    }
}

type DocumentsByCoi = HashMap<CoiId, BinaryHeap<DocumentByContext>>;

/// Group documents by coi and implicitly order them by context_value in the heap
fn group_by_coi(documents: Vec<DocumentDataWithContext>) -> DocumentsByCoi {
    documents
        .into_iter()
        .fold(DocumentsByCoi::new(), |mut groups, document| {
            let coi_id = document.coi.id;

            let document = DocumentByContext(document);

            match groups.entry(coi_id) {
                Entry::Occupied(mut entry) => {
                    entry.get_mut().push(document);
                }
                Entry::Vacant(entry) => {
                    let mut heap = BinaryHeap::new();
                    heap.push(document);
                    entry.insert(heap);
                }
            }

            groups
        })
}

// Here we implement the algorithm described at page 9 of:
// http://www.ecmlpkdd2018.org/wp-content/uploads/2018/09/723.pdf
// We do not update all context_value like they do in the paper.

/// Update `alpha` and `beta` values based on the `context_value` of document in that coi.
/// The `context_value` must be between 0 and 1 otherwise `MabError::InvalidContext` will be returned.
/// The updated `alpha` and `beta` will be always > 0.
fn update_cois(
    cois: HashMap<CoiId, Coi>,
    documents: &[DocumentDataWithContext],
) -> Result<HashMap<CoiId, Coi>, MabError> {
    documents.iter().try_fold(cois, |mut cois, document| {
        let coi = cois
            .get_mut(&document.coi.id)
            .ok_or(MabError::DocumentCoiDoesNotExist)?;

        let context_value = document.context.context_value;
        if !((0.)..=1.).contains(&context_value) {
            return Err(MabError::InvalidContext);
        }

        coi.alpha += context_value;
        coi.beta += 1. - context_value;

        Ok(cois)
    })
}

/// For each coi we take a sample from the beta distribution and we pick
/// the coi with the biggest sample. Then we take the document with the biggest `context_value` among
/// the documents within that coi.
fn pull_arms(
    beta_sampler: &impl BetaSample,
    cois: &HashMap<CoiId, Coi>,
    mut documents_by_coi: DocumentsByCoi,
) -> Result<(DocumentsByCoi, DocumentDataWithContext), MabError> {
    let sample_from_coi = |coi_id: &CoiId| {
        let coi = cois.get(&coi_id).ok_or(MabError::DocumentCoiDoesNotExist)?;
        beta_sampler.sample(coi.alpha, coi.beta)
    };

    let mut coi_id_it = documents_by_coi.keys();

    let first_coi_id = coi_id_it.next().ok_or(MabError::NoDocumentsToPull)?;
    let first_sample = sample_from_coi(first_coi_id)?;

    let coi_id = *coi_id_it
        .try_fold(
            (first_sample, first_coi_id),
            |max, coi_id| -> Result<_, MabError> {
                let sample = sample_from_coi(coi_id)?;

                if let Ordering::Greater = f32_total_cmp(&sample, &max.0) {
                    Ok((sample, coi_id))
                } else {
                    Ok(max)
                }
            },
        )?
        .1;

    if let Entry::Occupied(mut entry) = documents_by_coi.entry(coi_id) {
        let heap = entry.get_mut();
        let document = heap.pop().ok_or(MabError::ExtractedCoiNoDocuments)?;
        // remove coi when they have no documents left
        if heap.is_empty() {
            entry.remove_entry();
        }

        Ok((documents_by_coi, document.0))
    } else {
        // This should never occur because we select a `coi_id`
        // from the keys of `documents_by_coi`
        Err(MabError::ExtractedCoiNoDocuments)
    }
}

struct MabRankingIter<'bs, 'cois, BS> {
    beta_sampler: &'bs BS,
    cois: &'cois HashMap<CoiId, Coi>,
    documents_by_coi: DocumentsByCoi,
}

impl<'bs, 'cois, BS> MabRankingIter<'bs, 'cois, BS> {
    fn new(
        beta_sampler: &'bs BS,
        cois: &'cois HashMap<CoiId, Coi>,
        documents_by_coi: DocumentsByCoi,
    ) -> Self {
        Self {
            beta_sampler,
            cois,
            documents_by_coi,
        }
    }
}

impl<'bs, 'cois, BS> Iterator for MabRankingIter<'bs, 'cois, BS>
where
    BS: BetaSample,
{
    type Item = Result<DocumentDataWithContext, MabError>;

    fn next(&mut self) -> Option<Self::Item> {
        if !self.documents_by_coi.is_empty() {
            let mut documents_by_coi = HashMap::new();
            // take out self.documents_by_coi from &mut
            std::mem::swap(&mut self.documents_by_coi, &mut documents_by_coi);

            Some(
                pull_arms(self.beta_sampler, self.cois, documents_by_coi).map(
                    |(new_documents_by_coi, document)| {
                        self.documents_by_coi = new_documents_by_coi;
                        document
                    },
                ),
            )
        } else {
            None
        }
    }
}

pub struct MabRanking<BS> {
    beta_sampler: BS,
}

impl<BS> MabRanking<BS> {
    pub fn new(beta_sampler: BS) -> Self {
        Self { beta_sampler }
    }
}

impl<BS> MabSystem for MabRanking<BS>
where
    BS: BetaSample,
{
    fn compute_mab(
        &self,
        documents: Vec<DocumentDataWithContext>,
        mut user_interests: UserInterests,
    ) -> Result<(Vec<DocumentDataWithMab>, UserInterests), Error> {
        let cois = user_interests
            .positive
            .into_iter()
            .map(|coi| (coi.id, coi))
            .collect();
        let cois = update_cois(cois, &documents)?;

        let documents_by_coi = group_by_coi(documents);

        let mab_rerank = MabRankingIter::new(&self.beta_sampler, &cois, documents_by_coi);
        let documents = mab_rerank
            .enumerate()
            .map(|(rank, document)| {
                document.map(|document| {
                    DocumentDataWithMab::from_document(document, MabComponent { rank })
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        user_interests.positive = cois.into_iter().map(|(_, coi)| coi).collect();
        Ok((documents, user_interests))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::data::{
        document::DocumentId,
        document_data::{
            CoiComponent,
            ContextComponent,
            DocumentIdComponent,
            EmbeddingComponent,
            LtrComponent,
        },
    };
    use rubert::ndarray::arr1;

    use float_cmp::approx_eq;
    use maplit::hashmap;
    use std::collections::HashSet;

    fn with_ctx(id: DocumentId, coi_id: CoiId, context_value: f32) -> DocumentDataWithContext {
        DocumentDataWithContext {
            document_id: DocumentIdComponent { id },
            embedding: EmbeddingComponent {
                embedding: arr1(&[]).into(),
            },
            coi: CoiComponent {
                id: coi_id,
                pos_distance: 0.,
                neg_distance: 0.,
            },
            ltr: LtrComponent { ltr_score: 0.5 },
            context: ContextComponent { context_value },
        }
    }

    macro_rules! coi {
        ($id:expr) => {
            coi!($id, 1.)
        };
        ($id:expr, $params: expr) => {
            coi!($id, $params, $params)
        };
        ($id:expr, $alpha: expr, $beta: expr) => {
            Coi {
                id: $id,
                alpha: $alpha,
                beta: $beta,
                point: arr1(&[]).into(),
            }
        };
    }

    #[test]
    fn test_group_by_coi_empty() {
        let group = group_by_coi(vec![]);
        assert!(group.is_empty());
    }

    #[test]
    fn test_group_by_coi_group_ok() {
        let doc_id_0 = DocumentId("0".to_string());
        let doc_id_1 = DocumentId("1".to_string());
        let doc_id_2 = DocumentId("2".to_string());
        let doc_id_3 = DocumentId("3".to_string());
        let doc_id_4 = DocumentId("4".to_string());

        let group = group_by_coi(vec![
            with_ctx(doc_id_0.clone(), CoiId(0), 0.),
            with_ctx(doc_id_1.clone(), CoiId(4), 0.),
            with_ctx(doc_id_2.clone(), CoiId(9), 0.),
            with_ctx(doc_id_3.clone(), CoiId(4), 0.),
            with_ctx(doc_id_4.clone(), CoiId(9), 0.),
        ]);

        let check_contains = |coi_id: CoiId, docs_id_ok: Vec<DocumentId>| {
            let docs = group
                .get(&coi_id)
                .unwrap_or_else(|| panic!("document from coi id {:?}", coi_id));
            let docs_id: HashSet<DocumentId> = docs
                .iter()
                .map(|doc| doc.0.document_id.id.clone())
                .collect();

            assert_eq!(docs_id.len(), docs_id_ok.len());
            for doc_id in docs_id_ok {
                assert!(docs_id.contains(&doc_id));
            }
        };

        check_contains(CoiId(0), vec![doc_id_0]);
        check_contains(CoiId(4), vec![doc_id_1, doc_id_3]);
        check_contains(CoiId(9), vec![doc_id_2, doc_id_4]);
    }

    #[test]
    fn test_group_by_coi_order_by_context_value_desc() {
        let doc_id_0 = DocumentId("0".to_string());
        let doc_id_1 = DocumentId("1".to_string());
        let doc_id_2 = DocumentId("2".to_string());
        let doc_id_3 = DocumentId("3".to_string());
        let doc_id_4 = DocumentId("4".to_string());

        let mut group = group_by_coi(vec![
            with_ctx(doc_id_0.clone(), CoiId(0), 0.4),
            with_ctx(doc_id_1.clone(), CoiId(0), 0.8),
            with_ctx(doc_id_2.clone(), CoiId(0), 0.2),
            with_ctx(doc_id_3.clone(), CoiId(0), 0.9),
            with_ctx(doc_id_4.clone(), CoiId(0), 0.6),
        ]);

        let docs = group.remove(&CoiId(0)).expect("document from coi id");
        let docs_id: Vec<DocumentId> = docs
            .into_sorted_vec()
            .into_iter()
            // into_sorted_vec returns elements in the revers order of what using pop will do
            .rev()
            .map(|doc| doc.0.document_id.id)
            .collect();

        assert_eq!(
            docs_id,
            vec![doc_id_3, doc_id_1, doc_id_4, doc_id_0, doc_id_2]
        );
    }

    #[test]
    fn test_group_by_coi_context_value_nan_is_min() {
        let doc_id_0 = DocumentId("0".to_string());
        let doc_id_1 = DocumentId("1".to_string());
        let doc_id_2 = DocumentId("2".to_string());

        let mut group = group_by_coi(vec![
            with_ctx(doc_id_0.clone(), CoiId(0), 0.2),
            with_ctx(doc_id_1.clone(), CoiId(0), f32::NAN),
            with_ctx(doc_id_2.clone(), CoiId(0), 0.8),
        ]);

        let docs = group.remove(&CoiId(0)).expect("document from coi id");
        let docs_id: Vec<DocumentId> = docs
            .into_sorted_vec()
            .into_iter()
            // into_sorted_vec returns elements in the revers order of what using pop will do
            .rev()
            .map(|doc| doc.0.document_id.id)
            .collect();

        assert_eq!(docs_id, vec![doc_id_2, doc_id_0, doc_id_1]);
    }

    #[test]
    fn test_update_coi_empty() {
        let cois = update_cois(HashMap::new(), &[]).expect("cois");
        assert!(cois.is_empty());
    }

    #[test]
    fn test_update_coi_no_docs() {
        let cois = hashmap! {
            CoiId(0) => coi!(CoiId(0)),
            CoiId(1) => coi!(CoiId(1))
        };

        let new_cois = update_cois(cois.clone(), &[]).expect("cois");
        assert_eq!(cois, new_cois);
    }

    #[test]
    fn test_update_coi_no_coi() {
        let error = update_cois(
            HashMap::new(),
            &[with_ctx(DocumentId("0".to_string()), CoiId(0), 0.)],
        )
        .expect_err("no coi");
        assert!(matches!(error, MabError::DocumentCoiDoesNotExist));
    }

    #[test]
    fn test_update_coi_invalid_context_value() {
        let cois = hashmap! {
            CoiId(0) => coi!(CoiId(0), 0.91),
        };

        let error = update_cois(
            cois.clone(),
            &[
                with_ctx(DocumentId("".to_string()), CoiId(0), 0.35),
                with_ctx(DocumentId("".to_string()), CoiId(0), -1.),
            ],
        )
        .expect_err("invalid context value");
        assert!(matches!(error, MabError::InvalidContext));

        let error = update_cois(
            cois,
            &[
                with_ctx(DocumentId("".to_string()), CoiId(0), 0.35),
                with_ctx(DocumentId("".to_string()), CoiId(0), 1.01),
            ],
        )
        .expect_err("invalid context value");
        assert!(matches!(error, MabError::InvalidContext));
    }

    #[test]
    fn test_update_coi_ok() {
        let cois = hashmap! {
            CoiId(0) => coi!(CoiId(0), 0.91),
            CoiId(1) => coi!(CoiId(1), 0.27)
        };

        let cois = update_cois(
            cois,
            &vec![
                with_ctx(DocumentId("".to_string()), CoiId(1), 1.),
                with_ctx(DocumentId("".to_string()), CoiId(0), 0.35),
                with_ctx(DocumentId("".to_string()), CoiId(1), 0.2),
                with_ctx(DocumentId("".to_string()), CoiId(0), 0.6),
            ],
        )
        .expect("cois");

        // alpha is updated with `alpha += context_value`
        // beta is updated with `beta += (1. - context_value)`

        let coi = cois.get(&CoiId(0)).expect("coi");
        assert!(approx_eq!(f32, coi.alpha, 1.86));
        assert!(approx_eq!(f32, coi.beta, 1.96));

        let coi = cois.get(&CoiId(1)).expect("coi");
        assert!(approx_eq!(f32, coi.alpha, 1.47));
        assert!(approx_eq!(f32, coi.beta, 1.07));
    }

    #[test]
    fn test_pull_arms_coi_empty() {
        let documents_by_coi = group_by_coi(vec![
            with_ctx(DocumentId("0".to_string()), CoiId(0), 0.),
            with_ctx(DocumentId("1".to_string()), CoiId(1), 0.),
        ]);

        let beta_sampler = MockBetaSample::new();

        let error =
            pull_arms(&beta_sampler, &HashMap::new(), documents_by_coi).expect_err("no coi");
        assert!(matches!(error, MabError::DocumentCoiDoesNotExist));
    }

    #[test]
    fn test_pull_arms_no_coi() {
        let cois = hashmap! {
            CoiId(0) => coi!(CoiId(0), 0.91),
        };

        let documents_by_coi =
            group_by_coi(vec![with_ctx(DocumentId("1".to_string()), CoiId(1), 0.)]);

        let beta_sampler = MockBetaSample::new();

        let error = pull_arms(&beta_sampler, &cois, documents_by_coi).expect_err("no coi");

        assert!(matches!(error, MabError::DocumentCoiDoesNotExist));
    }

    #[test]
    fn test_pull_arms_documents_empty() {
        let beta_sampler = MockBetaSample::new();

        let error = pull_arms(&beta_sampler, &HashMap::new(), DocumentsByCoi::new())
            .expect_err("no documents");
        assert!(matches!(error, MabError::NoDocumentsToPull));

        let cois = hashmap! {
            CoiId(0) => coi!(CoiId(0), 0.91),
        };

        let error =
            pull_arms(&beta_sampler, &cois, DocumentsByCoi::new()).expect_err("no documents");
        assert!(matches!(error, MabError::NoDocumentsToPull));
    }

    #[test]
    fn test_pull_arms_sampler_error() {
        let cois = hashmap! {
            CoiId(0) => coi!(CoiId(0), 0.91),
            CoiId(1) => coi!(CoiId(1), 0.1),
        };

        let documents_by_coi = group_by_coi(vec![
            with_ctx(DocumentId("1".to_string()), CoiId(0), 0.),
            with_ctx(DocumentId("2".to_string()), CoiId(1), 0.),
        ]);

        let mut beta_sampler = MockBetaSample::new();
        beta_sampler
            .expect_sample()
            .returning(|_, _| Err(MabError::Sampling(BetaError::AlphaTooSmall)));

        let error =
            pull_arms(&beta_sampler, &cois, documents_by_coi.clone()).expect_err("sampler error");
        assert!(matches!(error, MabError::Sampling(_)));

        // the current implementation takes a sample for one coi and then enters a loop where
        // all other samples are taken. Here we fail in the loop.
        let mut seq = Sequence::new();
        let mut beta_sampler = MockBetaSample::new();
        beta_sampler
            .expect_sample()
            .times(1)
            .in_sequence(&mut seq)
            .returning(|alpha, _| Ok(alpha));

        beta_sampler
            .expect_sample()
            .times(1)
            .in_sequence(&mut seq)
            .returning(|_, _| Err(MabError::Sampling(BetaError::AlphaTooSmall)));

        let error = pull_arms(&beta_sampler, &cois, documents_by_coi).expect_err("sampler error");

        assert!(matches!(error, MabError::Sampling(_)));
    }

    #[test]
    fn test_pull_arms_malformed_documents_by_coi() {
        let cois = hashmap! {
            CoiId(0) => coi!(CoiId(0), 0.91),
        };

        let mut documents_by_coi = DocumentsByCoi::new();
        // If `group_by_coi` and `pull_arms` are behaving correctly we will never have
        // a coi with an empty heap.
        documents_by_coi.insert(CoiId(0), BinaryHeap::new());

        let mut beta_sampler = MockBetaSample::new();
        beta_sampler.expect_sample().returning(|_, _| Ok(0.2));

        let error = pull_arms(&beta_sampler, &cois, documents_by_coi).expect_err("sampler error");
        assert!(matches!(error, MabError::ExtractedCoiNoDocuments));
    }

    #[test]
    fn test_pull_arms_multiple_coi_ok() {
        let doc_id_0 = DocumentId("0".to_string());
        let doc_id_1 = DocumentId("1".to_string());
        let doc_id_2 = DocumentId("2".to_string());
        let doc_id_3 = DocumentId("3".to_string());
        let doc_id_4 = DocumentId("4".to_string());
        let doc_id_5 = DocumentId("5".to_string());

        let cois = hashmap! {
            CoiId(0) => coi!(CoiId(0), 0.1),
            CoiId(4) => coi!(CoiId(4), 0.5),
            CoiId(7) => coi!(CoiId(7), 0.8),
        };

        let documents_by_coi = group_by_coi(vec![
            with_ctx(doc_id_0.clone(), CoiId(0), 0.2),
            with_ctx(doc_id_1.clone(), CoiId(0), 0.5),
            with_ctx(doc_id_2.clone(), CoiId(0), 0.7),
            with_ctx(doc_id_3.clone(), CoiId(4), 0.4),
            with_ctx(doc_id_4.clone(), CoiId(4), 0.7),
            with_ctx(doc_id_5.clone(), CoiId(7), 0.2),
        ]);

        let mut beta_sampler = MockBetaSample::new();
        beta_sampler
            .expect_sample()
            .returning(|alpha, beta| Ok(alpha + beta));

        let documents_id = vec![doc_id_5, doc_id_4, doc_id_3, doc_id_2, doc_id_1, doc_id_0];

        let documents_by_coi =
            documents_id
                .into_iter()
                .fold(documents_by_coi, |documents_by_coi, doc_id| {
                    let (documents_by_coi, document) =
                        pull_arms(&beta_sampler, &cois, documents_by_coi).expect("document");

                    assert_eq!(doc_id, document.document_id.id);

                    documents_by_coi
                });

        assert!(documents_by_coi.is_empty());
    }

    #[test]
    fn test_pull_arms_multiple_coi_custom_sampler() {
        let doc_id_0 = DocumentId("0".to_string());
        let doc_id_1 = DocumentId("1".to_string());
        let doc_id_2 = DocumentId("2".to_string());
        let doc_id_3 = DocumentId("3".to_string());
        let doc_id_4 = DocumentId("4".to_string());
        let doc_id_5 = DocumentId("5".to_string());

        let coi1 = 1.;
        let coi4 = 4.;
        let coi7 = 7.;
        let cois = hashmap! {
            CoiId(1) => coi!(CoiId(1), coi1),
            CoiId(4) => coi!(CoiId(4), coi4),
            CoiId(7) => coi!(CoiId(7), coi7),
        };

        let documents_by_coi = group_by_coi(vec![
            with_ctx(doc_id_0.clone(), CoiId(1), 0.2),
            with_ctx(doc_id_1.clone(), CoiId(1), 0.5),
            with_ctx(doc_id_2.clone(), CoiId(1), 0.7),
            with_ctx(doc_id_3.clone(), CoiId(4), 0.4),
            with_ctx(doc_id_4.clone(), CoiId(4), 0.7),
            with_ctx(doc_id_5.clone(), CoiId(7), 0.2),
        ]);

        let mut coi_counter = 0;
        let mut beta_sampler = MockBetaSample::new();
        beta_sampler.expect_sample().returning(move |alpha, _| {
            // `pull_arms` calls sample once per coi, we have 3 cois till the last calls
            // where cois will be removed when we pull their last document.
            // We can then use `coi_counter / 3` to know how many times `pull_arms` has
            // been called before, we use this number to decide which coi should win in this "round".
            // `alpha` value is used to understand for which coi we are pulling.
            // We alternate CoiId(1) and CoiId(4) and we will serve CoiId(7) as last
            // to always have 3 cois to sample and make it easier to understand
            // the round we are in.
            let win = 0.9;
            let lose = 0.1;

            let make_coi_win = |coi_alpha: f32| -> f32 {
                #[allow(clippy::clippy::float_cmp)] // alpha is set by us and never changed
                if alpha == coi_alpha {
                    win
                } else {
                    loose
                }
            };

            let sample = match coi_counter / 3 {
                // CoiId(1) wins
                0 => make_coi_win(coi1),
                // CoiId(4) wins
                1 => make_coi_win(coi4),
                // CoiId(1) wins
                2 => make_coi_win(coi1),
                // CoiId(4) wins
                3 => make_coi_win(coi4),
                // This round involves 2 calls to pull_arms. In the first coi1 and coi7 are
                // present and coi1 wins, in the second there is only coi7 and any value will
                // make it win.
                4 => make_coi_win(coi1),
                _ => panic!("too many rounds"),
            };

            coi_counter += 1;
            Ok(sample)
        });

        let documents_id = vec![doc_id_2, doc_id_4, doc_id_1, doc_id_3, doc_id_0, doc_id_5];

        let documents_by_coi =
            documents_id
                .into_iter()
                .fold(documents_by_coi, |documents_by_coi, doc_id| {
                    let (documents_by_coi, document) =
                        pull_arms(&beta_sampler, &cois, documents_by_coi).expect("document");

                    assert_eq!(doc_id, document.document_id.id);

                    documents_by_coi
                });

        assert!(documents_by_coi.is_empty());
    }

    #[test]
    fn test_pull_arms_multiple_coi_real_sampler() {
        let doc_id_0 = DocumentId("0".to_string());
        let doc_id_1 = DocumentId("1".to_string());
        let doc_id_2 = DocumentId("2".to_string());
        let doc_id_3 = DocumentId("3".to_string());
        let doc_id_4 = DocumentId("4".to_string());

        let cois = hashmap! {
            // high probability of low values
            CoiId(0) => coi!(CoiId(0), 2., 8.),
            // high probability of high values
            CoiId(4) => coi!(CoiId(4), 8., 2.),
        };

        let documents_by_coi = group_by_coi(vec![
            with_ctx(doc_id_0.clone(), CoiId(0), 0.2),
            with_ctx(doc_id_1.clone(), CoiId(0), 0.5),
            with_ctx(doc_id_2.clone(), CoiId(0), 0.7),
            with_ctx(doc_id_3.clone(), CoiId(4), 0.4),
            with_ctx(doc_id_4.clone(), CoiId(4), 0.7),
        ]);

        let beta_sampler = BetaSampler;

        let n = 100;
        let mut ok_counter = 0;

        // we test multiple time since we have a random component in the test
        for _ in 0..n {
            let documents_id = vec![
                doc_id_4.clone(),
                doc_id_3.clone(),
                doc_id_2.clone(),
                doc_id_1.clone(),
                doc_id_0.clone(),
            ];

            let (ok, documents_by_coi) = documents_id.into_iter().fold(
                (true, documents_by_coi.clone()),
                |(ok, documents_by_coi), doc_id| {
                    let (documents_by_coi, document) =
                        pull_arms(&beta_sampler, &cois, documents_by_coi).expect("document");

                    let ok = ok && doc_id == document.document_id.id;

                    (ok, documents_by_coi)
                },
            );

            assert!(documents_by_coi.is_empty());

            ok_counter += ok as u32;
        }

        // if we got more ok that ko we are ok
        assert!(ok_counter as f32 > n as f32 * 0.7);
    }

    #[test]
    fn test_mab_ranking_iter_ok() {
        let doc_id_0 = DocumentId("0".to_string());
        let doc_id_1 = DocumentId("1".to_string());
        let doc_id_2 = DocumentId("2".to_string());
        let doc_id_3 = DocumentId("3".to_string());
        let doc_id_4 = DocumentId("4".to_string());
        let doc_id_5 = DocumentId("5".to_string());

        let cois = hashmap! {
            CoiId(0) => coi!(CoiId(0), 0.1),
            CoiId(4) => coi!(CoiId(4), 0.5),
            CoiId(7) => coi!(CoiId(7), 0.8),
        };

        let documents_by_coi = group_by_coi(vec![
            with_ctx(doc_id_0.clone(), CoiId(0), 0.2),
            with_ctx(doc_id_1.clone(), CoiId(0), 0.5),
            with_ctx(doc_id_2.clone(), CoiId(0), 0.7),
            with_ctx(doc_id_3.clone(), CoiId(4), 0.4),
            with_ctx(doc_id_4.clone(), CoiId(4), 0.7),
            with_ctx(doc_id_5.clone(), CoiId(7), 0.2),
        ]);

        let mut beta_sampler = MockBetaSample::new();
        beta_sampler
            .expect_sample()
            .returning(|alpha, beta| Ok(alpha + beta));

        let mab_rerank = MabRankingIter::new(&beta_sampler, &cois, documents_by_coi);

        let documents = mab_rerank
            .collect::<Result<Vec<_>, _>>()
            .expect("documents");
        let documents_id: Vec<_> = documents
            .into_iter()
            .map(|document| document.document_id.id)
            .collect();

        let documents_id_ok = vec![doc_id_5, doc_id_4, doc_id_3, doc_id_2, doc_id_1, doc_id_0];

        assert_eq!(documents_id, documents_id_ok);
    }

    #[test]
    fn test_mab_ranking_iter_empty_documents() {
        let beta_sampler = MockBetaSample::new();

        let cois = HashMap::new();
        let mut mab_rerank = MabRankingIter::new(&beta_sampler, &cois, DocumentsByCoi::new());

        assert!(mab_rerank.next().is_none());
    }

    #[test]
    fn test_mab_ranking_iter_propagate_errors() {
        let documents_by_coi =
            group_by_coi(vec![with_ctx(DocumentId("0".to_string()), CoiId(0), 0.)]);

        let beta_sampler = MockBetaSample::new();

        let cois = HashMap::new();
        let mab_rerank = MabRankingIter::new(&beta_sampler, &cois, documents_by_coi.clone());
        assert!(mab_rerank.collect::<Result<Vec<_>, _>>().is_err());

        let cois = hashmap! {
            CoiId(9) => coi!(CoiId(9), 0.1),
        };
        let mab_rerank = MabRankingIter::new(&beta_sampler, &cois, documents_by_coi.clone());
        assert!(mab_rerank.collect::<Result<Vec<_>, _>>().is_err());

        let mut beta_sampler = MockBetaSample::new();
        beta_sampler
            .expect_sample()
            .returning(|_, _| Err(MabError::Sampling(BetaError::AlphaTooSmall)));
        let mab_rerank = MabRankingIter::new(&beta_sampler, &cois, documents_by_coi);
        assert!(mab_rerank.collect::<Result<Vec<_>, _>>().is_err());

        let mut documents_by_coi = DocumentsByCoi::new();
        documents_by_coi.insert(CoiId(0), BinaryHeap::new());
        let mab_rerank = MabRankingIter::new(&beta_sampler, &cois, documents_by_coi);
        assert!(mab_rerank.collect::<Result<Vec<_>, _>>().is_err());
    }

    #[test]
    fn test_mab_ranking_ok() {
        let doc_id_0 = DocumentId("0".to_string());
        let doc_id_1 = DocumentId("1".to_string());
        let doc_id_2 = DocumentId("2".to_string());
        let doc_id_3 = DocumentId("3".to_string());
        let doc_id_4 = DocumentId("4".to_string());
        let doc_id_5 = DocumentId("5".to_string());

        let mut user_interests = UserInterests::new();
        user_interests.positive = vec![
            coi!(CoiId(0), 1.),
            coi!(CoiId(4), 10.),
            coi!(CoiId(7), 100.),
        ];

        // we use a small context_value to avoid changing alpha and beta too much
        let documents = vec![
            with_ctx(doc_id_0.clone(), CoiId(0), 0.01),
            with_ctx(doc_id_1.clone(), CoiId(0), 0.02),
            with_ctx(doc_id_2.clone(), CoiId(0), 0.03),
            with_ctx(doc_id_3.clone(), CoiId(4), 0.01),
            with_ctx(doc_id_4.clone(), CoiId(4), 0.02),
            with_ctx(doc_id_5.clone(), CoiId(7), 0.01),
        ];

        let mut beta_sampler = MockBetaSample::new();
        beta_sampler
            .expect_sample()
            .returning(|alpha, beta| Ok(alpha + beta));

        let system = MabRanking::new(beta_sampler);
        let (documents, user_interests) = system
            .compute_mab(documents, user_interests)
            .expect("documents");

        let documents_id_to_rank = vec![doc_id_5, doc_id_4, doc_id_3, doc_id_2, doc_id_1, doc_id_0]
            .into_iter()
            .enumerate()
            .map(|(rank, id)| (id, rank))
            .collect::<HashMap<_, _>>();

        for document in documents {
            let rank = documents_id_to_rank
                .get(&document.document_id.id)
                .expect("rank");
            assert_eq!(document.mab.rank, *rank);
        }

        let cois = user_interests
            .positive
            .iter()
            .map(|coi| (coi.id, coi))
            .collect::<HashMap<_, _>>();

        let coi = cois.get(&CoiId(0)).expect("coi");
        assert!(approx_eq!(f32, coi.alpha, 1.06));
        assert!(approx_eq!(f32, coi.beta, 3.94));

        let coi = cois.get(&CoiId(4)).expect("coi");
        assert!(approx_eq!(f32, coi.alpha, 10.03));
        assert!(approx_eq!(f32, coi.beta, 11.97));

        let coi = cois.get(&CoiId(7)).expect("coi");
        assert!(approx_eq!(f32, coi.alpha, 100.01));
        assert!(approx_eq!(f32, coi.beta, 100.99));
    }
}
