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
use rand_distr::{Beta, Distribution};
use thiserror::Error;

#[derive(Error, Debug, Display)]
pub enum MabError {
    /// The coi id assigned to a document does not exist
    DocumentCoiDoesNotExist,
    /// No documents to pull
    NoDocumentsToPull,
    /// Extracted coi does not have documents
    ExtractedCoiNoDocuments,
}

/// Sample a value from a beta distribution
pub struct BetaSampler;

impl BetaSampler {
    fn sample(&self, alpha: f32, beta: f32) -> Result<f32, Error> {
        let beta = Beta::new(alpha, beta)?;
        Ok(beta.sample(&mut rand::thread_rng()))
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
            _ => Ordering::Greater,
        }
    })
}

/// Wrapper to order documents by `context_value`.
/// We need to implement `Ord` to use it in the `BinaryHeap`.
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
fn group_by_coi(documents: Vec<DocumentDataWithContext>) -> Result<DocumentsByCoi, Error> {
    documents
        .into_iter()
        .try_fold(DocumentsByCoi::new(), |mut groups, document| {
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

            Ok(groups)
        })
}

// Here we implement the algorithm described at page 9 of:
// http://www.ecmlpkdd2018.org/wp-content/uploads/2018/09/723.pdf
// We do not update all context_value like they do in the paper.

/// Update `alpha` and `beta` values based on the `context_value` of document in that coi
fn update_cois(
    cois: HashMap<CoiId, Coi>,
    documents: &[DocumentDataWithContext],
) -> Result<HashMap<CoiId, Coi>, Error> {
    documents.iter().try_fold(cois, |mut cois, document| {
        let coi = cois
            .get_mut(&document.coi.id)
            .ok_or(MabError::DocumentCoiDoesNotExist)?;

        let context_value = document.context.context_value;
        coi.alpha += context_value;
        coi.beta += 1. - context_value;

        Ok(cois)
    })
}

/// For each coi we take a sample from the beta distribution and we pick
/// the coi with the biggest sample. Then we take the document with the biggest `context_value` among
/// the documents within that coi.
fn pull_arms(
    beta_sampler: &BetaSampler,
    cois: &HashMap<CoiId, Coi>,
    mut documents_by_coi: DocumentsByCoi,
) -> Result<(DocumentsByCoi, DocumentDataWithContext), Error> {
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
            |max, coi_id| -> Result<_, Error> {
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
        Err(MabError::ExtractedCoiNoDocuments.into())
    }
}

struct MabRankingIter<'bs, 'cois> {
    beta_sampler: &'bs BetaSampler,
    cois: &'cois HashMap<CoiId, Coi>,
    documents_by_coi: DocumentsByCoi,
}

impl<'bs, 'cois> MabRankingIter<'bs, 'cois> {
    fn new(
        beta_sampler: &'bs BetaSampler,
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

impl<'bs, 'cois> Iterator for MabRankingIter<'bs, 'cois> {
    type Item = Result<DocumentDataWithContext, Error>;

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

pub struct MabRanking {
    beta_sampler: BetaSampler,
}

impl MabRanking {
    pub fn new(beta_sampler: BetaSampler) -> Self {
        Self { beta_sampler }
    }
}

impl MabSystem for MabRanking {
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

        let documents_by_coi = group_by_coi(documents)?;

        let mab_rerank = MabRankingIter::new(&self.beta_sampler, &cois, documents_by_coi);
        let documents = mab_rerank
            .enumerate()
            .map(|(rank, document)| {
                document.map(|document| {
                    DocumentDataWithMab::from_document(document, MabComponent { rank })
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;

        user_interests.positive = cois.into_iter().map(|(_, coi)| coi).collect();
        Ok((documents, user_interests))
    }
}
