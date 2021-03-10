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

use rand_distr::{Beta, Distribution};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum MabError {
    #[error("The coi id assigned to a document does not exist")]
    DocumentCoiDoesNotExist,
    #[error("No documents to pull")]
    NoDocumentsToPull,
    #[error("Extracted coi does not have documents")]
    ExtractedCoiNoDocuments,
}

/// Allow to sample a value from a beta distribution
pub trait BetaSample {
    fn sample(&self, alpha: f32, beta: f32) -> Result<f32, Error>;
}

pub struct BetaSampler;

impl BetaSample for BetaSampler {
    fn sample(&self, alpha: f32, beta: f32) -> Result<f32, Error> {
        let beta = Beta::new(alpha, beta)?;
        Ok(beta.sample(&mut rand::thread_rng()))
    }
}

trait MabReadyData {
    fn coi_id(&self) -> CoiId;
    fn context_value(&self) -> f32;
}

impl MabReadyData for DocumentDataWithContext {
    fn coi_id(&self) -> CoiId {
        self.coi.id
    }

    fn context_value(&self) -> f32 {
        self.context.context_value
    }
}

/// Pretend that comparing two f32 is total. The function will rank `NaN`
/// as the lowest value, similar to what [`f32::max`] does.
fn f32_total_cmp(a: &f32, b: &f32) -> Ordering {
    a.partial_cmp(&b).unwrap_or_else(|| {
        // if `partial_cmp` returns None we have at least one `NaN`,
        // we treat it as the lowest value
        if a.is_nan() {
            Ordering::Less
        } else {
            // other_value is NaN
            Ordering::Greater
        }
    })
}

/// Wrapper to order documents by `context_value`.
/// We need to implement Ord for it to use it in the `BinaryHeap`.
struct DocumentByContext<T>(T);

impl<T> PartialEq for DocumentByContext<T>
where
    T: MabReadyData,
{
    fn eq(&self, other: &Self) -> bool {
        self.0.context_value().eq(&other.0.context_value())
    }
}
impl<T> Eq for DocumentByContext<T> where T: MabReadyData {}

impl<T> PartialOrd for DocumentByContext<T>
where
    T: MabReadyData,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for DocumentByContext<T>
where
    T: MabReadyData,
{
    fn cmp(&self, other: &Self) -> Ordering {
        f32_total_cmp(&self.0.context_value(), &other.0.context_value())
    }
}

type DocumentsByCoi<T> = HashMap<CoiId, BinaryHeap<DocumentByContext<T>>>;

/// Group documents by coi and implicitly order them by context_value in the heap
fn groups_by_coi<T>(documents: Vec<T>) -> Result<DocumentsByCoi<T>, Error>
where
    T: MabReadyData,
{
    documents
        .into_iter()
        .try_fold(DocumentsByCoi::new(), |mut groups, document| {
            let coi_id = document.coi_id();

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
fn update_cois<T>(cois: HashMap<CoiId, Coi>, documents: &[T]) -> Result<HashMap<CoiId, Coi>, Error>
where
    T: MabReadyData,
{
    documents.iter().try_fold(cois, |mut cois, document| {
        let coi = cois
            .get_mut(&document.coi_id())
            .ok_or(MabError::DocumentCoiDoesNotExist)?;

        let context_value = document.context_value();
        coi.alpha += context_value;
        coi.beta += 1. - context_value;

        Ok(cois)
    })
}

/// For each coi with documents we take a sample from the beta distribution and we pick
/// the coi with the biggest sample. Then we take the document with the biggest `context_value` among
/// the documents within that coi.
fn pull_arms<T>(
    beta_sampler: &impl BetaSample,
    cois: &HashMap<CoiId, Coi>,
    mut documents_by_coi: DocumentsByCoi<T>,
) -> Result<(DocumentsByCoi<T>, T), Error>
where
    T: MabReadyData,
{
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

struct MabRankingIter<'bs, 'cois, BS, T> {
    beta_sampler: &'bs BS,
    cois: &'cois HashMap<CoiId, Coi>,
    documents_by_coi: DocumentsByCoi<T>,
}

impl<'bs, 'cois, BS, T> MabRankingIter<'bs, 'cois, BS, T>
where
    BS: BetaSample,
    T: MabReadyData,
{
    fn new(
        beta_sampler: &'bs BS,
        cois: &'cois HashMap<CoiId, Coi>,
        documents_by_coi: DocumentsByCoi<T>,
    ) -> Self {
        Self {
            beta_sampler,
            cois,
            documents_by_coi,
        }
    }
}

impl<'bs, 'cois, BS, T> Iterator for MabRankingIter<'bs, 'cois, BS, T>
where
    BS: BetaSample,
    T: MabReadyData,
{
    type Item = Result<T, Error>;

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

impl<BS> MabRanking<BS>
where
    BS: BetaSample,
{
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

        let documents_by_coi = groups_by_coi(documents)?;

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
