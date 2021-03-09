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
    #[error("The coi id assigned to a document does not exists")]
    DocumentCoiDoesNotExists,
    #[error("No documents to pull")]
    NoDocumentsToPull,
    #[error("Extracted coi does not have documents")]
    ExtractedCoiNoDocuments,
}

/// Allow to sample a value form a beta distrubtion
pub trait BetaSample {
    fn sample(&self, alpha: f32, beta: f32) -> Result<f32, Error>;
}

pub struct BetaSampler {}

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

/// Pretend that comparing two f32 is total. The function will rank `Nan`
/// as the lowest value, similar to what `f32::max` does.
fn f32_total_cmp(a: &f32, b: &f32) -> Ordering {
    a.partial_cmp(&b).unwrap_or_else(|| {
        // if `partial_cmp` returns None we have at least one `NaN`
        // we treat `NaN` as the lowest value
        if a.is_nan() {
            Ordering::Less
        } else {
            // other_value is NaN
            Ordering::Greater
        }
    })
}

/// Wrapper to order documents by `context_value`
struct DocumentByContext<T: MabReadyData>(T);

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
        self.0.context_value().partial_cmp(&other.0.context_value())
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

/// Group documents by coi and order them by context_value
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
// We do not update all y like they do in the paper.

fn update_cois<T>(cois: HashMap<CoiId, Coi>, documents: &[T]) -> Result<HashMap<CoiId, Coi>, Error>
where
    T: MabReadyData,
{
    documents.iter().try_fold(cois, |mut cois, document| {
        let coi = cois
            .get_mut(&document.coi_id())
            .ok_or(MabError::DocumentCoiDoesNotExists)?;

        let context_value = document.context_value();
        coi.alpha += context_value;
        coi.beta += 1. - context_value;

        Ok(cois)
    })
}

fn pull_arms<T>(
    beta_sampler: &impl BetaSample,
    cois: &HashMap<CoiId, Coi>,
    mut documents_by_coi: DocumentsByCoi<T>,
) -> Result<(DocumentsByCoi<T>, T), Error>
where
    T: MabReadyData,
{
    let coi_id = *documents_by_coi
        .keys()
        // sampling beta distribution for each coi
        .map(|coi_id| {
            let coi = cois.get(coi_id).ok_or(MabError::DocumentCoiDoesNotExists)?;

            beta_sampler
                .sample(coi.alpha, coi.beta)
                .map(|sample| (sample, coi_id))
        })
        .collect::<Result<Vec<_>, _>>()?
        .iter()
        // get the coi whose sample is biggest
        .max_by(|(a, _), (b, _)| f32_total_cmp(a, b))
        .ok_or(MabError::NoDocumentsToPull)?
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

fn mab_ranking(
    beta_sampler: &impl BetaSample,
    cois: &HashMap<CoiId, Coi>,
    mut documents_by_coi: DocumentsByCoi<DocumentDataWithContext>,
    // max documents to extract
    max_documents: usize,
) -> Result<Vec<DocumentDataWithMab>, Error> {
    let mut with_mab = Vec::with_capacity(max_documents);
    let mut rank = 0;

    while !documents_by_coi.is_empty() && rank < max_documents {
        let (new_documents_by_coi, document) = pull_arms(beta_sampler, cois, documents_by_coi)?;
        documents_by_coi = new_documents_by_coi;

        with_mab.push(DocumentDataWithMab::from_document(
            document,
            MabComponent { rank },
        ));

        rank += 1;
    }

    Ok(with_mab)
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
            .collect::<HashMap<_, _>>();
        let cois = update_cois(cois, &documents)?;

        let documents_len = documents.len();
        let documents_by_coi = groups_by_coi(documents)?;

        let documents = mab_ranking(&self.beta_sampler, &cois, documents_by_coi, documents_len)?;

        user_interests.positive = cois.into_iter().map(|(_, coi)| coi).collect();
        Ok((documents, user_interests))
    }
}
