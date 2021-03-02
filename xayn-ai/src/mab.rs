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

use anyhow::anyhow;

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
struct DocumentByContext(DocumentDataWithContext);

impl PartialEq for DocumentByContext {
    fn eq(&self, other: &Self) -> bool {
        self.0
            .context
            .context_value
            .eq(&other.0.context.context_value)
    }
}
impl Eq for DocumentByContext {}

impl PartialOrd for DocumentByContext {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0
            .context
            .context_value
            .partial_cmp(&other.0.context.context_value)
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

/// Group documents by coi and order them by context_value
fn groups_by_coi(documents: Vec<DocumentDataWithContext>) -> Result<DocumentsByCoi, Error> {
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

fn update_cois(
    cois: HashMap<CoiId, Coi>,
    documents: &[DocumentDataWithContext],
) -> Result<HashMap<CoiId, Coi>, Error> {
    documents.iter().try_fold(cois, |mut cois, document| {
        let coi = cois
            .get_mut(&document.coi.id)
            .ok_or_else(|| anyhow!("The coi id assigned to a document does not exists"))?;

        let context_value = document.context.context_value;
        coi.alpha += context_value;
        coi.beta += 1. - context_value;

        Ok(cois)
    })
}

fn pull_arms(
    beta_sampler: &impl BetaSample,
    cois: &HashMap<CoiId, Coi>,
    mut documents_by_coi: DocumentsByCoi,
) -> Result<(DocumentsByCoi, DocumentDataWithContext), Error> {
    let coi_id = *documents_by_coi
        .keys()
        .map(|coi_id| {
            let coi = cois
                .get(coi_id)
                .ok_or_else(|| anyhow!("The coi id assigned to a document does not exists"))?;

            beta_sampler
                .sample(coi.alpha, coi.beta)
                .map(|sample| (sample, coi_id))
        })
        .collect::<Result<Vec<_>, _>>()?
        .iter()
        .max_by(|(a, _), (b, _)| f32_total_cmp(a, b))
        .ok_or_else(|| anyhow!("Cannot get coi when pulling arms"))?
        .1;

    if let Entry::Occupied(mut entry) = documents_by_coi.entry(coi_id) {
        let heap = entry.get_mut();
        let document = heap
            .pop()
            .ok_or_else(|| anyhow!("Extracted coi does not have documents"))?;
        // remove coi when they have no documents left
        if heap.is_empty() {
            entry.remove_entry();
        }

        Ok((documents_by_coi, document.0))
    } else {
        Err(anyhow!("Extracted coi does not have documents"))
    }
}

fn mab_ranking(
    beta_sampler: &impl BetaSample,
    cois: &HashMap<CoiId, Coi>,
    mut documents_by_coi: DocumentsByCoi,
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
