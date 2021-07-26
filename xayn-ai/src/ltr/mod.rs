mod features;

#[doc(hidden)]
pub mod list_net;

use std::{
    fs::File,
    io::{BufReader, Read},
    path::Path,
};

use itertools::{izip, Itertools};
use ndarray::{Array1, Array2};

use crate::{
    data::{
        document::DocumentHistory,
        document_data::{DocumentDataWithCoi, DocumentDataWithLtr, LtrComponent},
    },
    error::Error,
    reranker::systems::LtrSystem,
};

use features::{build_features, features_to_ndarray};
use list_net::ListNet;

use self::features::{DocSearchResult, Features, HistSearchResult};

/// Domain reranker consisting of a ListNet model trained on engineered features.
pub(crate) struct DomainReranker {
    model: ListNet,
}

impl LtrSystem for DomainReranker {
    fn compute_ltr(
        &self,
        history: &[DocumentHistory],
        documents: Vec<DocumentDataWithCoi>,
    ) -> Result<Vec<DocumentDataWithLtr>, Error> {
        let hists = history.iter().map_into().collect_vec();
        let docs = documents.iter().map_into().collect_vec();

        let features = build_features(hists, docs)?;
        let ltr_scores = self.predict(&features);

        Ok(izip!(documents, ltr_scores)
            .map(|(document, ltr_score)| {
                DocumentDataWithLtr::from_document(document, LtrComponent { ltr_score })
            })
            .collect())
    }
}

impl DomainReranker {
    /// Predicts LTR scores element-wise over the given sequence of `Features`.
    fn predict(&self, features: &[Features]) -> Vec<f32> {
        let feats_arr = features_to_ndarray(features);
        self.model.run(feats_arr)
    }
}

/// Builder for [`DomainReranker`].
pub(crate) struct DomainRerankerBuilder<M> {
    model_params: M,
}

impl DomainRerankerBuilder<BufReader<File>> {
    /// Creates a [`LtrBuilder`] from a model params file.
    pub(crate) fn from_file(model_params: impl AsRef<Path>) -> Result<Self, Error> {
        let model_params = BufReader::new(File::open(model_params)?);
        Ok(Self::new(model_params))
    }
}

impl<M> DomainRerankerBuilder<M> {
    /// Creates a [`DomainRerankerBuilder`] from in-memory model params.
    pub(crate) fn new(model_params: M) -> Self {
        Self { model_params }
    }

    /// Builds a [`DomainReranker`].
    pub(crate) fn build(self) -> Result<DomainReranker, Error>
    where
        M: Read,
    {
        let model = ListNet::deserialize_from(self.model_params)?;
        Ok(DomainReranker { model })
    }
}

/// LTR with constant value.
pub(crate) struct ConstLtr;

impl ConstLtr {
    const SCORE: f32 = 0.5;

    #[allow(unused)] // TODO move ConstLtr into tests / remove later
    pub(crate) fn new() -> Self {
        // 0.5 is the only valid value.
        // It must be between 0 and 1. Since this is used to compute the context value
        // and context value is used to update `alpha` and `beta` of the cois.
        // Using a value different from 0.5 will change the parameters of a coi in an
        // unbalanced way.
        Self
    }
}

impl LtrSystem for ConstLtr {
    fn compute_ltr(
        &self,
        _history: &[DocumentHistory],
        documents: Vec<DocumentDataWithCoi>,
    ) -> Result<Vec<DocumentDataWithLtr>, Error> {
        let ltr_score = Self::SCORE;
        Ok(documents
            .into_iter()
            .map(|doc| DocumentDataWithLtr::from_document(doc, LtrComponent { ltr_score }))
            .collect())
    }
}

pub type OwnedSample = (Array2<f32>, Array1<f32>);

/// Creates training data for ListNet from a users history.
pub fn list_net_training_data_from_history(
    mut history: &[DocumentHistory],
) -> Result<Vec<OwnedSample>, Error> {
    // FIXME[follow up PR] We have a few ways to do this:
    // 1: Create a single sample based on the last query in the history as it's done in soundgarden.
    // 2: Create samples for the last n unique queries in history (or based on percentage of queries).
    // 3: Go through the history and create a query for each query in history but discount sample relevance (weights)
    //    for older queries. (Discount == given the gradient based on the sample a smaller weight when averaging the gradients)
    // ...
    // For now we will go with 1. but this is like not the best choice.
    loop {
        // History has no relevant query and as such no samples.
        if history.is_empty() {
            return Ok(Vec::new());
        }

        let last_query_results_id = {
            let last = history.last().unwrap();
            (last.session, last.query_count)
        };
        let start_of_last_query = history
            .iter()
            .rposition(|doc| (doc.session, doc.query_count) != last_query_results_id)
            .map(|last_before_last_query_idx| last_before_last_query_idx + 1)
            .unwrap_or_default();

        let mut sorted_last_query = history[start_of_last_query..].iter().collect_vec();
        sorted_last_query.sort_by_key(|doc| doc.rank);

        let relevances = sorted_last_query
            .iter()
            .map(|doc| doc.relevance)
            .collect_vec();
        let relevances =
            if let Some(relevances) = self::list_net::prepare_target_prob_dist(&relevances) {
                relevances
            } else {
                // The last query is irrelevant so ignore pretend it doesn't exist.
                history = &history[..start_of_last_query];
                continue;
            };

        let pseudo_current =
            create_pseudo_current_query_from_historic_query(sorted_last_query.iter().copied());
        let history = history[..start_of_last_query]
            .iter()
            .map_into()
            .collect_vec();
        let features = build_features(history, pseudo_current)?;
        let features = features_to_ndarray(&features);
        return Ok(vec![(features, relevances)]);
    }
}

fn create_pseudo_current_query_from_historic_query<'a>(
    historic_query: impl IntoIterator<Item = &'a DocumentHistory>,
) -> Vec<DocSearchResult> {
    historic_query
        .into_iter()
        .map(|doc| {
            let HistSearchResult {
                query,
                url,
                domain,
                final_rank,
                ..
            } = HistSearchResult::from(doc);
            DocSearchResult {
                query,
                url,
                domain,
                initial_rank: final_rank,
            }
        })
        .collect_vec()
}

#[cfg(test)]
mod tests {
    use ndarray::arr1;

    use super::*;
    use crate::{
        data::{
            document::DocumentId,
            document_data::{
                CoiComponent,
                DocumentBaseComponent,
                DocumentContentComponent,
                QAMBertComponent,
                SMBertComponent,
            },
        },
        utils::mock_coi_id,
    };

    #[test]
    fn test_const_value() {
        let id = DocumentId::from_u128(0);
        let embedding = arr1(&[1., 2., 3., 4.]).into();
        let coi = CoiComponent {
            id: mock_coi_id(9),
            pos_distance: 0.7,
            neg_distance: 0.2,
        };
        let doc1 = DocumentDataWithCoi {
            document_base: DocumentBaseComponent {
                id,
                initial_ranking: 24,
            },
            document_content: DocumentContentComponent {
                ..Default::default()
            },
            smbert: SMBertComponent { embedding },
            qambert: QAMBertComponent { similarity: 0.5 },
            coi,
        };

        let id = DocumentId::from_u128(1);
        let embedding = arr1(&[5., 6., 7.]).into();
        let coi = CoiComponent {
            id: mock_coi_id(5),
            pos_distance: 0.3,
            neg_distance: 0.9,
        };
        let doc2 = DocumentDataWithCoi {
            document_base: DocumentBaseComponent {
                id,
                initial_ranking: 42,
            },
            document_content: DocumentContentComponent {
                ..Default::default()
            },
            smbert: SMBertComponent { embedding },
            qambert: QAMBertComponent { similarity: 0.5 },
            coi,
        };

        let res = ConstLtr::new().compute_ltr(&[], vec![doc1, doc2]);
        assert!(res.is_ok());
        let ltr_docs = res.unwrap();
        assert_eq!(ltr_docs.len(), 2);
        assert_approx_eq!(f32, ltr_docs[0].ltr.ltr_score, 0.5, ulps = 0);
        assert_approx_eq!(f32, ltr_docs[1].ltr.ltr_score, 0.5, ulps = 0);
    }
}
