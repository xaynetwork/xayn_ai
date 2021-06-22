mod features;
mod list_net;

use std::{
    fs::File,
    io::{BufReader, Read},
    path::Path,
};

use itertools::{izip, Itertools};

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

use self::features::Features;

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
        let model = ListNet::load_from_source(self.model_params)?;
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

#[cfg(test)]
mod tests {
    use ndarray::arr1;

    use super::*;
    use crate::data::{
        document::DocumentId,
        document_data::{
            CoiComponent,
            DocumentBaseComponent,
            DocumentContentComponent,
            QAMBertComponent,
            SMBertComponent,
        },
        CoiId,
    };

    #[test]
    fn test_const_value() {
        let id = DocumentId::from_u128(0);
        let embedding = arr1(&[1., 2., 3., 4.]).into();
        let coi = CoiComponent {
            id: CoiId(9),
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
            id: CoiId(5),
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
