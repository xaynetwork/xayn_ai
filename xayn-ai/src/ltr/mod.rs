mod features;
mod list_net;

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

const BINPARAMS_PATH: &str = "../data/ltr_v0000/ltr.binparams";

/// Domain reranker consisting of a ListNet model trained on engineered features.
pub(crate) struct DomainReranker;

impl LtrSystem for DomainReranker {
    fn compute_ltr(
        &self,
        history: &[DocumentHistory],
        documents: Vec<DocumentDataWithCoi>,
    ) -> Result<Vec<DocumentDataWithLtr>, Error> {
        let hists = history.iter().map_into().collect_vec();
        let docs = documents.iter().map_into().collect_vec();

        let feats = build_features(hists, docs)?;
        let feats_arr = features_to_ndarray(&feats);
        let model = ListNet::load_from_file(BINPARAMS_PATH)?;
        let ltr_scores = model.run(feats_arr);

        Ok(izip!(documents, ltr_scores)
            .map(|(document, ltr_score)| {
                DocumentDataWithLtr::from_document(document, LtrComponent { ltr_score })
            })
            .collect())
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
    use float_cmp::approx_eq;
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
        assert!(approx_eq!(f32, ltr_docs[0].ltr.ltr_score, 0.5));
        assert!(approx_eq!(f32, ltr_docs[1].ltr.ltr_score, 0.5));
    }
}
