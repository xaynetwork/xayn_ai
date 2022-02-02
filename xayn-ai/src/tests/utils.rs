use std::ops::Range;

use ndarray::arr1;
use uuid::Uuid;

use crate::{
    coi::{
        point::{
            tests::CoiPointConstructor,
            NegativeCoi,
            PositiveCoi,
            PositiveCoi_v0_0_0,
            PositiveCoi_v0_1_0,
            PositiveCoi_v0_2_0,
        },
        CoiId,
    },
    data::{
        document::{Relevance, UserFeedback},
        document_data::{
            CoiComponent,
            ContextComponent,
            DocumentBaseComponent,
            DocumentContentComponent,
            DocumentDataWithCoi,
            DocumentDataWithDocument,
            DocumentDataWithRank,
            DocumentDataWithSMBert,
            LtrComponent,
            QAMBertComponent,
            RankComponent,
            SMBertComponent,
        },
    },
    embedding::utils::Embedding,
    reranker::systems::{CoiSystemData, SMBertSystem},
    Document,
    DocumentHistory,
    DocumentId,
};

/// Creates an UUID by combining `fcb6a685-eb92-4d36-8686-XXXXXXXXXXXX` with the given `sub_id`.
pub(crate) const fn mock_uuid(sub_id: usize) -> Uuid {
    const BASE_UUID: u128 = 0xfcb6a685eb924d368686000000000000;
    Uuid::from_u128(BASE_UUID | (sub_id as u128))
}

#[test]
fn test_mock_uuid() {
    assert_eq!(
        format!("{}", mock_uuid(0xABCDEF0A)),
        "fcb6a685-eb92-4d36-8686-0000abcdef0a",
    );
}

pub(crate) fn documents_from_ids(ids: Range<u128>) -> Vec<Document> {
    ids.enumerate()
        .map(|(rank, id)| Document {
            id: DocumentId::from_u128(id),
            rank,
            title: id.to_string(),
            ..Default::default()
        })
        .collect()
}

pub(crate) fn documents_from_words(
    ctx: impl Iterator<Item = (usize, impl ToString)>,
) -> Vec<Document> {
    ctx.map(|(id, title)| Document {
        id: DocumentId::from_u128(id as u128),
        rank: id,
        title: title.to_string(),
        ..Default::default()
    })
    .collect()
}

fn cois_from_words<CP: CoiPointConstructor>(
    titles: &[&str],
    smbert: impl SMBertSystem,
    start_id: usize,
) -> Vec<CP> {
    let documents = titles
        .iter()
        .enumerate()
        .map(|(id, title)| DocumentDataWithDocument {
            document_base: DocumentBaseComponent {
                id: DocumentId::from_u128(id as u128),
                initial_ranking: id,
            },
            document_content: DocumentContentComponent {
                title: title.to_string(),
                snippet: format!("snippet of {}", title),
                query_words: "query".to_string(),
                ..Default::default()
            },
        })
        .collect::<Vec<_>>();

    smbert
        .compute_embedding(&documents)
        .unwrap()
        .into_iter()
        .enumerate()
        .map(|(offset, doc)| CP::new(CoiId::mocked(start_id + offset), doc.smbert.embedding))
        .collect()
}

pub(crate) fn pos_cois_from_words_v0(
    titles: &[&str],
    smbert: impl SMBertSystem,
) -> Vec<PositiveCoi_v0_0_0> {
    cois_from_words(titles, smbert, 0)
}

pub(crate) fn pos_cois_from_words_v1(
    titles: &[&str],
    smbert: impl SMBertSystem,
) -> Vec<PositiveCoi_v0_1_0> {
    cois_from_words(titles, smbert, 0)
}

pub(crate) fn pos_cois_from_words_v2(
    titles: &[&str],
    smbert: impl SMBertSystem,
) -> Vec<PositiveCoi_v0_2_0> {
    cois_from_words(titles, smbert, 0)
}

pub(crate) fn pos_cois_from_words(titles: &[&str], smbert: impl SMBertSystem) -> Vec<PositiveCoi> {
    cois_from_words(titles, smbert, 0)
}

pub(crate) fn neg_cois_from_words(titles: &[&str], smbert: impl SMBertSystem) -> Vec<NegativeCoi> {
    cois_from_words(titles, smbert, 0)
}

pub(crate) fn pos_cois_from_words_with_ids(
    titles: &[&str],
    smbert: impl SMBertSystem,
    start_id: usize,
) -> Vec<PositiveCoi> {
    cois_from_words(titles, smbert, start_id)
}

pub(crate) fn neg_cois_from_words_with_ids(
    titles: &[&str],
    smbert: impl SMBertSystem,
    start_id: usize,
) -> Vec<NegativeCoi> {
    cois_from_words(titles, smbert, start_id)
}

pub(crate) fn history_for_prev_docs(
    prev_documents: &[&dyn CoiSystemData],
    relevance: Vec<(Relevance, UserFeedback)>,
) -> Vec<DocumentHistory> {
    prev_documents
        .iter()
        .zip(relevance)
        .map(|(doc, (relevance, user_feedback))| DocumentHistory {
            id: doc.id(),
            relevance,
            user_feedback,
            ..Default::default()
        })
        .collect()
}

pub(crate) fn data_with_rank(
    ids_and_embeddings: impl Iterator<Item = (DocumentId, usize, Embedding)>,
) -> Vec<DocumentDataWithRank> {
    ids_and_embeddings
        .map(|(id, initial_ranking, embedding)| DocumentDataWithRank {
            document_base: DocumentBaseComponent {
                id,
                initial_ranking,
            },
            document_content: DocumentContentComponent {
                title: id.to_string(),
                ..DocumentContentComponent::default()
            },
            smbert: SMBertComponent { embedding },
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

pub(crate) fn documents_with_embeddings_from_ids(ids: Range<u32>) -> Vec<DocumentDataWithSMBert> {
    from_ids(ids)
        .map(|(id, initial_ranking, embedding)| DocumentDataWithSMBert {
            document_base: DocumentBaseComponent {
                id,
                initial_ranking,
            },
            document_content: DocumentContentComponent {
                title: "title".to_string(),
                snippet: "snippet".to_string(),
                query_words: "query".to_string(),
                ..Default::default()
            },
            smbert: SMBertComponent { embedding },
        })
        .collect()
}

pub(crate) fn documents_with_embeddings_from_snippet_and_query(
    query: &str,
    snippets: &[&str],
) -> Vec<DocumentDataWithCoi> {
    from_ids(0..snippets.len() as u32)
        .map(|(id, initial_ranking, embedding)| DocumentDataWithCoi {
            document_base: DocumentBaseComponent {
                id,
                initial_ranking,
            },
            document_content: DocumentContentComponent {
                title: format!("title for {}", snippets[initial_ranking]),
                snippet: snippets[initial_ranking].to_string(),
                query_words: query.to_string(),
                ..Default::default()
            },
            smbert: SMBertComponent { embedding },
            coi: CoiComponent {
                id: CoiId::mocked(1),
                pos_similarity: 0.,
                neg_similarity: 0.,
            },
        })
        .collect()
}

pub(crate) fn expected_rerank_unchanged(docs: &[Document]) -> Vec<u16> {
    docs.iter().map(|doc| doc.rank as u16).collect()
}

pub(crate) fn document_history(docs: Vec<(u32, Relevance, UserFeedback)>) -> Vec<DocumentHistory> {
    docs.into_iter()
        .map(|(id, relevance, user_feedback)| DocumentHistory {
            id: DocumentId::from_u128(id as u128),
            relevance,
            user_feedback,
            ..Default::default()
        })
        .collect()
}

/// Return a sequence of `(document_id, initial_ranking, embedding)` tuples.
///
/// The passed in integer ids are converted to a string and used as document_id's as
/// well as used as the initial_ranking.
pub(crate) fn from_ids(ids: Range<u32>) -> impl Iterator<Item = (DocumentId, usize, Embedding)> {
    ids.map(|id| {
        (
            DocumentId::from_u128(id as u128),
            id as usize,
            arr1(&vec![id as f32; 128]).into(),
        )
    })
}
