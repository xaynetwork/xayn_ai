use std::ops::Range;

use ndarray::arr1;

use crate::{
    data::{
        document::{Relevance, UserFeedback},
        document_data::{
            CoiComponent,
            ContextComponent,
            DocumentBaseComponent,
            DocumentContentComponent,
            DocumentDataWithDocument,
            DocumentDataWithMab,
            DocumentDataWithSMBert,
            LtrComponent,
            MabComponent,
            QAMBertComponent,
            SMBertComponent,
        },
        CoiId,
        CoiPoint,
        NegativeCoi,
        PositiveCoi,
    },
    embedding::utils::Embedding,
    reranker::systems::{CoiSystemData, SMBertSystem},
    Document,
    DocumentHistory,
    DocumentId,
};

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

fn cois_from_words<CP: CoiPoint>(titles: &[&str], smbert: impl SMBertSystem) -> Vec<CP> {
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
        .collect();

    smbert
        .compute_embedding(documents)
        .unwrap()
        .into_iter()
        .enumerate()
        .map(|(id, doc)| CP::new(id, doc.smbert.embedding))
        .collect()
}

pub(crate) fn pos_cois_from_words(titles: &[&str], smbert: impl SMBertSystem) -> Vec<PositiveCoi> {
    cois_from_words(titles, smbert)
}

pub(crate) fn neg_cois_from_words(titles: &[&str], smbert: impl SMBertSystem) -> Vec<NegativeCoi> {
    cois_from_words(titles, smbert)
}

pub(crate) fn history_for_prev_docs(
    prev_documents: &[&dyn CoiSystemData],
    relevance: Vec<(Relevance, UserFeedback)>,
) -> Vec<DocumentHistory> {
    prev_documents
        .iter()
        .zip(relevance)
        .map(|(doc, (relevance, user_feedback))| DocumentHistory {
            id: doc.id().clone(),
            relevance,
            user_feedback,
            ..Default::default()
        })
        .collect()
}

pub(crate) fn data_with_mab(
    ids_and_embeddings: impl Iterator<Item = (DocumentId, usize, Embedding)>,
) -> Vec<DocumentDataWithMab> {
    ids_and_embeddings
        .map(|(id, initial_ranking, embedding)| DocumentDataWithMab {
            document_base: DocumentBaseComponent {
                id,
                initial_ranking,
            },
            smbert: SMBertComponent { embedding },
            qambert: QAMBertComponent { similarity: 0.5 },
            coi: CoiComponent {
                id: CoiId(1),
                pos_distance: 0.1,
                neg_distance: 0.1,
            },
            ltr: LtrComponent { ltr_score: 0.5 },
            context: ContextComponent { context_value: 0.5 },
            mab: MabComponent { rank: 0 },
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
) -> Vec<DocumentDataWithSMBert> {
    from_ids(0..snippets.len() as u32)
        .map(|(id, initial_ranking, embedding)| DocumentDataWithSMBert {
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
