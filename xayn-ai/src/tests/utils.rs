use std::ops::Range;

use ndarray::arr1;

use crate::{
    bert::Embedding,
    data::{
        document::{DocumentsRank, Relevance, UserFeedback},
        document_data::{
            CoiComponent,
            ContextComponent,
            DocumentContentComponent,
            DocumentDataWithDocument,
            DocumentDataWithEmbedding,
            DocumentDataWithMab,
            DocumentIdComponent,
            EmbeddingComponent,
            LtrComponent,
            MabComponent,
        },
        Coi,
        CoiId,
    },
    reranker::systems::{BertSystem, CoiSystemData},
    Document,
    DocumentHistory,
    DocumentId,
};

pub(crate) fn documents_from_ids(ids: Range<u32>) -> Vec<Document> {
    ids.enumerate()
        .map(|(rank, id)| Document {
            id: DocumentId(id.to_string()),
            rank,
            snippet: id.to_string(),
        })
        .collect()
}

pub(crate) fn documents_from_words(
    ctx: impl Iterator<Item = (usize, impl ToString)>,
) -> Vec<Document> {
    ctx.map(|(id, snippet)| Document {
        id: DocumentId(id.to_string()),
        rank: id,
        snippet: snippet.to_string(),
    })
    .collect()
}

pub(crate) fn cois_from_words(words: &[&str], bert: impl BertSystem) -> Vec<Coi> {
    documents_with_embeddings_from_words(words, bert)
        .enumerate()
        .map(|(id, doc)| Coi::new(id, doc.embedding.embedding))
        .collect()
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
        })
        .collect()
}

pub(crate) fn data_with_mab(
    ids_and_embeddings: impl Iterator<Item = (DocumentId, Embedding)>,
) -> Vec<DocumentDataWithMab> {
    ids_and_embeddings
        .map(|(id, embedding)| DocumentDataWithMab {
            document_id: DocumentIdComponent { id },
            embedding: EmbeddingComponent { embedding },
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

pub(crate) fn documents_with_embeddings_from_ids(
    ids: Range<u32>,
) -> Vec<DocumentDataWithEmbedding> {
    from_ids(ids)
        .map(|(id, embedding)| DocumentDataWithEmbedding {
            document_id: DocumentIdComponent { id },
            embedding: EmbeddingComponent { embedding },
        })
        .collect()
}

pub(crate) fn documents_with_embeddings_from_words(
    words: &[&str],
    bert: impl BertSystem,
) -> impl Iterator<Item = DocumentDataWithEmbedding> {
    let documents = words
        .iter()
        .enumerate()
        .map(|(id, snippet)| DocumentDataWithDocument {
            document_id: DocumentIdComponent {
                id: DocumentId(id.to_string()),
            },
            document_content: DocumentContentComponent {
                snippet: snippet.to_string(),
            },
        })
        .collect();

    bert.compute_embedding(documents).unwrap().into_iter()
}

pub(crate) fn expected_rerank_unchanged(docs: &[Document]) -> DocumentsRank {
    docs.iter()
        .enumerate()
        .map(|(rank, doc)| (doc.id.clone(), rank))
        .collect()
}

pub(crate) fn document_history(docs: Vec<(u32, Relevance, UserFeedback)>) -> Vec<DocumentHistory> {
    docs.into_iter()
        .map(|(id, relevance, user_feedback)| DocumentHistory {
            id: DocumentId(id.to_string()),
            relevance,
            user_feedback,
        })
        .collect()
}

pub(crate) fn from_ids(ids: Range<u32>) -> impl Iterator<Item = (DocumentId, Embedding)> {
    ids.map(|id| {
        (
            DocumentId(id.to_string()),
            arr1(&vec![id as f32; 128]).into(),
        )
    })
}
