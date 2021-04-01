use std::ops::Range;

use rubert::ndarray::arr1;

use crate::{
    data::{
        document::{Relevance, UserFeedback},
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
    reranker::DocumentsRank,
    reranker_systems::{BertSystem, CoiSystemData},
    Document,
    DocumentHistory,
    DocumentId,
};

pub fn documents_from_ids(ids: Range<u32>) -> Vec<Document> {
    ids.enumerate()
        .map(|(rank, id)| Document {
            id: DocumentId(id.to_string()),
            rank,
            snippet: id.to_string(),
        })
        .collect()
}

pub fn documents_from_words(ctx: impl Iterator<Item = (usize, impl ToString)>) -> Vec<Document> {
    ctx.map(|(id, snippet)| Document {
        id: DocumentId(id.to_string()),
        rank: id,
        snippet: snippet.to_string(),
    })
    .collect()
}

pub fn cois_from_words(snippets: &[&str], bert: impl BertSystem) -> Vec<Coi> {
    let documents = snippets
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

    bert.compute_embedding(documents)
        .unwrap()
        .into_iter()
        .enumerate()
        .map(|(id, doc)| Coi::new(id, doc.embedding.embedding))
        .collect()
}

pub fn history_for_prev_docs(
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

pub fn data_with_mab(
    ids_and_embeddings: impl Iterator<Item = (u32, Vec<f32>)>,
) -> Vec<DocumentDataWithMab> {
    ids_and_embeddings
        .map(|(id, embedding)| DocumentDataWithMab {
            document_id: DocumentIdComponent {
                id: DocumentId(id.to_string()),
            },
            embedding: EmbeddingComponent {
                embedding: arr1(&embedding).into(),
            },
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

pub fn data_with_embedding(
    ids_and_embeddings: impl Iterator<Item = (u32, Vec<f32>)>,
) -> Vec<DocumentDataWithEmbedding> {
    ids_and_embeddings
        .map(|(id, embedding)| DocumentDataWithEmbedding {
            document_id: DocumentIdComponent {
                id: DocumentId(id.to_string()),
            },
            embedding: EmbeddingComponent {
                embedding: arr1(&embedding).into(),
            },
        })
        .collect()
}

pub fn expected_rerank_unchanged(docs: &[Document]) -> DocumentsRank {
    docs.iter()
        .enumerate()
        .map(|(rank, doc)| (doc.id.clone(), rank))
        .collect()
}

pub fn document_history(docs: Vec<(u32, Relevance, UserFeedback)>) -> Vec<DocumentHistory> {
    docs.into_iter()
        .map(|(id, relevance, user_feedback)| DocumentHistory {
            id: DocumentId(id.to_string()),
            relevance,
            user_feedback,
        })
        .collect()
}
