use std::ops::Range;

use ndarray::arr1;

use crate::{
    bert::Embedding,
    data::{
        document::{Ranks, Relevance, UserFeedback},
        document_data::{
            CoiComponent,
            ContextComponent,
            DocumentBaseComponent,
            DocumentContentComponent,
            DocumentDataWithDocument,
            DocumentDataWithSMBert,
            DocumentDataWithMab,
            SMBertEmbeddingComponent,
            LtrComponent,
            MabComponent,
        },
        CoiId,
        CoiPoint,
        NegativeCoi,
        PositiveCoi,
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

fn cois_from_words<CP: CoiPoint>(snippets: &[&str], bert: impl BertSystem) -> Vec<CP> {
    let documents = snippets
        .iter()
        .enumerate()
        .map(|(id, snippet)| DocumentDataWithDocument {
            document_base: DocumentBaseComponent {
                id: DocumentId(id.to_string()),
                initial_ranking: id,
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
        .map(|(id, doc)| CP::new(id, doc.embedding.embedding))
        .collect()
}

pub(crate) fn pos_cois_from_words(snippets: &[&str], bert: impl BertSystem) -> Vec<PositiveCoi> {
    cois_from_words(snippets, bert)
}

pub(crate) fn neg_cois_from_words(snippets: &[&str], bert: impl BertSystem) -> Vec<NegativeCoi> {
    cois_from_words(snippets, bert)
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
    ids_and_embeddings: impl Iterator<Item = (DocumentId, usize, Embedding)>,
) -> Vec<DocumentDataWithMab> {
    ids_and_embeddings
        .map(|(id, initial_ranking, embedding)| DocumentDataWithMab {
            document_base: DocumentBaseComponent {
                id,
                initial_ranking,
            },
            embedding: SMBertEmbeddingComponent { embedding },
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
) -> Vec<DocumentDataWithSMBert> {
    from_ids(ids)
        .map(
            |(id, initial_ranking, embedding)| DocumentDataWithSMBert {
                document_base: DocumentBaseComponent {
                    id,
                    initial_ranking,
                },
                embedding: SMBertEmbeddingComponent { embedding },
            },
        )
        .collect()
}

// Not used at the moment, but could be useful in the short future
// pub(crate) fn documents_with_embeddings_from_words(
//     words: &[&str],
//     bert: impl BertSystem,
// ) -> impl Iterator<Item = DocumentDataWithEmbedding> {
//     let documents = words
//         .iter()
//         .enumerate()
//         .map(|(id, snippet)| DocumentDataWithDocument {
//             document_id: DocumentIdComponent {
//                 id: DocumentId(id.to_string()),
//             },
//             document_content: DocumentContentComponent {
//                 snippet: snippet.to_string(),
//             },
//         })
//         .collect();

//     bert.compute_embedding(documents).unwrap().into_iter()
// }

pub(crate) fn expected_rerank_unchanged(docs: &[Document]) -> Ranks {
    docs.iter().map(|doc| doc.rank).collect()
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

/// Return a sequence of `(document_id, initial_ranking, embedding)` tuples.
///
/// The passed in integer ids are converted to a string and used as document_id's as
/// well as used as the initial_ranking.
pub(crate) fn from_ids(ids: Range<u32>) -> impl Iterator<Item = (DocumentId, usize, Embedding)> {
    ids.map(|id| {
        (
            DocumentId(id.to_string()),
            id as usize,
            arr1(&vec![id as f32; 128]).into(),
        )
    })
}
