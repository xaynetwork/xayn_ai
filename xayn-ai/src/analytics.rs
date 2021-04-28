use std::{cmp::Ordering, collections::HashMap};

use anyhow::bail;
use displaydoc::Display;
use serde::Serialize;
use thiserror::Error;

use crate::{
    data::{
        document::{DocumentHistory, Relevance},
        document_data::DocumentDataWithMab,
    },
    error::Error,
    reranker::systems,
    utils::nan_safe_f32_cmp_desc,
};

/// Which k to use for nDCG@k
const DEFAULT_NDCG_K: usize = 2;
/// Calculated analytics data.
#[derive(Clone, Serialize)]
pub struct Analytics {
    /// The nDCG@k score between the LTR ranking and the relevance based ranking
    pub ndcg_ltr: f32,
    /// The nDCG@k score between the Context ranking and the relevance based ranking
    pub ndcg_context: f32,
    /// The nDCG@k score between the initial ranking and the relevance based ranking
    pub ndcg_initial_ranking: f32,
    /// The nDCG@k score between the final ranking and the relevance based ranking
    pub ndcg_final_ranking: f32,
}

#[derive(Error, Debug, Display)]
/// Can not calculate Analytics as no relevant history is available.
pub(crate) struct NoRelevantHistoricInfo;

pub(crate) struct AnalyticsSystem;

impl systems::AnalyticsSystem for AnalyticsSystem {
    fn compute_analytics(
        &self,
        history: &[DocumentHistory],
        documents: &[DocumentDataWithMab],
    ) -> Result<Analytics, Error> {
        // We need to be able to lookup relevances by document id.
        // and linear search is most likely a bad idea. So we create
        // a hashmap for the lookups.
        let relevance_lookups: HashMap<_, _> = history
            .iter()
            .map(|h_doc| (&h_doc.id, score_for_relevance(h_doc.relevance)))
            .collect();

        let mut paired_ltr_scores = Vec::new();
        let mut paired_context_scores = Vec::new();
        let mut paired_final_ranking_scores = Vec::new();
        let mut paired_initial_ranking_scores = Vec::new();

        for document in documents {
            if let Some(relevance) = relevance_lookups.get(&document.document_base.id).copied() {
                paired_ltr_scores.push((relevance, document.ltr.ltr_score));
                paired_context_scores.push((relevance, document.context.context_value));

                // nDCG expects higher scores to be better but for the ranking
                // it's the oposite, the solution carried over from the dart impl
                // is to multiply by -1.
                let final_ranking_desc = -(document.mab.rank as f32);
                paired_final_ranking_scores.push((relevance, final_ranking_desc));

                let intial_ranking_desc = -(document.document_base.initial_ranking as f32);
                paired_initial_ranking_scores.push((relevance, intial_ranking_desc));
            }
        }

        if paired_ltr_scores.is_empty() {
            bail!(NoRelevantHistoricInfo);
        }

        let ndcg_ltr = calcuate_reordered_ndcg_at_k_score(&mut paired_ltr_scores, DEFAULT_NDCG_K);

        let ndcg_context =
            calcuate_reordered_ndcg_at_k_score(&mut paired_context_scores, DEFAULT_NDCG_K);

        let ndcg_final_ranking =
            calcuate_reordered_ndcg_at_k_score(&mut paired_final_ranking_scores, DEFAULT_NDCG_K);

        let ndcg_initial_ranking =
            calcuate_reordered_ndcg_at_k_score(&mut paired_initial_ranking_scores, DEFAULT_NDCG_K);

        Ok(Analytics {
            ndcg_ltr,
            ndcg_context,
            ndcg_initial_ranking,
            ndcg_final_ranking,
        })
    }
}

/// Returns a score for the given `Relevance`.
fn score_for_relevance(relevance: Relevance) -> f32 {
    match relevance {
        Relevance::Low => 0.,
        Relevance::Medium => 1.,
        Relevance::High => 2.,
    }
}

/// Calculates the nDCG@k for given paired relevances.
///
/// The input is a slice over `(relevance, ordering_score)` pairs,
/// where the `ordering_score` is used to reorder the relevances
/// based on sorting them in descending order.
///
/// **Note that the `paired_relevances` are sorted in place.**
///
/// After the reordering of the pairs the `relevance` values
/// are used to calculate the nDCG@k.
///
/// ## NaN Handling.
///
/// NaN values are treated as the lowest possible socres wrt. the sorting.
///
/// If a `NaN` is in the k-first relevances the resulting nDCG@k score will be `NaN`.
fn calcuate_reordered_ndcg_at_k_score(paired_relevances: &mut [(f32, f32)], k: usize) -> f32 {
    paired_relevances
        .sort_by(|(_, ord_sc_1), (_, ord_sc_2)| nan_safe_f32_cmp_desc(ord_sc_1, ord_sc_2));
    ndcg_at_k(paired_relevances.iter().map(|(rel, _ord)| *rel), k)
}

/// Calculates the nDCG@k.
///
/// This taks the first k values for the DCG score and the "best" k values
/// for the IDCG score and then calculates the nDCG score with that.
fn ndcg_at_k(relevances: impl Iterator<Item = f32> + Clone + ExactSizeIterator, k: usize) -> f32 {
    let dcg_at_k = dcg(relevances.clone().take(k));

    let ideal_relevances = pick_k_highest_sorted_desc(relevances, k);
    let idcg_at_k = dcg(ideal_relevances.into_iter());

    // if there is no ideal score, pretent the ideal score is 1
    if idcg_at_k == 0.0 {
        dcg_at_k
    } else {
        dcg_at_k / idcg_at_k
    }
}

/// Pick the k-highest values in given iterator.
///
/// (As if a vector is sorted and then &sorted_score[..k]).
///
/// If `NaN`'s is treated as the smallest possible value, i.e.
/// preferably not picked at all if possible.
fn pick_k_highest_sorted_desc(
    scores: impl Iterator<Item = f32> + ExactSizeIterator,
    k: usize,
) -> Vec<f32> {
    // Due to specialization this has no overhead if scores is already fused.
    let mut scores = scores.fuse();
    let mut k_highest: Vec<_> = (&mut scores).take(k).collect();

    k_highest.sort_by(nan_safe_f32_cmp_desc);

    for score in scores {
        //Supposed to act as NaN safe version of: if k_highest[k-1] < score {
        if nan_safe_f32_cmp_desc(&k_highest[k - 1], &score) == Ordering::Greater {
            let _ = k_highest.pop();

            let idx = k_highest
                .binary_search_by(|other| nan_safe_f32_cmp_desc(other, &score))
                .unwrap_or_else(|not_found_insert_idx| not_found_insert_idx);

            k_highest.insert(idx, score);
        }
    }

    k_highest
}

/// Calculates the DCG of given input sequence.
fn dcg(scores: impl Iterator<Item = f32>) -> f32 {
    // - As this is only used for analytics and bound by `k`(==2) and `&[Document].len()` (~ 10 to 40)
    //   no further optimizations make sense. Especially not if they require memory allocations.
    // - A "simple commulative" sum is ok as we only use small number of scores (default k=2)
    scores.enumerate().fold(0.0, |sum, (idx, score)| {
        //it's i+2 as our i starts with 0, while the formular starts with 1 and uses i+1
        sum + (2f32.powf(score) - 1.) / (idx as f32 + 2.).log2()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{reranker::systems::AnalyticsSystem, tests, UserFeedback};

    #[test]
    fn test_full_analytics_system() {
        let history = tests::document_history(vec![
            (2, Relevance::Low, UserFeedback::None),
            (3, Relevance::Medium, UserFeedback::None),
            (1, Relevance::High, UserFeedback::None),
            (0, Relevance::Medium, UserFeedback::None),
            (10, Relevance::Low, UserFeedback::None),
        ]);

        let mut documents = tests::data_with_mab(tests::from_ids(0..3));
        documents[0].ltr.ltr_score = 3.;
        documents[0].context.context_value = 3.5;
        documents[0].mab.rank = 1;

        documents[1].ltr.ltr_score = 2.;
        documents[1].context.context_value = 7.;
        documents[1].mab.rank = 0;

        documents[2].ltr.ltr_score = 7.;
        documents[2].context.context_value = 6.;
        documents[2].mab.rank = 2;

        let Analytics {
            ndcg_ltr,
            ndcg_context,
            ndcg_initial_ranking,
            ndcg_final_ranking,
        } = AnalyticsSystem
            .compute_analytics(&history, &documents)
            .unwrap();

        assert_approx_eq!(f32, ndcg_ltr, 0.173_765_35);
        assert_approx_eq!(f32, ndcg_context, 0.826_234_64);
        assert_approx_eq!(f32, ndcg_initial_ranking, 0.796_707_6);
        assert_approx_eq!(f32, ndcg_final_ranking, 1.0);
    }

    #[test]
    fn test_calcuate_reordered_ndcg_at_k_score_tests_from_dart() {
        let relevances = &mut [(0., -50.), (0., 0.001), (1., 4.14), (2., 1000.)];
        let res = calcuate_reordered_ndcg_at_k_score(relevances, 2);
        assert_eq!(format!("{:.4}", res), "1.0000");

        let relevances = &mut [(0., -10.), (0., 1.), (1., 0.), (2., 6.)];
        let res = calcuate_reordered_ndcg_at_k_score(relevances, 2);
        let res2 = ndcg_at_k([2., 0., 1., 0.].iter().copied(), 2);
        assert_approx_eq!(f32, res, res2);

        let relevances = &mut [(0., 1.), (0., -10.), (1., -11.), (2., -11.6)];
        let res = calcuate_reordered_ndcg_at_k_score(relevances, 2);
        assert_approx_eq!(f32, res, 0.0);

        let relevances = &mut [(0., 1.), (0., -10.), (1., 100.), (2., 99.)];
        let res = calcuate_reordered_ndcg_at_k_score(relevances, 2);
        let res2 = ndcg_at_k([1., 2., 1., 0.].iter().copied(), 2);
        assert_approx_eq!(f32, res, res2);
    }

    #[test]
    fn test_calcuate_reordered_ndcg_at_k_score_without_reordering() {
        let relevances = &mut [(1., 12.), (4., 9.), (10., 7.), (3., 5.), (0., 4.), (6., 1.)];
        let res = calcuate_reordered_ndcg_at_k_score(relevances, 2);
        assert_approx_eq!(f32, res, 0.009_846_116);

        let relevances = &mut [(1., 12.), (4., 9.), (10., 7.), (3., 5.), (0., 4.), (6., 1.)];
        let res = calcuate_reordered_ndcg_at_k_score(relevances, 4);
        assert_approx_eq!(f32, res, 0.489_142_48);

        let relevances = &mut [(1., 12.), (4., 9.), (10., 7.), (3., 5.), (0., 4.), (6., 1.)];
        let res = calcuate_reordered_ndcg_at_k_score(relevances, 100);
        assert_approx_eq!(f32, res, 0.509_867_9);

        let relevances = &mut [
            (-1., 12.),
            (7., 9.),
            (-10., 7.),
            (3., 5.),
            (0., 4.),
            (-6., 1.),
        ];
        let res = calcuate_reordered_ndcg_at_k_score(relevances, 2);
        assert_approx_eq!(f32, res, 0.605_921_45);

        let relevances = &mut [
            (-1., 12.),
            (7., 9.),
            (-10., 7.),
            (3., 5.),
            (0., 4.),
            (-6., 1.),
        ];
        let res = calcuate_reordered_ndcg_at_k_score(relevances, 4);
        assert_approx_eq!(f32, res, 0.626_086_65);

        let relevances = &mut [
            (-1., 12.),
            (7., 9.),
            (-10., 7.),
            (3., 5.),
            (0., 4.),
            (-6., 1.),
        ];
        let res = calcuate_reordered_ndcg_at_k_score(relevances, 100);
        assert_approx_eq!(f32, res, 0.626_934_23);
    }

    #[test]
    fn test_calcuate_reordered_ndcg_at_k_score_with_reordering() {
        let relevances = &mut [(4., 9.), (10., 7.), (6., 1.), (0., 4.), (3., 5.), (1., 12.)];
        let res = calcuate_reordered_ndcg_at_k_score(relevances, 2);
        assert_approx_eq!(f32, res, 0.009_846_116);

        let relevances = &mut [(4., 9.), (10., 7.), (6., 1.), (0., 4.), (3., 5.), (1., 12.)];
        let res = calcuate_reordered_ndcg_at_k_score(relevances, 4);
        assert_approx_eq!(f32, res, 0.489_142_48);

        let relevances = &mut [(4., 9.), (10., 7.), (6., 1.), (0., 4.), (3., 5.), (1., 12.)];
        let res = calcuate_reordered_ndcg_at_k_score(relevances, 100);
        assert_approx_eq!(f32, res, 0.509_867_9);

        let relevances = &mut [
            (3., 5.),
            (-10., 7.),
            (0., 4.),
            (-1., 12.),
            (7., 9.),
            (-6., 1.),
        ];
        let res = calcuate_reordered_ndcg_at_k_score(relevances, 2);
        assert_approx_eq!(f32, res, 0.605_921_45);

        let relevances = &mut [
            (3., 5.),
            (-10., 7.),
            (0., 4.),
            (-1., 12.),
            (7., 9.),
            (-6., 1.),
        ];
        let res = calcuate_reordered_ndcg_at_k_score(relevances, 4);
        assert_approx_eq!(f32, res, 0.626_086_65);

        let relevances = &mut [
            (3., 5.),
            (-10., 7.),
            (0., 4.),
            (-1., 12.),
            (7., 9.),
            (-6., 1.),
        ];
        let res = calcuate_reordered_ndcg_at_k_score(relevances, 100);
        assert_approx_eq!(f32, res, 0.626_934_23);
    }

    #[test]
    fn test_ndcg_at_k_tests_from_dart() {
        let res = ndcg_at_k([0., 0., 1., 2.].iter().copied(), 4);
        assert_eq!(format!("{:.4}", res), "0.4935");
    }

    #[test]
    fn test_ndcg_at_k_produces_expected_values_for_k_larger_then_input() {
        let res = ndcg_at_k([1., 4., 10., 3., 0., 6.].iter().copied(), 100);
        assert_approx_eq!(f32, res, 0.509_867_9);

        let res = ndcg_at_k([-1., 7., -10., 3., 0., -6.].iter().copied(), 100);
        assert_approx_eq!(f32, res, 0.626_934_23);
    }

    #[test]
    fn test_ndcg_at_k_produces_expected_values_for_k_smaller_then_input() {
        let res = ndcg_at_k([1., 4., 10., 3., 0., 6.].iter().copied(), 2);
        assert_approx_eq!(f32, res, 0.009_846_116);
        let res = ndcg_at_k([1., 4., 10., 3., 0., 6.].iter().copied(), 4);
        assert_approx_eq!(f32, res, 0.489_142_48);

        let res = ndcg_at_k([-1., 7., -10., 3., 0., -6.].iter().copied(), 2);
        assert_approx_eq!(f32, res, 0.605_921_45);
        let res = ndcg_at_k([-1., 7., -10., 3., 0., -6.].iter().copied(), 4);
        assert_approx_eq!(f32, res, 0.626_086_65);
    }

    #[test]
    fn test_ndcg_at_k_works_if_the_ideal_dcg_is_0() {
        let res = ndcg_at_k([0.0, 0.0].iter().copied(), 2);
        assert_f32_eq!(res, 0.0);
        let res = ndcg_at_k([-10.0, 0.0, 0.0, -8.0].iter().copied(), 2);
        assert_f32_eq!(res, -0.999_023_44);
    }

    #[test]
    fn test_dcg_tests_from_dart() {
        // there is no dcg@k function in my code. It's dcg(input_iter.take(k)).
        let res = dcg([0., 0., 1., 1.].iter().copied().take(2));
        assert_approx_eq!(f32, res, 0.0);

        let res = dcg([0., 0., 1., 1.].iter().copied().take(4));
        // Dart used ln instead of log2 so the values diverge.
        // assert_eq!(format!("{:.4}", res), "1.3426");
        assert_eq!(format!("{:.4}", res), "0.9307");

        let res = dcg([0., 0., 1., 2.].iter().copied().take(4));
        // Dart used ln instead of log2 so the values diverge.
        // assert_eq!(format!("{:.4}", res), "2.5853");
        assert_eq!(format!("{:.4}", res), "1.7920");
    }

    #[test]
    fn dcg_produces_expected_results() {
        assert_approx_eq!(
            f32,
            dcg([3f32, 2., 3., 0., 1., 2.].iter().copied()),
            13.848_264
        );
        assert_approx_eq!(
            f32,
            dcg([-3.2, -2., -4., 0., -1., -2.].iter().copied()),
            -2.293_710_2
        );
    }

    #[test]
    fn test_pick_k_highest_picks_the_highest_values_and_only_them() {
        let cases: &[(&[f32], &[f32])] = &[
            (&[3., 2., 1., 0.], &[3., 2.]),
            (&[0., 1., 2., 3.], &[3., 2.]),
            (&[-2., -2.], &[-2., -2.]),
            (&[-30., 3., 2., 10., -3., 0.], &[10., 3.]),
            (&[-3., 0., -1., -2.], &[0., -1.]),
        ];

        for (input, pick) in cases {
            let res = pick_k_highest_sorted_desc(input.iter().copied(), 2);
            assert_eq!(
                &*res, &**pick,
                "res={:?}, expected={:?}, input={:?}",
                res, pick, input
            );
        }
    }

    #[test]
    fn test_pick_k_highest_does_not_pick_nans_if_possible() {
        #![allow(clippy::float_cmp)]

        let res = pick_k_highest_sorted_desc([3., 2., f32::NAN].iter().copied(), 2);
        assert_eq!(&*res, &[3., 2.]);

        let res = pick_k_highest_sorted_desc(
            [f32::NAN, 3., f32::NAN, f32::NAN, 2., 4., f32::NAN]
                .iter()
                .copied(),
            2,
        );
        assert_eq!(&*res, &[4., 3.]);

        let res = pick_k_highest_sorted_desc([f32::NAN, 3., 2., f32::NAN].iter().copied(), 3);
        assert_eq!(&res[..2], &[3., 2.]);
        assert!(res[2].is_nan());

        let res = pick_k_highest_sorted_desc([f32::NAN].iter().copied(), 1);
        assert_eq!(res.len(), 1);
        assert!(res[0].is_nan());
    }
}
