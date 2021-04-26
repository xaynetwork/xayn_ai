use std::{cmp::Ordering, collections::HashMap, iter::FusedIterator};

use crate::{
    data::{document::DocumentHistory, document_data::DocumentDataWithMab},
    error::Error,
    reranker::systems,
    Relevance,
};

/// Which k to use for nDCG@k
const DEFAULT_NDCG_K: usize = 2;
/// Calculated analytics data.
#[derive(Clone)]
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
        let relevance_lookups: HashMap<_, _> = {
            history
                .iter()
                .map(|h_doc| (&h_doc.id, score_for_relevance(h_doc.relevance)))
                .collect()
        };

        let mut paired_ltr_scores = Vec::new();
        let mut paired_context_scores = Vec::new();
        let mut paired_final_ranking_score = Vec::new();

        for document in documents {
            // We should never need the `_or(0.0)` but if we run into
            // it it's best to give it a relevance of 0. As a document
            // not in the history is irrelevant for this analytics.
            let relevance = relevance_lookups
                .get(&document.document_id.id)
                .copied()
                .unwrap_or(0.0);

            paired_ltr_scores.push((relevance, document.ltr.ltr_score));
            paired_context_scores.push((relevance, document.context.context_value));

            // nDCG expects higher scores to be better but for the ranking
            // it's the oposite, the solution carried over from the dart impl
            // is to multiply by -1.
            let final_ranking_desc = -(document.mab.rank as f32);
            paired_final_ranking_score.push((relevance, final_ranking_desc));
        }

        let ndcg_ltr = calcuate_reordered_ndcg_at_k_score(&mut paired_ltr_scores, DEFAULT_NDCG_K);

        let ndcg_context =
            calcuate_reordered_ndcg_at_k_score(&mut paired_context_scores, DEFAULT_NDCG_K);

        let ndcg_final_ranking =
            calcuate_reordered_ndcg_at_k_score(&mut paired_final_ranking_score, DEFAULT_NDCG_K);

        Ok(Analytics {
            //FIXME: We currently have no access to the initial score as thiss will require
            //       some changes to the main applications type state/component system this
            //       will be done in a followup PR.
            ndcg_initial_ranking: f32::NAN,
            ndcg_ltr,
            ndcg_context,
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
        .sort_by(|(_, ord_sc_1), (_, ord_sc_2)| nan_safe_sort_desc_comparsion(ord_sc_1, ord_sc_2));
    ndcg_at_k(paired_relevances.iter().map(|(rel, _ord)| *rel), k)
}

/// Calculates the nDCG@k.
///
/// This taks the first k values for the DCG score and the "best" k values
/// for the IDCG score and then calculates the nDCG score with that.
fn ndcg_at_k(
    relevances: impl Iterator<Item = f32> + Clone + ExactSizeIterator + FusedIterator,
    k: usize,
) -> f32 {
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
    mut scores: impl Iterator<Item = f32> + ExactSizeIterator + FusedIterator,
    k: usize,
) -> Vec<f32> {
    let mut k_highest: Vec<_> = (&mut scores).take(k).collect();

    k_highest.sort_by(nan_safe_sort_desc_comparsion);

    for score in scores {
        let idx = k_highest
            .binary_search_by(|other| nan_safe_sort_desc_comparsion(other, &score))
            .unwrap_or_else(|not_found_insert_idx| not_found_insert_idx);

        if idx < k {
            let _ = k_highest.pop();
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
    let mut sum = 0.;
    for (i, score) in scores.enumerate() {
        //it's i+2 as our i starts with 0, while the formular starts with 1 and uses i+1
        sum += (2f32.powf(score) - 1.) / (i as f32 + 2.).log2()
    }
    sum
}

/// Use for getting a descending sort ordering of floats.
///
/// `NaN`'s are treated as the smallest possible value
/// for this sorting they are also treated as equal to each other.
/// This is not standard comform but works for sorting,
/// at least for our use-case.
fn nan_safe_sort_desc_comparsion(a: &f32, b: &f32) -> Ordering {
    // switched a,b to have descending instead of ascending sorting
    b.partial_cmp(a)
        .unwrap_or_else(|| match (a.is_nan(), b.is_nan()) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Greater,
            (false, true) => Ordering::Less,
            (false, false) => unreachable!(),
        })
}

#[cfg(test)]
mod tests {
    use crate::{reranker::systems::AnalyticsSystem, tests, UserFeedback};

    use super::*;
    use float_cmp::approx_eq;

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
            ndcg_initial_ranking: _,
            ndcg_ltr,
            ndcg_context,
            ndcg_final_ranking,
        } = AnalyticsSystem
            .compute_analytics(&history, &documents)
            .unwrap();

        assert!(approx_eq!(f32, ndcg_ltr, 0.17376534287144002, ulps = 2));
        assert!(approx_eq!(f32, ndcg_context, 0.8262346571285599, ulps = 2));
        //FIXME: Currently not possible as `ndcg_initial_ranking` is not yet computed
        // assert!(approx_eq!(f32, ndcg_initial_ranking, 0.7967075809905066, ulps = 2));
        assert!(approx_eq!(f32, ndcg_final_ranking, 1.0, ulps = 2));
    }

    #[test]
    fn test_calcuate_reordered_ndcg_at_k_score_tests_from_dart() {
        let relevances = &mut [(0., -50.), (0., 0.001), (1., 4.14), (2., 1000.)];
        let res = calcuate_reordered_ndcg_at_k_score(relevances, 2);
        assert_eq!(format!("{:.4}", res), "1.0000");

        let relevances = &mut [(0., -10.), (0., 1.), (1., 0.), (2., 6.)];
        let res = calcuate_reordered_ndcg_at_k_score(relevances, 2);
        let res2 = ndcg_at_k([2., 0., 1., 0.].iter().copied(), 2);
        assert!(approx_eq!(f32, res, res2, ulps = 2));

        let relevances = &mut [(0., 1.), (0., -10.), (1., -11.), (2., -11.6)];
        let res = calcuate_reordered_ndcg_at_k_score(relevances, 2);
        assert!(approx_eq!(f32, res, 0.0, ulps = 2));

        let relevances = &mut [(0., 1.), (0., -10.), (1., 100.), (2., 99.)];
        let res = calcuate_reordered_ndcg_at_k_score(relevances, 2);
        let res2 = ndcg_at_k([1., 2., 1., 0.].iter().copied(), 2);
        assert!(approx_eq!(f32, res, res2, ulps = 2));
    }

    #[test]
    fn test_calcuate_reordered_ndcg_at_k_score_without_reordering() {
        let relevances = &mut [(1., 12.), (4., 9.), (10., 7.), (3., 5.), (0., 4.), (6., 1.)];
        let res = calcuate_reordered_ndcg_at_k_score(relevances, 2);
        assert!(approx_eq!(f32, res, 0.009_846_116, ulps = 2));

        let relevances = &mut [(1., 12.), (4., 9.), (10., 7.), (3., 5.), (0., 4.), (6., 1.)];
        let res = calcuate_reordered_ndcg_at_k_score(relevances, 4);
        assert!(approx_eq!(f32, res, 0.489_142_48, ulps = 2));

        let relevances = &mut [(1., 12.), (4., 9.), (10., 7.), (3., 5.), (0., 4.), (6., 1.)];
        let res = calcuate_reordered_ndcg_at_k_score(relevances, 100);
        assert!(approx_eq!(f32, res, 0.509_867_9, ulps = 2));

        let relevances = &mut [
            (-1., 12.),
            (7., 9.),
            (-10., 7.),
            (3., 5.),
            (0., 4.),
            (-6., 1.),
        ];
        let res = calcuate_reordered_ndcg_at_k_score(relevances, 2);
        assert!(approx_eq!(f32, res, 0.605_921_45, ulps = 2));

        let relevances = &mut [
            (-1., 12.),
            (7., 9.),
            (-10., 7.),
            (3., 5.),
            (0., 4.),
            (-6., 1.),
        ];
        let res = calcuate_reordered_ndcg_at_k_score(relevances, 4);
        assert!(approx_eq!(f32, res, 0.626_086_65, ulps = 2));

        let relevances = &mut [
            (-1., 12.),
            (7., 9.),
            (-10., 7.),
            (3., 5.),
            (0., 4.),
            (-6., 1.),
        ];
        let res = calcuate_reordered_ndcg_at_k_score(relevances, 100);
        assert!(approx_eq!(f32, res, 0.626_934_23, ulps = 2));
    }

    #[test]
    fn test_calcuate_reordered_ndcg_at_k_score_with_reordering() {
        let relevances = &mut [(4., 9.), (10., 7.), (6., 1.), (0., 4.), (3., 5.), (1., 12.)];
        let res = calcuate_reordered_ndcg_at_k_score(relevances, 2);
        assert!(approx_eq!(f32, res, 0.009_846_116, ulps = 2));

        let relevances = &mut [(4., 9.), (10., 7.), (6., 1.), (0., 4.), (3., 5.), (1., 12.)];
        let res = calcuate_reordered_ndcg_at_k_score(relevances, 4);
        assert!(approx_eq!(f32, res, 0.489_142_48, ulps = 2));

        let relevances = &mut [(4., 9.), (10., 7.), (6., 1.), (0., 4.), (3., 5.), (1., 12.)];
        let res = calcuate_reordered_ndcg_at_k_score(relevances, 100);
        assert!(approx_eq!(f32, res, 0.509_867_9, ulps = 2));

        let relevances = &mut [
            (3., 5.),
            (-10., 7.),
            (0., 4.),
            (-1., 12.),
            (7., 9.),
            (-6., 1.),
        ];
        let res = calcuate_reordered_ndcg_at_k_score(relevances, 2);
        assert!(approx_eq!(f32, res, 0.605_921_45, ulps = 2));

        let relevances = &mut [
            (3., 5.),
            (-10., 7.),
            (0., 4.),
            (-1., 12.),
            (7., 9.),
            (-6., 1.),
        ];
        let res = calcuate_reordered_ndcg_at_k_score(relevances, 4);
        assert!(approx_eq!(f32, res, 0.626_086_65, ulps = 2));

        let relevances = &mut [
            (3., 5.),
            (-10., 7.),
            (0., 4.),
            (-1., 12.),
            (7., 9.),
            (-6., 1.),
        ];
        let res = calcuate_reordered_ndcg_at_k_score(relevances, 100);
        assert!(approx_eq!(f32, res, 0.626_934_23, ulps = 2));
    }

    #[test]
    fn test_ndcg_at_k_tests_from_dart() {
        let res = ndcg_at_k([0., 0., 1., 2.].iter().copied(), 4);
        assert_eq!(format!("{:.4}", res), "0.4935");
    }

    #[test]
    fn ndcg_at_k_produces_expected_values_for_k_larger_then_input() {
        let res = ndcg_at_k([1., 4., 10., 3., 0., 6.].iter().copied(), 100);
        assert!(approx_eq!(f32, res, 0.509_867_9, ulps = 2));

        let res = ndcg_at_k([-1., 7., -10., 3., 0., -6.].iter().copied(), 100);
        assert!(approx_eq!(f32, res, 0.626_934_23, ulps = 2));
    }

    #[test]
    fn ndcg_at_k_produces_expected_values_for_k_smaller_then_input() {
        let res = ndcg_at_k([1., 4., 10., 3., 0., 6.].iter().copied(), 2);
        assert!(approx_eq!(f32, res, 0.009_846_116, ulps = 2));
        let res = ndcg_at_k([1., 4., 10., 3., 0., 6.].iter().copied(), 4);
        assert!(approx_eq!(f32, res, 0.489_142_48, ulps = 2));

        let res = ndcg_at_k([-1., 7., -10., 3., 0., -6.].iter().copied(), 2);
        assert!(approx_eq!(f32, res, 0.605_921_45, ulps = 2));
        let res = ndcg_at_k([-1., 7., -10., 3., 0., -6.].iter().copied(), 4);
        assert!(approx_eq!(f32, res, 0.626_086_65, ulps = 2));
    }

    #[test]
    fn test_dcg_tests_from_dart() {
        /*
            List<double> relevances1 = [0, 0, 1, 1];
            List<double> relevances2 = [0, 0, 1, 2];
            expect(Metrics.dcgAtK(relevances1, 2), 0);
            expect(Metrics.dcgAtK(relevances1, 4).toStringAsFixed(4), 1.34268.toStringAsFixed(4));
            expect(Metrics.dcgAtK(relevances2, 4).toStringAsFixed(4), 2.5853523.toStringAsFixed(4));
        */
        // there is no dcg@k function in my code. It's dcg(input_iter.take(k)).
        let res = dcg([0., 0., 1., 1.].iter().copied().take(2));
        assert!(approx_eq!(f32, res, 0.0, ulps = 2));

        // FIXME: It turns out dart uses `log` (natural) but we and wikipedia do use `log2`...
        // so this test will fail if the dart test values are used.
        // Until this is resolved I will used the values from calculating the result
        // "by hand: using log2.
        let res = dcg([0., 0., 1., 1.].iter().copied().take(4));
        // assert_eq!(format!("{:.4}", res), "1.3426");
        assert_eq!(format!("{:.4}", res), "0.9307");

        let res = dcg([0., 0., 1., 2.].iter().copied().take(4));
        // assert_eq!(format!("{:.4}", res), "2.5853");
        assert_eq!(format!("{:.4}", res), "1.7920");
    }

    #[test]
    fn dcg_produces_expected_results() {
        assert!(approx_eq!(
            f32,
            dcg([3f32, 2., 3., 0., 1., 2.].iter().copied()),
            13.848_264,
            ulps = 2
        ));
        assert!(approx_eq!(
            f32,
            dcg([-3.2, -2., -4., 0., -1., -2.].iter().copied()),
            -2.293_710_2,
            ulps = 2
        ));
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

    #[test]
    fn test_nan_safe_sort_desc_comparsion_sorts_in_the_right_order() {
        #![allow(clippy::float_cmp)]

        let data = &mut [f32::NAN, 1., 5., f32::NAN, 4.];
        data.sort_by(nan_safe_sort_desc_comparsion);

        assert_eq!(&data[..3], &[5., 4., 1.]);
        assert!(data[3].is_nan());
        assert!(data[4].is_nan());

        let data = &mut [1., 5., 3., 4.];
        data.sort_by(nan_safe_sort_desc_comparsion);

        assert_eq!(&data[..], &[5., 4., 3., 1.]);
    }

    #[test]
    fn test_nan_safe_sort_desc_comparsion_nans_compare_as_expected() {
        assert_eq!(
            nan_safe_sort_desc_comparsion(&f32::NAN, &f32::NAN),
            Ordering::Equal
        );
        assert_eq!(
            nan_safe_sort_desc_comparsion(&-12., &f32::NAN),
            Ordering::Less
        );
        assert_eq!(
            nan_safe_sort_desc_comparsion(&f32::NAN, &-12.),
            Ordering::Greater
        );
        assert_eq!(
            nan_safe_sort_desc_comparsion(&12., &f32::NAN),
            Ordering::Less
        );
        assert_eq!(
            nan_safe_sort_desc_comparsion(&f32::NAN, &12.),
            Ordering::Greater
        );
    }
}
