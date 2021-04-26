use std::cmp::{min, Ordering};

use crate::{
    data::{document::DocumentHistory, document_data::DocumentDataWithMab},
    error::Error,
    reranker::systems,
    Relevance,
};

/// Which k to use for nDCG@k
const DEFAULT_NDCG_K: usize = 2;
#[derive(Clone)]
pub struct Analytics {
    /// The nDCG@k score between the initial ranking and the relevance based ranking
    pub ndcg_initial: f32,
    /// The nDCG@k score between the LTR ranking and the relevance based ranking
    pub ndcg_ltr: f32,
    /// The nDCG@k score between the Context ranking and the relevance based ranking
    pub ndcg_context: f32,
    /// THe nDCG@k score between the final ranking and the relevance based ranking
    pub ndcg_final_ranking: f32,
}

pub(crate) struct AnalyticsSystem;

impl systems::AnalyticsSystem for AnalyticsSystem {
    fn compute_analytics(
        &self,
        history: &[DocumentHistory],
        documents: &[DocumentDataWithMab],
    ) -> Result<Analytics, Error> {
        let mut relevances = Vec::new();
        let mut ltr_scores = Vec::new();
        let mut context_scores = Vec::new();
        let mut final_ranking_score = Vec::new();

        for document in documents {
            // - FIXME this is *slow* we probably want to
            //   have some lookup by DocumentId for
            //   this. Depending on a lot of factors it
            //   might make sense to create a hashmap once
            //   before the loop. (But that might also be slower
            //   depending of the size of history...).
            // - FIXME the dart version doesn't hadnle the
            //   not found case? Should I panic?
            //   Currently I give it a relevance of 0,
            //   which will lead to the entry not having
            //   an effect on the final score which is nice
            //   so even if it can't happen it might be good
            //   to just do so anyway instead of panicing.
            let relevance = history
                .iter()
                .find(|h| &h.id == &document.document_id.id)
                .map(|h| match h.relevance {
                    Relevance::Low => 0.,
                    Relevance::Medium => 1.,
                    Relevance::High => 2.,
                })
                .unwrap_or(0.0);
            relevances.push(relevance);
            ltr_scores.push(document.ltr.ltr_score);
            context_scores.push(document.context.context_value);
            // nDCG expects higher scores to be better but for the ranking
            // it's the oposite, the solution carried over from the dart impl
            // is to multiply by -1. Another would be to have the max rank (or
            // and number greater then it which isn't too big) and then use `max-rank`.
            // While negative ranks work mathematically fine I'm not sure about
            // rounding problems due to f32. I really can't judge it it's a problem
            // or not.
            final_ranking_score.push(-(document.mab.rank as f32));
        }

        let pair_buffer = &mut Vec::with_capacity(relevances.len());

        // FIXME with cloneable/resetable iterators we can eleminate this buffer
        let buffer = &mut Vec::with_capacity(relevances.len());

        let ndcg_ltr = calcuate_reordered_ndcg_at_k_score(
            relevances.iter().copied(),
            ltr_scores,
            DEFAULT_NDCG_K,
            pair_buffer,
            buffer,
        );

        let ndcg_context = calcuate_reordered_ndcg_at_k_score(
            relevances.iter().copied(),
            context_scores,
            DEFAULT_NDCG_K,
            pair_buffer,
            buffer,
        );

        let ndcg_final_ranking = calcuate_reordered_ndcg_at_k_score(
            relevances.iter().copied(),
            final_ranking_score,
            DEFAULT_NDCG_K,
            pair_buffer,
            buffer,
        );

        Ok(Analytics {
            //FIXME: We currently have no access to the initial score as thiss will require
            //       some changes to the main applications type state/component system this
            //       will be done in a followup PR.
            ndcg_initial: f32::NAN,
            ndcg_ltr,
            ndcg_context,
            ndcg_final_ranking,
        })
    }
}

fn calcuate_reordered_ndcg_at_k_score(
    relevances: impl IntoIterator<Item = f32>,
    ordering_scores: impl IntoIterator<Item = f32>,
    k: usize,
    pair_buffer: &mut Vec<(f32, f32)>,
    buffer: &mut Vec<f32>,
) -> f32 {
    reorder_relevances_based_on_scores_replacing_buffer(relevances, ordering_scores, pair_buffer);
    copy_second_value_replacing_buffer(pair_buffer.drain(..), buffer);
    ndcg_at_k(&buffer[..], k)
}

fn reorder_relevances_based_on_scores_replacing_buffer(
    relevances: impl IntoIterator<Item = f32>,
    ordering_scores: impl IntoIterator<Item = f32>,
    buffer: &mut Vec<(f32, f32)>,
) {
    buffer.truncate(0);
    buffer.extend(ordering_scores.into_iter().zip(relevances.into_iter()));
    buffer
        .sort_by(|(ord_sc_1, _), (ord_sc_2, _)| nan_safe_sort_desc_comparsion(ord_sc_1, ord_sc_2));
}

fn copy_second_value_replacing_buffer(
    input: impl IntoIterator<Item = (f32, f32)>,
    output: &mut Vec<f32>,
) {
    output.truncate(0);
    output.extend(input.into_iter().map(|(_, second)| second));
}

/// Calculates the nDCG@k, `k` defaults to 2 if `None` is passed in.
///
/// This taks the first k values for the DCG score and the "best" k values
/// for the IDCG score and then calculates the nDCG score with that.
fn ndcg_at_k(scores: &[f32], k: usize) -> f32 {
    // if we have less then k values we just use a smaller k
    // it's mathematically equivalent to padding with 0 scores.
    let k = min(k, scores.len());

    let dcg_at_k = dcg(&scores[..k]);

    let other_scores = pick_k_highest_scores(scores, k);
    let idcg_at_k = dcg(&other_scores);

    // if there is no ideal score our score pretent the ideal score is 1
    if idcg_at_k == 0.0 {
        dcg_at_k
    } else {
        dcg_at_k / idcg_at_k
    }
}

/// Pick the k-highest values (as if score.sort() and then &score[..k]).
///
/// If `NaN`'s is treated as the smallest possible value, i.e.
/// preferably not picked at all if possible.
///
/// # Panics
///
/// If `k > scores.len()` this will panic.
fn pick_k_highest_scores(scores: &[f32], k: usize) -> Vec<f32> {
    let mut k_highest = Vec::from(&scores[..k]);

    // TODO: Potentially handle `NaN` better by treating them as
    // "lowest possible" scores.
    k_highest.sort_by(nan_safe_sort_desc_comparsion);

    for score in &scores[k..] {
        let idx = k_highest
            .binary_search_by(|other| nan_safe_sort_desc_comparsion(other, score))
            .unwrap_or_else(|not_found_insert_idx| not_found_insert_idx);

        if idx < k {
            let _ = k_highest.pop();
            k_highest.insert(idx, *score);
        }
    }

    k_highest
}

fn dcg(scores: &[f32]) -> f32 {
    //Note: It migth seem to be faster to create two ndarrays and then use
    //      a / broadcast in the hope this will take advantage of SIMD at
    //      least on some platforms. But given that `scores` is more or
    //      less alwasys very small (e.g. k=2) this is unlikely to yield
    //      any benefits and migth even slow things down due to uneccesary
    //      allocation. If k is fixed we could use stack allocated buffers
    //      and a tight loop, which problably would be the fastest.

    // a "simple commulative" sum is ok as we only use small number of scores (default k=2)
    let mut sum = 0.;
    for (i, score) in scores.iter().copied().enumerate() {
        //it's i+2 as our i starts with 0, while the formular starts with 1 and uses i+1
        sum += (2f32.powf(score) - 1.) / (i as f32 + 2.).log2()
    }
    sum
}

/// Use for getting a descending ordering of floats.
///
/// `NaN`'s are treated as the smallest possible value
/// for this sorting they are also treated as equal to each other.
/// this is not standard comform but works for sorting.
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

    mod create_reordered_ndcg_at_k_score {
        use super::super::*;
        use float_cmp::approx_eq;

        #[test]
        fn without_reordering() {
            let buffer = &mut Vec::with_capacity(6);
            let pair_buffer = &mut Vec::with_capacity(6);

            let relevances = &[1., 4., 10., 3., 0., 6.];
            let ordering_scores = &[12., 9., 7., 5., 4., 1.];
            let res = calcuate_reordered_ndcg_at_k_score(
                relevances.iter().copied(),
                ordering_scores.iter().copied(),
                2,
                pair_buffer,
                buffer,
            );
            assert!(approx_eq!(f32, res, 0.009846116527364958, ulps = 2));

            let res = calcuate_reordered_ndcg_at_k_score(
                relevances.iter().copied(),
                ordering_scores.iter().copied(),
                4,
                pair_buffer,
                buffer,
            );
            assert!(approx_eq!(f32, res, 0.4891424845441425, ulps = 2));

            let res = calcuate_reordered_ndcg_at_k_score(
                relevances.iter().copied(),
                ordering_scores.iter().copied(),
                100,
                pair_buffer,
                buffer,
            );
            assert!(approx_eq!(f32, res, 0.5098678822644145, ulps = 2));

            let relevances = &[-1., 7., -10., 3., 0., -6.];
            let ordering_scores = &[12., 9., 7., 5., 4., 1.];
            let res = calcuate_reordered_ndcg_at_k_score(
                relevances.iter().copied(),
                ordering_scores.iter().copied(),
                2,
                pair_buffer,
                buffer,
            );
            assert!(approx_eq!(f32, res, 0.6059214306390379, ulps = 2));

            let res = calcuate_reordered_ndcg_at_k_score(
                relevances.iter().copied(),
                ordering_scores.iter().copied(),
                4,
                pair_buffer,
                buffer,
            );
            assert!(approx_eq!(f32, res, 0.6260866644243038, ulps = 2));

            let res = calcuate_reordered_ndcg_at_k_score(
                relevances.iter().copied(),
                ordering_scores.iter().copied(),
                100,
                pair_buffer,
                buffer,
            );
            assert!(approx_eq!(f32, res, 0.6269342228326248, ulps = 2));
        }

        #[test]
        fn with_reordering() {
            let buffer = &mut Vec::with_capacity(6);
            let pair_buffer = &mut Vec::with_capacity(6);

            let relevances = &[4., 10., 6., 0., 3., 1.];
            let ordering_scores = &[9., 7., 1., 4., 5., 12.];
            let res = calcuate_reordered_ndcg_at_k_score(
                relevances.iter().copied(),
                ordering_scores.iter().copied(),
                2,
                pair_buffer,
                buffer,
            );
            assert!(approx_eq!(f32, res, 0.009846116527364958, ulps = 2));

            let res = calcuate_reordered_ndcg_at_k_score(
                relevances.iter().copied(),
                ordering_scores.iter().copied(),
                4,
                pair_buffer,
                buffer,
            );
            assert!(approx_eq!(f32, res, 0.4891424845441425, ulps = 2));

            let res = calcuate_reordered_ndcg_at_k_score(
                relevances.iter().copied(),
                ordering_scores.iter().copied(),
                100,
                pair_buffer,
                buffer,
            );
            assert!(approx_eq!(f32, res, 0.5098678822644145, ulps = 2));

            let relevances = &[3., -10., 0., -1., 7., -6.];
            let ordering_scores = &[5., 7., 4., 12., 9., 1.];
            let res = calcuate_reordered_ndcg_at_k_score(
                relevances.iter().copied(),
                ordering_scores.iter().copied(),
                2,
                pair_buffer,
                buffer,
            );
            assert!(approx_eq!(f32, res, 0.6059214306390379, ulps = 2));

            let res = calcuate_reordered_ndcg_at_k_score(
                relevances.iter().copied(),
                ordering_scores.iter().copied(),
                4,
                pair_buffer,
                buffer,
            );
            assert!(approx_eq!(f32, res, 0.6260866644243038, ulps = 2));

            let res = calcuate_reordered_ndcg_at_k_score(
                relevances.iter().copied(),
                ordering_scores.iter().copied(),
                100,
                pair_buffer,
                buffer,
            );
            assert!(approx_eq!(f32, res, 0.6269342228326248, ulps = 2));
        }

        #[test]
        fn buffers_can_be_any_capacity() {
            let buffer = &mut Vec::new();
            let pair_buffer = &mut Vec::new();

            let relevances = &[4., 10., 6., 0., 3., 1.];
            let ordering_scores = &[9., 7., 1., 4., 5., 12.];
            let res = calcuate_reordered_ndcg_at_k_score(
                relevances.iter().copied(),
                ordering_scores.iter().copied(),
                2,
                pair_buffer,
                buffer,
            );
            assert!(approx_eq!(f32, res, 0.009846116527364958, ulps = 2));
        }
    }

    mod ndcg_at_k {
        use super::super::*;
        use float_cmp::approx_eq;
        #[test]
        fn produces_expected_values_for_k_larger_then_input() {
            let res = ndcg_at_k(&[1., 4., 10., 3., 0., 6.], 100);
            assert!(approx_eq!(f32, res, 0.5098678822644145, ulps = 2));

            let res = ndcg_at_k(&[-1., 7., -10., 3., 0., -6.], 100);
            assert!(approx_eq!(f32, res, 0.6269342228326248, ulps = 2));
        }

        #[test]
        fn produces_expected_values_for_k_smaller_then_input() {
            let res = ndcg_at_k(&[1., 4., 10., 3., 0., 6.], 2);
            assert!(approx_eq!(f32, res, 0.009846116527364958, ulps = 2));
            let res = ndcg_at_k(&[1., 4., 10., 3., 0., 6.], 4);
            assert!(approx_eq!(f32, res, 0.4891424845441425, ulps = 2));

            let res = ndcg_at_k(&[-1., 7., -10., 3., 0., -6.], 2);
            assert!(approx_eq!(f32, res, 0.6059214306390379, ulps = 2));
            let res = ndcg_at_k(&[-1., 7., -10., 3., 0., -6.], 4);
            assert!(approx_eq!(f32, res, 0.6260866644243038, ulps = 2));
        }
    }

    mod dcg {
        use super::super::*;
        use float_cmp::approx_eq;

        #[test]
        fn running_it_results_in_expected_results() {
            //FIXME we should test for the result to be at most 1 float increament above/below the given value
            //      not that it's exact the same as "valid" changes in how we can do the calculation can lead to
            //      slightly different result due to rounding
            assert!(approx_eq!(
                f32,
                dcg(&[3f32, 2., 3., 0., 1., 2.]),
                13.848263629272981,
                ulps = 2
            ));
            assert!(approx_eq!(
                f32,
                dcg(&[-3.2, -2., -4., 0., -1., -2.]),
                -2.293710288714865,
                ulps = 2
            ));
        }
    }

    mod pick_k_highest_scores {
        use super::super::*;

        #[test]
        fn picks_the_highest_values_and_only_them() {
            let cases: &[(&[f32], &[f32])] = &[
                (&[3., 2., 1., 0.], &[3., 2.]),
                (&[0., 1., 2., 3.], &[3., 2.]),
                (&[-2., -2.], &[-2., -2.]),
                (&[-30., 3., 2., 10., -3., 0.], &[10., 3.]),
                (&[-3., 0., -1., -2.], &[0., -1.]),
            ];

            for (input, pick) in cases {
                let res = pick_k_highest_scores(input, 2);
                assert_eq!(
                    &*res, &**pick,
                    "res={:?}, expected={:?}, input={:?}",
                    res, pick, input
                );
            }
        }

        #[test]
        fn nans_are_preferably_not_picked_at_all() {
            let res = pick_k_highest_scores(&[3., 2., f32::NAN], 2);
            assert_eq!(&*res, &[3., 2.]);

            let res =
                pick_k_highest_scores(&[f32::NAN, 3., f32::NAN, f32::NAN, 2., 4., f32::NAN], 2);
            assert_eq!(&*res, &[4., 3.]);

            let res = pick_k_highest_scores(&[f32::NAN, 3., 2., f32::NAN], 3);
            assert_eq!(&res[..2], &[3., 2.]);
            assert!(res[2].is_nan());

            let res = pick_k_highest_scores(&[f32::NAN], 1);
            assert_eq!(res.len(), 1);
            assert!(res[0].is_nan());
        }
    }

    mod nan_safe_sort_desc_comparsion {
        use super::super::*;

        #[test]
        fn sorting_sorts_in_the_right_order() {
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
        fn nans_compare_as_expected() {
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
}
