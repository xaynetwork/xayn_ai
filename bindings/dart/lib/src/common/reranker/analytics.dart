class Analytics {
  /// The nDCG@k score between the LTR ranking and the relevance based ranking.
  final double ndcgLtr;

  /// The nDCG@k score between the Context ranking and the relevance based ranking.
  final double ndcgContext;

  /// The nDCG@k score between the initial ranking and the relevance based ranking.
  final double ndcgInitialRanking;

  /// The nDCG@k score between the final ranking and the relevance based ranking.
  final double ndcgFinalRanking;

  Analytics(
    this.ndcgLtr,
    this.ndcgContext,
    this.ndcgInitialRanking,
    this.ndcgFinalRanking,
  );
}
