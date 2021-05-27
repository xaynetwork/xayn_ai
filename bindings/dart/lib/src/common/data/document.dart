import 'package:meta/meta.dart' show immutable;

/// The document.
@immutable
class Document {
  final String id;
  final String snippet;
  final int rank;
  final String session;
  final int queryCount;
  final String queryId;
  final String queryWords;
  final String url;
  final String domain;

  /// Creates the document.
  Document(
    this.id,
    this.snippet,
    this.rank,
    this.session,
    this.queryCount,
    this.queryId,
    this.queryWords,
    this.url,
    this.domain,
  ) {
    if (id.isEmpty) {
      throw ArgumentError('empty document id');
    }
    if (rank.isNegative) {
      throw ArgumentError('negative document rank');
    }
    if (session.isEmpty) {
      throw ArgumentError('empty session id');
    }
    if (queryCount < 1) {
      throw ArgumentError('non-positive query count');
    }
    if (queryId.isEmpty) {
      throw ArgumentError('empty query id');
    }
    if (queryWords.isEmpty) {
      throw ArgumentError('empty query words');
    }
    if (url.isEmpty) {
      throw ArgumentError('empty document url');
    }
    if (domain.isEmpty) {
      throw ArgumentError('empty document domain');
    }
  }
}
