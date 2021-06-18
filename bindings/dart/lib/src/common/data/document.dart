import 'package:json_annotation/json_annotation.dart' show JsonSerializable;
import 'package:meta/meta.dart' show immutable;

part 'document.g.dart';

/// The document.
@JsonSerializable()
@immutable
class Document {
  /// Unique identifier of the document.
  final String id;

  /// Text title of the document.
  final String title;

  /// Text snippet of the document.
  final String snippet;

  /// Position of the document from the source.
  final int rank;

  /// Session of the document.
  final String session;

  /// Query count within session.
  final int queryCount;

  /// Query identifier of the document.
  final String queryId;

  /// Query of the document.
  final String queryWords;

  /// URL of the document.
  final String url;

  /// Domain of the document
  final String domain;

  /// Creates the document.
  Document({
    required this.id,
    required this.title,
    required this.snippet,
    required this.rank,
    required this.session,
    required this.queryCount,
    required this.queryId,
    required this.queryWords,
    required this.url,
    required this.domain,
  }) {
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

  factory Document.fromJson(Map<String, dynamic> json) =>
      _$DocumentFromJson(json);

  Map<String, dynamic> toJson() => _$DocumentToJson(this);
}
