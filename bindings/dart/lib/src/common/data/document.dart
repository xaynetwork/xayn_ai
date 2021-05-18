import 'package:meta/meta.dart' show immutable;

/// The document.
@immutable
class Document {
  final String id;
  final String snippet;
  final int rank;

  /// Creates the document.
  Document(this.id, this.snippet, this.rank) {
    if (id.isEmpty) {
      throw ArgumentError('empty document id');
    }
    if (rank.isNegative) {
      throw ArgumentError('negative document rank');
    }
  }
}
