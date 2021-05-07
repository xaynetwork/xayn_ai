/// The document.
class Document {
  final String _id;
  final String _snippet;
  final int _rank;

  /// Creates the document.
  Document(this._id, this._snippet, this._rank) {
    if (_id.isEmpty) {
      throw ArgumentError('empty document id');
    }
    if (_rank.isNegative) {
      throw ArgumentError('negative document rank');
    }
  }

  /// Gets the id.
  String get id => _id;

  /// Gets the snippet.
  String get snippet => _snippet;

  /// Gets the rank.
  int get rank => _rank;
}
