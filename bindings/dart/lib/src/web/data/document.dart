@JS()
library document;

import 'package:js/js.dart' show anonymous, JS;

import 'package:xayn_ai_ffi_dart/src/common/data/document.dart' show Document;

@JS()
@anonymous
class JsDocument {
  external factory JsDocument({
    String id,
    int rank,
    String title,
    String session,
    // ignore: non_constant_identifier_names
    int query_count,
    // ignore: non_constant_identifier_names
    String query_id,
    // ignore: non_constant_identifier_names
    String query_words,
    String url,
    String domain,
  });
}

extension ToJsDocuments on List<Document> {
  /// Creates JS documents from the current documents.
  List<JsDocument> toJsDocuments() => List.generate(
        length,
        (i) => JsDocument(
          id: this[i].id,
          rank: this[i].rank,
          title: this[i].title,
          session: this[i].session,
          query_count: this[i].queryCount,
          query_id: this[i].queryId,
          query_words: this[i].queryWords,
          url: this[i].url,
          domain: this[i].domain,
        ),
        growable: false,
      );
}
