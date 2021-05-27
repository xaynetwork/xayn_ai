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
      String snippet,
      String session,
      int queryCount,
      String queryId,
      String queryWords,
      String url,
      String domain
  });
}

extension ToJsDocuments on List<Document> {
  /// Creates JS documents from the current documents.
  List<JsDocument> toJsDocuments() => List.generate(
        length,
        (i) => JsDocument(
          id: this[i].id,
          rank: this[i].rank,
          snippet: this[i].snippet,
          session: this[i].session,
          queryCount: this[i].queryCount,
          queryId: this[i].queryId,
          queryWords: this[i].queryWords,
          url: this[i].url,
          domain: this[i].domain,
        ),
        growable: false,
      );
}
