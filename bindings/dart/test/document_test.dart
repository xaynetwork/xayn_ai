import 'package:flutter_test/flutter_test.dart'
    show equals, expect, group, test, throwsArgumentError;

import 'package:xayn_ai_ffi_dart/src/doc/document.dart' show Document, History;
import 'utils.dart' show documents, histories;

void main() {
  group('History', () {
    test('new', () {
      final hist = History(
          histories[0].id, histories[0].relevance, histories[0].feedback);
      expect(hist.id, equals(histories[0].id));
      expect(hist.relevance, equals(histories[0].relevance));
      expect(hist.feedback, equals(histories[0].feedback));
    });

    test('empty', () {
      expect(() => History('', histories[0].relevance, histories[0].feedback),
          throwsArgumentError);
    });
  });

  group('Document', () {
    test('new', () {
      final doc =
          Document(documents[0].id, documents[0].snippet, documents[0].rank);
      expect(doc.id, equals(documents[0].id));
      expect(doc.snippet, equals(documents[0].snippet));
      expect(doc.rank, equals(documents[0].rank));
    });

    test('empty', () {
      expect(() => Document('', documents[0].snippet, documents[0].rank),
          throwsArgumentError);
      expect(() => Document(documents[0].id, '', documents[0].rank),
          throwsArgumentError);
    });
  });
}
