import 'dart:typed_data' show Uint8List;

import 'package:flutter_test/flutter_test.dart'
    show contains, equals, expect, group, isEmpty, isNot, test;

import 'package:xayn_ai_ffi_dart/src/common/result/error.dart' show Code;
import 'package:xayn_ai_ffi_dart/src/common/reranker/ai.dart' show RerankMode;
import 'package:xayn_ai_ffi_dart/src/mobile/reranker/ai.dart' show XaynAi;
import '../utils.dart'
    show documents, histories, mkSetupData, throwsXaynAiException;

void main() {
  group('XaynAi', () {
    test('rerank full', () async {
      final ai = await XaynAi.create(mkSetupData());
      final outcome = ai.rerank(RerankMode.search, histories, documents);
      final faults = ai.faults();

      expect(outcome.finalRanks.length, equals(documents.length));
      documents.forEach(
          (document) => expect(outcome.finalRanks, contains(document.rank)));
      expect(faults, isNot(isEmpty));

      ai.free();
    });

    test('rerank empty', () async {
      final ai = await XaynAi.create(mkSetupData());
      final outcome = ai.rerank(RerankMode.search, [], []);
      final faults = ai.faults();

      expect(outcome.finalRanks, isEmpty);
      expect(faults, isNot(isEmpty));

      ai.free();
    });

    test('rerank empty hists', () async {
      final ai = await XaynAi.create(mkSetupData());
      final outcome = ai.rerank(RerankMode.search, [], documents);
      final faults = ai.faults();

      expect(outcome.finalRanks.length, equals(documents.length));
      documents.forEach(
          (document) => expect(outcome.finalRanks, contains(document.rank)));
      expect(faults, isNot(isEmpty));

      ai.free();
    });

    test('rerank empty docs', () async {
      final ai = await XaynAi.create(mkSetupData());
      final outcome = ai.rerank(RerankMode.search, histories, []);
      final faults = ai.faults();

      expect(outcome.finalRanks, isEmpty);
      expect(faults, isNot(isEmpty));

      ai.free();
    });

    test('invalid paths', () {
      expect(
        () async {
          var data = mkSetupData();
          data.smbertVocab = '';
          await XaynAi.create(data);
        },
        throwsXaynAiException(Code.readFile),
      );
      expect(
        () async {
          var data = mkSetupData();
          data.smbertModel = '';
          await XaynAi.create(data);
        },
        throwsXaynAiException(Code.readFile),
      );
      expect(
        () async {
          var data = mkSetupData();
          data.qambertVocab = '';
          await XaynAi.create(data);
        },
        throwsXaynAiException(Code.readFile),
      );
      expect(
        () async {
          var data = mkSetupData();
          data.qambertModel = '';
          await XaynAi.create(data);
        },
        throwsXaynAiException(Code.readFile),
      );
      expect(
        () async {
          var data = mkSetupData();
          data.ltrModel = '';
          await XaynAi.create(data);
        },
        throwsXaynAiException(Code.readFile),
      );
    });

    test('empty serialized', () async {
      final serialized = Uint8List(0);
      final ai = await XaynAi.create(mkSetupData(), serialized);
      ai.free();
    });

    test('invalid serialized', () {
      expect(
        () async =>
            await XaynAi.create(mkSetupData(), Uint8List.fromList([255])),
        throwsXaynAiException(Code.rerankerDeserialization),
      );
    });

    test('serialize syncdata', () async {
      final ai = await XaynAi.create(mkSetupData());

      expect(ai.syncdataBytes(), isNot(isEmpty));

      ai.free();
    });

    test('synchronize empty', () async {
      final ai = await XaynAi.create(mkSetupData());
      expect(
        () => ai.synchronize(Uint8List(0)),
        throwsXaynAiException(Code.synchronization),
      );

      ai.free();
    });

    test('synchronize invalid', () async {
      final ai = await XaynAi.create(mkSetupData());
      expect(
        () => ai.synchronize(Uint8List.fromList([255])),
        throwsXaynAiException(Code.synchronization),
      );

      ai.free();
    });
  });
}
