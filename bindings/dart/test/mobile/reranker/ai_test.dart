import 'dart:typed_data' show Uint8List;

import 'package:flutter_test/flutter_test.dart'
    show
        contains,
        equals,
        expect,
        group,
        isEmpty,
        isNot,
        test,
        throwsA,
        TypeMatcher;

import 'package:xayn_ai_ffi_dart/src/common/reranker/ai.dart' show RerankMode;
import 'package:xayn_ai_ffi_dart/src/common/result/error.dart' show Code;
import 'package:xayn_ai_ffi_dart/src/mobile/reranker/ai.dart' show XaynAi;
import '../utils.dart'
    show documents, histories, mkSetupData, throwsXaynAiException;

void main() {
  group('XaynAi', () {
    test('rerank full', () async {
      final ai = await XaynAi.create(mkSetupData());
      final outcome =
          await ai.rerank(RerankMode.personalizedSearch, histories, documents);
      final faults = await ai.faults();

      expect(outcome.finalRanks.length, equals(documents.length));
      documents.forEach(
          (document) => expect(outcome.finalRanks, contains(document.rank)));
      expect(faults, isNot(isEmpty));

      await ai.free();
    });

    test('rerank empty', () async {
      final ai = await XaynAi.create(mkSetupData());
      final outcome = await ai.rerank(RerankMode.personalizedSearch, [], []);
      final faults = await ai.faults();

      expect(outcome.finalRanks, isEmpty);
      expect(faults, isEmpty);

      await ai.free();
    });

    test('rerank empty hists', () async {
      final ai = await XaynAi.create(mkSetupData());
      final outcome =
          await ai.rerank(RerankMode.personalizedSearch, [], documents);
      final faults = await ai.faults();

      expect(outcome.finalRanks.length, equals(documents.length));
      documents.forEach(
          (document) => expect(outcome.finalRanks, contains(document.rank)));
      expect(faults, isNot(isEmpty));

      await ai.free();
    });

    test('rerank empty docs', () async {
      final ai = await XaynAi.create(mkSetupData());
      final outcome =
          await ai.rerank(RerankMode.personalizedSearch, histories, []);
      final faults = await ai.faults();

      expect(outcome.finalRanks, isEmpty);
      expect(faults, isEmpty);

      await ai.free();
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
      expect(
        () async => await XaynAi.restore(mkSetupData(), Uint8List(0)),
        throwsA(TypeMatcher<ArgumentError>()),
      );
    });

    test('invalid serialized', () {
      expect(
        () async =>
            await XaynAi.restore(mkSetupData(), Uint8List.fromList([255])),
        throwsXaynAiException(Code.rerankerDeserialization),
      );
    });

    test('serialize syncdata', () async {
      final ai = await XaynAi.create(mkSetupData());

      expect(await ai.syncdataBytes(), isNot(isEmpty));

      await ai.free();
    });

    test('synchronize empty', () async {
      final ai = await XaynAi.create(mkSetupData());
      expect(
        () async => await ai.synchronize(Uint8List(0)),
        throwsXaynAiException(Code.synchronization),
      );

      await ai.free();
    });

    test('synchronize invalid', () async {
      final ai = await XaynAi.create(mkSetupData());
      expect(
        () async => await ai.synchronize(Uint8List.fromList([255])),
        throwsXaynAiException(Code.synchronization),
      );

      await ai.free();
    });
  });
}
