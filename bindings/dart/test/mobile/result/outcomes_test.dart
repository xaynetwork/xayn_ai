import 'dart:ffi'
    show
        NativeType,
        Pointer,
        nullptr,
        AllocatorAlloc,
        StructPointer,
        Uint16,
        // ignore: unused_shown_name
        Uint16Pointer,
        Float,
        // ignore: unused_shown_name
        FloatPointer;

import 'package:ffi/ffi.dart' show calloc;
import 'package:flutter_test/flutter_test.dart'
    show
        equals,
        expect,
        group,
        isNull,
        test,
        throwsArgumentError,
        throwsStateError;

import 'package:xayn_ai_ffi_dart/src/common/result/outcomes.dart'
    show RerankingOutcomes;
import 'package:xayn_ai_ffi_dart/src/mobile/result/outcomes.dart'
    show RerankingOutcomesBuilder;
import 'package:xayn_ai_ffi_dart/src/mobile/ffi/genesis.dart'
    show CRerankingOutcomes;

class Delayed {
  final List<void Function()> _toClean;

  Delayed() : _toClean = List.empty(growable: true);

  // Frees the memory (calloc) once cleanup is called.
  // It might seem that wrapping `alloc` would be a good idea, but it doesn't
  // work due to limitations of `malloc`/`calloc` and `.call`/`.allocate`.
  void free<T extends NativeType>(Pointer<T> ptr) {
    _toClean.add(() {
      calloc.free(ptr);
    });
  }

  void runDelayed() {
    while (_toClean.isNotEmpty) {
      final last = _toClean.removeLast();
      last();
    }
  }
}

void main() {
  group('RerankingOutcomes', () {
    test('nullptr throws as this only happens if error codes where ignored',
        () {
      final builder = RerankingOutcomesBuilder(nullptr);
      expect(() {
        builder.build();
      }, throwsStateError);
    });

    test('use after free (of builder) throws', () {
      final builder = RerankingOutcomesBuilder(nullptr);
      builder.free();
      expect(() {
        builder.build();
      }, throwsStateError);
    });

    test('create from C repr', () {
      final RerankingOutcomes dartOutcomes;
      const numberOfDocs = 3;
      final delayed = Delayed();
      try {
        // As we allocate this manually we also need to free it manually!
        // We MUST NOT call `builder.free()` or any other function which
        // passes back ownership to rust (where it could be dropped).
        final outcomes = calloc.call<CRerankingOutcomes>();
        delayed.free(outcomes);

        // setup ranking
        outcomes.ref.final_ranking.data = calloc.call<Uint16>(numberOfDocs);
        delayed.free(outcomes.ref.final_ranking.data);
        outcomes.ref.final_ranking.len = numberOfDocs;
        outcomes.ref.final_ranking.data[0] = 1;
        outcomes.ref.final_ranking.data[1] = 0;
        outcomes.ref.final_ranking.data[2] = 2;

        // setup context
        outcomes.ref.context_scores.data = calloc.call<Float>(numberOfDocs);
        delayed.free(outcomes.ref.context_scores.data);
        outcomes.ref.context_scores.len = numberOfDocs;
        outcomes.ref.context_scores.data[0] = 0.25;
        outcomes.ref.context_scores.data[1] = 0.5;
        outcomes.ref.context_scores.data[2] = 0.75;

        // setup QA-mBERT (as not run)
        outcomes.ref.qambert_similarities.data = nullptr;
        outcomes.ref.qambert_similarities.len = 0;

        final builder = RerankingOutcomesBuilder(outcomes);
        dartOutcomes = builder.build();
        //We MUST NOT call: builder.free()
      } finally {
        delayed.runDelayed();
      }

      expect(dartOutcomes.finalRanks, equals([1, 0, 2]));

      expect(dartOutcomes.contextScores, equals([0.25, 0.5, 0.75]));

      expect(dartOutcomes.qaMBertSimilarities, isNull);
    });

    test('create from C repr with null rankings throw', () {
      final delayed = Delayed();
      try {
        // We MUST NOT call `builder.free()` or similar.
        final outcomes = calloc.call<CRerankingOutcomes>();
        delayed.free(outcomes);

        outcomes.ref.final_ranking.data = nullptr;
        outcomes.ref.final_ranking.len = 0;
        outcomes.ref.context_scores.data = nullptr;
        outcomes.ref.context_scores.len = 0;
        outcomes.ref.qambert_similarities.data = nullptr;
        outcomes.ref.qambert_similarities.len = 0;

        final builder = RerankingOutcomesBuilder(outcomes);
        expect(() {
          builder.build();
        }, throwsArgumentError);
      } finally {
        delayed.runDelayed();
      }
    });

    test('create from C repr with 0 rankings works', () {
      final RerankingOutcomes dartOutcomes;
      final delayed = Delayed();
      try {
        // We MUST NOT call `builder.free()` or similar.
        final outcomes = calloc.call<CRerankingOutcomes>();
        delayed.free(outcomes);

        // We need the equivalent of `NonNull::dangling()`,
        // which is the alginment as ptr address which here is 2.
        outcomes.ref.final_ranking.data = Pointer<Uint16>.fromAddress(2);
        // length needs to be 0
        outcomes.ref.final_ranking.len = 0;

        outcomes.ref.context_scores.data = nullptr;
        outcomes.ref.context_scores.len = 0;
        outcomes.ref.qambert_similarities.data = nullptr;
        outcomes.ref.qambert_similarities.len = 0;

        final builder = RerankingOutcomesBuilder(outcomes);
        dartOutcomes = builder.build();
      } finally {
        delayed.runDelayed();
      }

      expect(dartOutcomes.finalRanks.length, equals(0));
      expect(dartOutcomes.contextScores, isNull);
      expect(dartOutcomes.qaMBertSimilarities, isNull);
    });
  });
}
