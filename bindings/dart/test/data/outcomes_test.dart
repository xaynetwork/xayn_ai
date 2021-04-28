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
import 'dart:typed_data' show Float32List;

import 'package:ffi/ffi.dart' show calloc;
import 'package:flutter_test/flutter_test.dart'
    show equals, expect, group, test, throwsStateError;

import 'package:xayn_ai_ffi_dart/src/data/outcomes.dart'
    show RerankingOutcomes, RerankingOutcomesBuilder;
import 'package:xayn_ai_ffi_dart/src/ffi/genesis.dart' show CRerankingOutcomes;

class Delayed {
  final List<void Function()> _toClean;

  Delayed() : _toClean = List.empty(growable: true);

  /// Frees the memory (calloc) once cleanup is called.
  // I might seem that wrapping `alloc` would be a good idea, but it doesn't
  // work due to limitations of `malloc`/`calloc` `.call`/`.allocate`.
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
        // As we allocate this manually we also need to free it manually!!
        // we MUST NOT call `builder.free()` or any other function which
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
        outcomes.ref.qa_mbert_similarities.data = nullptr;
        outcomes.ref.qa_mbert_similarities.len = 0;

        final builder = RerankingOutcomesBuilder(outcomes);
        dartOutcomes = builder.build();
        //must not call: builder.free()
      } finally {
        delayed.runDelayed();
      }

      expect(dartOutcomes.finalRanks, equals([1, 0, 2]));

      expect(dartOutcomes.contextScores, equals([0.25, 0.5, 0.75]));

      expect(dartOutcomes.qaMBertSimilarities, equals(<Float32List>[]));
    });
  });
}
