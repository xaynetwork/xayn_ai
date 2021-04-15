import 'dart:ffi' show AllocatorAlloc, nullptr, StructPointer, Uint32;

import 'package:ffi/ffi.dart' show malloc;
import 'package:flutter_test/flutter_test.dart'
    show equals, expect, group, isEmpty, test;

import 'package:xayn_ai_ffi_dart/src/doc/rank.dart' show Ranks;
import 'package:xayn_ai_ffi_dart/src/ffi/genesis.dart' show CRanks;

void main() {
  group('Ranks', () {
    test('to list', () {
      final len = 10;
      final ranksPtr = malloc.call<CRanks>();
      ranksPtr.ref.data = malloc.call<Uint32>(len);
      ranksPtr.ref.len = len;
      final ranks = Ranks(ranksPtr);
      expect(ranks.toList().length, equals(len));
      malloc.free(ranksPtr.ref.data);
      malloc.free(ranksPtr);
    });

    test('empty', () {
      final ranks = Ranks(nullptr);
      expect(ranks.toList(), isEmpty);
    });
  });
}
