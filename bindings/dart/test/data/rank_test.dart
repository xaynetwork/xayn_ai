import 'dart:ffi'
    show
        AllocatorAlloc,
        nullptr,
        StructPointer,
        Uint32,
        // ignore: unused_shown_name
        Uint32Pointer;

import 'package:ffi/ffi.dart' show malloc;
import 'package:flutter_test/flutter_test.dart'
    show equals, expect, group, isEmpty, test;

import 'package:xayn_ai_ffi_dart/src/data/rank.dart' show Ranks;
import 'package:xayn_ai_ffi_dart/src/ffi/genesis.dart' show CBoxedSlice_u32;

void main() {
  group('Ranks', () {
    test('to list', () {
      final ranks = List.generate(10, (i) => i);
      final ranksPtr = malloc.call<CBoxedSlice_u32>();
      ranksPtr.ref.data = malloc.call<Uint32>(ranks.length);
      ranksPtr.ref.len = ranks.length;
      ranks.asMap().forEach((i, rank) => ranksPtr.ref.data[i] = rank);
      expect(Ranks(ranksPtr).toList(), equals(ranks));
      malloc.free(ranksPtr.ref.data);
      malloc.free(ranksPtr);
    });

    test('null', () {
      final ranks = Ranks(nullptr);
      expect(ranks.toList(), isEmpty);
    });

    test('empty', () {
      final ranksPtr = malloc.call<CBoxedSlice_u32>();
      ranksPtr.ref.data = nullptr;
      ranksPtr.ref.len = 0;
      expect(Ranks(ranksPtr).toList(), isEmpty);
      malloc.free(ranksPtr.ref.data);
      malloc.free(ranksPtr);
    });
  });
}
