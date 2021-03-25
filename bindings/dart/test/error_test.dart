import 'dart:ffi' show Int8, StructPointer;

import 'package:ffi/ffi.dart' show malloc, StringUtf8Pointer;
import 'package:flutter_test/flutter_test.dart'
    show equals, expect, group, test;

import 'package:xayn_ai_ffi_dart/error.dart' show XaynAiError;
import 'package:xayn_ai_ffi_dart/ffi.dart' show CXaynAiErrorCode;

void main() {
  group('XaynAiError', () {
    test('panic', () {
      final error = XaynAiError();
      final message = 'test panic';

      error.ptr.ref.code = CXaynAiErrorCode.Panic;
      error.ptr.ref.message = message.toNativeUtf8().cast<Int8>();

      expect(error.isPanic(), equals(true));
      expect(error.isSuccess(), equals(false));
      expect(error.isError(), equals(false));
      expect(error.toString(), equals('Panic: $message'));

      malloc.free(error.ptr.ref.message);
      malloc.free(error.ptr);
    });

    test('success', () {
      final error = XaynAiError();

      expect(error.isPanic(), equals(false));
      expect(error.isSuccess(), equals(true));
      expect(error.isError(), equals(false));
      expect(error.toString(), equals(''));

      malloc.free(error.ptr);
    });

    test('error', () {
      final error = XaynAiError();
      final message = 'test error';

      error.ptr.ref.code = CXaynAiErrorCode.XaynAiPointer;
      error.ptr.ref.message = message.toNativeUtf8().cast<Int8>();

      expect(error.isPanic(), equals(false));
      expect(error.isSuccess(), equals(false));
      expect(error.isError(), equals(true));
      expect(error.toString(), equals('XaynAiPointer: $message'));

      malloc.free(error.ptr.ref.message);
      malloc.free(error.ptr);
    });
  });
}
