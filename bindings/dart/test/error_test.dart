import 'dart:ffi' show Int8, nullptr, StructPointer;

import 'package:ffi/ffi.dart' show malloc, StringUtf8Pointer;
import 'package:flutter_test/flutter_test.dart'
    show equals, expect, group, isNot, test;

import 'package:xayn_ai_ffi_dart/error.dart'
    show XaynAiCode, XaynAiCodeInt, XaynAiError, XaynAiException;
import 'utils.dart' show throwsXaynAiException;

void main() {
  group('XaynAiError', () {
    test('panic', () {
      final error = XaynAiError();
      error.ptr.ref.code = XaynAiCode.panic.toInt();
      error.ptr.ref.message = 'test panic'.toNativeUtf8().cast<Int8>();

      expect(error.isPanic(), equals(true));
      expect(error.isSuccess(), equals(false));
      expect(error.isError(), equals(false));

      malloc.free(error.ptr.ref.message);
      malloc.free(error.ptr);
    });

    test('success', () {
      final error = XaynAiError();

      expect(error.ptr, isNot(equals(nullptr)));
      expect(error.isPanic(), equals(false));
      expect(error.isSuccess(), equals(true));
      expect(error.isError(), equals(false));

      malloc.free(error.ptr);
    });

    test('error', () {
      final error = XaynAiError();
      error.ptr.ref.code = XaynAiCode.xaynAiPointer.toInt();
      error.ptr.ref.message = 'test error'.toNativeUtf8().cast<Int8>();

      expect(error.isPanic(), equals(false));
      expect(error.isSuccess(), equals(false));
      expect(error.isError(), equals(true));

      malloc.free(error.ptr.ref.message);
      malloc.free(error.ptr);
    });

    test('double free', () {
      final error = XaynAiError();
      error.free();
      error.free();
    });
  });

  group('XaynAiException', () {
    test('new', () {
      final code = XaynAiCode.panic;
      final message = 'test panic';

      expect(() => throw XaynAiException(code, message),
          throwsXaynAiException(code, message));
    });

    test('to', () {
      final error = XaynAiError();
      final code = XaynAiCode.panic;
      final message = 'test panic';
      error.ptr.ref.code = code.toInt();
      error.ptr.ref.message = message.toNativeUtf8().cast<Int8>();

      expect(() => throw error.toException(),
          throwsXaynAiException(code, message));

      malloc.free(error.ptr.ref.message);
      malloc.free(error.ptr);
    });
  });
}
