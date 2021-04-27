import 'dart:ffi' show nullptr, StructPointer, Uint8;

import 'package:ffi/ffi.dart' show malloc, StringUtf8Pointer;
import 'package:flutter_test/flutter_test.dart'
    show equals, expect, group, isNot, test;

import 'package:xayn_ai_ffi_dart/src/result/error.dart'
    show Code, CodeInt, XaynAiError, XaynAiException;
import '../utils.dart' show throwsXaynAiException;

void main() {
  group('XaynAiError', () {
    test('fault', () {
      final error = XaynAiError();
      error.ptr.ref.code = Code.fault.toInt();
      error.ptr.ref.message = 'test fault'.toNativeUtf8().cast<Uint8>();

      expect(error.isFault(), equals(true));
      expect(error.isPanic(), equals(false));
      expect(error.isSuccess(), equals(false));
      expect(error.isError(), equals(false));

      malloc.free(error.ptr.ref.message);
      malloc.free(error.ptr);
    });

    test('panic', () {
      final error = XaynAiError();
      error.ptr.ref.code = Code.panic.toInt();
      error.ptr.ref.message = 'test panic'.toNativeUtf8().cast<Uint8>();

      expect(error.isFault(), equals(false));
      expect(error.isPanic(), equals(true));
      expect(error.isSuccess(), equals(false));
      expect(error.isError(), equals(true));

      malloc.free(error.ptr.ref.message);
      malloc.free(error.ptr);
    });

    test('success', () {
      final error = XaynAiError();

      expect(error.ptr, isNot(equals(nullptr)));
      expect(error.isFault(), equals(false));
      expect(error.isPanic(), equals(false));
      expect(error.isSuccess(), equals(true));
      expect(error.isError(), equals(false));

      malloc.free(error.ptr);
    });

    test('error', () {
      final error = XaynAiError();
      error.ptr.ref.code = Code.aiPointer.toInt();
      error.ptr.ref.message = 'test error'.toNativeUtf8().cast<Uint8>();

      expect(error.isFault(), equals(false));
      expect(error.isPanic(), equals(false));
      expect(error.isSuccess(), equals(false));
      expect(error.isError(), equals(true));

      malloc.free(error.ptr.ref.message);
      malloc.free(error.ptr);
    });

    test('free', () {
      final error = XaynAiError();

      expect(error.ptr, isNot(equals(nullptr)));
      error.free();
      expect(error.ptr, equals(nullptr));
    });
  });

  group('XaynAiException', () {
    test('new', () {
      final code = Code.panic;
      final message = 'test panic';

      final exception = XaynAiException(code, message);
      expect(exception.code, equals(code));
      expect(exception.toString(), equals(message));
      expect(() => throw exception, throwsXaynAiException(code));
    });

    test('to', () {
      final error = XaynAiError();
      final code = Code.panic;
      final message = 'test panic';
      error.ptr.ref.code = code.toInt();
      error.ptr.ref.message = message.toNativeUtf8().cast<Uint8>();

      final exception = error.toException();
      expect(exception.code, equals(code));
      expect(exception.toString(), equals(message));
      expect(() => throw exception, throwsXaynAiException(code));

      malloc.free(error.ptr.ref.message);
      malloc.free(error.ptr);
    });
  });
}
