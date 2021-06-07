import 'dart:ffi' show AllocatorAlloc, nullptr, StructPointer, Uint8;

import 'package:ffi/ffi.dart' show malloc, StringUtf8Pointer, Utf8, Utf8Pointer;
import 'package:flutter_test/flutter_test.dart'
    show equals, expect, group, isNot, test;

import 'package:xayn_ai_ffi_dart/src/common/result/error.dart'
    show Code, CodeToInt, XaynAiException;
import 'package:xayn_ai_ffi_dart/src/mobile/ffi/genesis.dart'
    show CBoxedSlice_u8;
import 'package:xayn_ai_ffi_dart/src/mobile/result/error.dart' show XaynAiError;
import '../utils.dart' show throwsXaynAiException;

void main() {
  group('XaynAiError', () {
    test('fault', () {
      final error = XaynAiError();
      error.ptr.ref.code = Code.fault.toInt();
      error.ptr.ref.message = malloc.call<CBoxedSlice_u8>();
      error.ptr.ref.message.ref.data =
          'test fault'.toNativeUtf8().cast<Uint8>();
      error.ptr.ref.message.ref.len =
          error.ptr.ref.message.ref.data.cast<Utf8>().length + 1;

      expect(error.isFault(), equals(true));
      expect(error.isPanic(), equals(false));
      expect(error.isNone(), equals(false));
      expect(error.isError(), equals(false));

      malloc.free(error.ptr.ref.message.ref.data);
      malloc.free(error.ptr.ref.message);
      error.ptr.ref.message = nullptr;
      error.free();
    });

    test('panic', () {
      final error = XaynAiError();
      error.ptr.ref.code = Code.panic.toInt();
      error.ptr.ref.message = malloc.call<CBoxedSlice_u8>();
      error.ptr.ref.message.ref.data =
          'test panic'.toNativeUtf8().cast<Uint8>();
      error.ptr.ref.message.ref.len =
          error.ptr.ref.message.ref.data.cast<Utf8>().length + 1;

      expect(error.isFault(), equals(false));
      expect(error.isPanic(), equals(true));
      expect(error.isNone(), equals(false));
      expect(error.isError(), equals(true));

      malloc.free(error.ptr.ref.message.ref.data);
      malloc.free(error.ptr.ref.message);
      error.ptr.ref.message = nullptr;
      error.free();
    });

    test('none', () {
      final error = XaynAiError();

      expect(error.ptr, isNot(equals(nullptr)));
      expect(error.isFault(), equals(false));
      expect(error.isPanic(), equals(false));
      expect(error.isNone(), equals(true));
      expect(error.isError(), equals(false));

      error.free();
    });

    test('error', () {
      final error = XaynAiError();
      error.ptr.ref.code = Code.aiPointer.toInt();
      error.ptr.ref.message = malloc.call<CBoxedSlice_u8>();
      error.ptr.ref.message.ref.data =
          'test error'.toNativeUtf8().cast<Uint8>();
      error.ptr.ref.message.ref.len =
          error.ptr.ref.message.ref.data.cast<Utf8>().length + 1;

      expect(error.isFault(), equals(false));
      expect(error.isPanic(), equals(false));
      expect(error.isNone(), equals(false));
      expect(error.isError(), equals(true));

      malloc.free(error.ptr.ref.message.ref.data);
      malloc.free(error.ptr.ref.message);
      error.ptr.ref.message = nullptr;
      error.free();
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
      error.ptr.ref.message = malloc.call<CBoxedSlice_u8>();
      error.ptr.ref.message.ref.data = message.toNativeUtf8().cast<Uint8>();
      error.ptr.ref.message.ref.len =
          error.ptr.ref.message.ref.data.cast<Utf8>().length + 1;

      final exception = error.toException();
      expect(exception.code, equals(code));
      expect(exception.toString(), equals(message));
      expect(() => throw exception, throwsXaynAiException(code));

      malloc.free(error.ptr.ref.message.ref.data);
      malloc.free(error.ptr.ref.message);
      malloc.free(error.ptr);
    });
  });
}
