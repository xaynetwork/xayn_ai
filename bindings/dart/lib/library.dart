import 'dart:ffi' show DynamicLibrary;
import 'dart:io' show Platform;

import 'package:xayn_ai_ffi_dart/ffi.dart' show XaynAiFfi;

final XaynAiFfi ffi = XaynAiFfi(Platform.isAndroid
    ? DynamicLibrary.open('libxayn_ai_ffi_c.so')
    : Platform.isIOS
        ? DynamicLibrary.process()
        : Platform.isLinux
            ? DynamicLibrary.open('../../target/debug/libxayn_ai_ffi_c.so')
            : Platform.isMacOS
                ? DynamicLibrary.open(
                    '../../target/debug/libxayn_ai_ffi_c.dylib')
                : throw UnsupportedError('Unsupported platform.'));
