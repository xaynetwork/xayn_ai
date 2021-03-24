import 'dart:ffi' show DynamicLibrary;
import 'dart:io' show Platform;

import 'package:xayn_ai_ffi_dart/ffi.dart' show XaynAiFfi;

extension XaynAiFfiImpl on XaynAiFfi {
  static DynamicLibrary load() {
    if (Platform.isAndroid) {
      return DynamicLibrary.open('libxayn_ai_ffi_c.so');
    }
    if (Platform.isIOS) {
      return DynamicLibrary.process();
    }
    if (Platform.isLinux) {
      return DynamicLibrary.open('../target/debug/libxayn_ai_ffi_c.so');
    }
    if (Platform.isMacOS) {
      return DynamicLibrary.open('../target/debug/libxayn_ai_ffi_c.dylib');
    }
    throw UnsupportedError('Unsupported platform.');
  }
}

final XaynAiFfi xaynAiFfi = XaynAiFfi(XaynAiFfiImpl.load());
