import 'dart:ffi' show DynamicLibrary;
import 'dart:io' show Platform;

import 'package:xayn_ai_ffi_dart/src/mobile/ffi/genesis.dart' show XaynAiFfi;

DynamicLibrary _open() {
  if (Platform.isAndroid) {
    return DynamicLibrary.open('libxayn_ai_ffi_c.so');
  }
  if (Platform.isIOS) {
    return DynamicLibrary.process();
  }
  if (Platform.isLinux) {
    return DynamicLibrary.open('../../target/debug/libxayn_ai_ffi_c.so');
  }
  if (Platform.isMacOS) {
    return DynamicLibrary.open('../../target/debug/libxayn_ai_ffi_c.dylib');
  }
  throw UnsupportedError('Unsupported platform.');
}

final ffi = XaynAiFfi(_open());
