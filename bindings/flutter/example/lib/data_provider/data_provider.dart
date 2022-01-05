import 'package:xayn_ai_ffi_flutter/package.dart' show SetupData;

/// Prepares and returns the data that is needed to init [`XaynAi`].
Future<SetupData> getInputData() async =>
    throw UnsupportedError('Unsupported platform.');

String joinPaths(List<String> paths) {
  return paths.where((e) => e.isNotEmpty).join('/');
}
