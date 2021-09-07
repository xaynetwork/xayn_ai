import 'package:xayn_ai_ffi_dart/package.dart' show FeatureHint, SetupData;

/// Prepares and returns the data that is needed to init [`XaynAi`].
Future<SetupData> getInputData([FeatureHint? hint]) async =>
    throw UnsupportedError('Unsupported platform.');

String joinPaths(List<String> paths) {
  return paths.where((e) => e.isNotEmpty).join('/');
}
