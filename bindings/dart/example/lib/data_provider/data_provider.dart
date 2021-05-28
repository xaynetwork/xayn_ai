import 'package:xayn_ai_ffi_dart/package.dart' show SetupData;

class DataProvider {
  /// Prepares and returns the data that is needed to init [`XaynAi`].
  static Future<SetupData> getInputData() async =>
      throw UnsupportedError('Unsupported platform.');
}
