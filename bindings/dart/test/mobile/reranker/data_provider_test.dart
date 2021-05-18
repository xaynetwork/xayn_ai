import 'dart:io' show Directory;

import 'package:flutter_test/flutter_test.dart'
    show group, test, TestWidgetsFlutterBinding;

import 'package:xayn_ai_ffi_dart/src/mobile/reranker/ai.dart' show XaynAi;
import 'package:xayn_ai_ffi_dart/src/mobile/reranker/data_provider.dart'
    show SetupData;

void main() {
  group('SetupData', () {
    test('init from tmp dir', () async {
      // We need to access the assets in the boundle
      TestWidgetsFlutterBinding.ensureInitialized();
      final tmpDir = Directory.systemTemp.createTempSync('xayn_ai_assets_');
      final setupData = await SetupData.getInputData(tmpDir.path);
      final ai = XaynAi(setupData);
      ai.free();
    });
  });
}
