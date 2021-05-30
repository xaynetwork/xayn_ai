import 'dart:io';

import 'package:path_provider/path_provider.dart';
import 'package:xayn_ai_ffi_dart/package.dart'
    show assets, AssetType, SetupData;
import 'package:xayn_ai_ffi_dart_example/data_provider/io/assets_loader.dart'
    show AssetsLoader;

const assetsDir = 'ai_assets';

/// Prepares and returns the data that is needed to init [`XaynAi`].
///
/// This function needs to be called in the main thread because it will not be allowed
/// to access the assets from an isolate.
Future<SetupData> getInputData() async {
  final saveDir = await _createSaveDirectory();
  final aiAssets = await AssetsLoader(saveDir).load(assets);
  return SetupData(aiAssets[AssetType.vocab]!, aiAssets[AssetType.smbert]!);
}

Future<String> _createSaveDirectory() async {
  final baseDir = await getApplicationDocumentsDirectory();
  final saveDir = _joinPaths([baseDir.path, assetsDir]);
  await Directory(saveDir).create(recursive: true);
  return saveDir;
}

String _joinPaths(List<String> paths) {
  return paths.where((e) => e.isNotEmpty).join('/');
}
