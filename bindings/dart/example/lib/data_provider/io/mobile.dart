import 'dart:io';

import 'package:path_provider/path_provider.dart';
import 'package:xayn_ai_ffi_dart/package.dart' show SetupData;
import 'package:xayn_ai_ffi_dart_example/data_provider/io/asset_loader.dart'
    show AssertLoader;

/// Prepares and returns the data that is needed to init [`XaynAi`].
///
/// This function needs to be called in the main thread because it will not be allowed
/// to access the assets from an isolate.
///
/// [`baseDiskPath`] must be a path to a directory where it's possible to store the data.
Future<SetupData> getInputData() async {
  final baseDir = await getApplicationDocumentsDirectory();
  final rubertDir = 'ai_assets';
  final saveDir = _joinPaths([baseDir.path, rubertDir]);
  await Directory(saveDir).create(recursive: true);

  await AssertLoader(saveDir).load();

  final vocabPath = _joinPaths([saveDir, 'vocab.txt']);
  final modelPath = _joinPaths([saveDir, 'smbert.onnx']);
  return SetupData(vocabPath, modelPath);
}

String _joinPaths(List<String> paths) {
  return paths.where((e) => e.isNotEmpty).join('/');
}
