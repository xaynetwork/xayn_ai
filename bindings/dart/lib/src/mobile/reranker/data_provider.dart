import 'dart:io' show Directory, File;

import 'package:async/async.dart' show AsyncMemoizer;
import 'package:flutter/services.dart' show rootBundle;

import 'package:xayn_ai_ffi_dart/src/common/reranker/data_provider.dart'
    as common show SetupData;

/// Data that can be used to initialize [`XaynAi`].
class SetupData implements common.SetupData {
  final String vocab;
  final String smbertModel;
  final String qambertModel;

  static const _baseAssetsPath = 'packages/xayn_ai_ffi_dart/assets';
  static final AsyncMemoizer<SetupData> _pathsCache = AsyncMemoizer();

  SetupData(this.vocab, this.smbertModel, this.qambertModel);

  /// Prepares and returns the data that is needed to init [`XaynAi`].
  ///
  /// This function needs to be called in the main thread because it will not be allowed
  /// to access the assets from an isolate.
  ///
  /// [`baseDiskPath`] must be a path to a directory where it's possible to store the data.
  static Future<SetupData> getInputData(String baseDiskPath) async =>
      // This is to avoid that two calls of this function can run concurrently.
      // Doing that can lead to invalid data on the disk.
      _pathsCache.runOnce(() async => _getInputData(baseDiskPath));

  static Future<SetupData> _getInputData(String baseDiskPath) async {
    final smbertDir = 'smbert_v0000';
    final qambertDir = 'qambert_v0000';

    final vocab = await _getData(baseDiskPath, smbertDir, 'vocab.txt');
    final smbertModel = await _getData(baseDiskPath, smbertDir, 'smbert.onnx');
    final qambertModel =
        await _getData(baseDiskPath, qambertDir, 'qambert.onnx');

    return SetupData(vocab, smbertModel, qambertModel);
  }

  /// Returns the path to the data, if the data is not on disk yet
  /// it will be copied from the bundle to the disk.
  static Future<String> _getData(
    String baseDiskPath,
    String assetDirName,
    String assetName,
  ) async {
    final assetPath = _joinPaths([_baseAssetsPath, assetDirName, assetName]);
    final data = await rootBundle.load(assetPath);

    final diskDirPath = _joinPaths([baseDiskPath, assetDirName]);
    await Directory(diskDirPath).create(recursive: true);
    final diskPath = _joinPaths([diskDirPath, assetName]);
    final file = File(diskPath);

    // Only write the data on disk if the file does not exist or the size does not match.
    // The last check is useful in case the app is closed before we can finish to write,
    // and it can be also useful during development to test with different models.
    if (!file.existsSync() || await file.length() != data.lengthInBytes) {
      final bytes =
          data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);
      await file.writeAsBytes(bytes, flush: true);
    }

    return diskPath;
  }
}

String _joinPaths(List<String> paths) {
  return paths.where((e) => e.isNotEmpty).join('/');
}
