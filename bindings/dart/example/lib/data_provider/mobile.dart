import 'dart:io' show Directory, File;

import 'package:crypto/crypto.dart' show sha256;
import 'package:flutter/services.dart' show rootBundle;
import 'package:path_provider/path_provider.dart'
    show getApplicationDocumentsDirectory;
import 'package:xayn_ai_ffi_dart/package.dart'
    show AssetType, getAssets, SetupData;

import 'package:xayn_ai_ffi_dart_example/data_provider/data_provider.dart'
    show joinPaths;

const _baseAssetsPath = 'assets';

/// Prepares and returns the data that is needed to init [`XaynAi`].
///
/// This function needs to be called in the main thread because it will not be allowed
/// to access the assets from an isolate.
Future<SetupData> getInputData() async {
  final baseDiskPath = await getApplicationDocumentsDirectory();

  final paths = <AssetType, String>{};
  for (var asset in getAssets().entries) {
    final file = await _getData(baseDiskPath.path, asset.value.suffix);
    final checksum = asset.value.getChecksumAsHex();
    if (await _verifyChecksum(file, checksum)) {
      paths.putIfAbsent(asset.key, () => file.path);
    } else {
      throw 'checksum of ${file.path} does not match $checksum';
    }
  }

  return SetupData(paths[AssetType.vocab]!, paths[AssetType.smbert]!,
      paths[AssetType.qambert]!);
}

/// Returns the path to the data, if the data is not on disk yet
/// it will be copied from the bundle to the disk.
Future<File> _getData(
  String baseDiskPath,
  String assetSuffixPath,
) async {
  final assetPath = joinPaths([_baseAssetsPath, assetSuffixPath]);
  final data = await rootBundle.load(assetPath);

  final diskDirPath =
      joinPaths([baseDiskPath, File(assetSuffixPath).parent.path]);
  await Directory(diskDirPath).create(recursive: true);

  final diskPath = joinPaths([diskDirPath, _getFilename(assetSuffixPath)]);
  final file = File(diskPath);

  // Only write the data on disk if the file does not exist or the size does not match.
  // The last check is useful in case the app is closed before we can finish to write,
  // and it can be also useful during development to test with different models.
  if (!file.existsSync() || await file.length() != data.lengthInBytes) {
    final bytes =
        data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);
    await file.writeAsBytes(bytes, flush: true);
  }

  return file;
}

String _getFilename(String path) {
  return path.split('/').last;
}

Future<bool> _verifyChecksum(File file, String checksum) async {
  final digest = await sha256.bind(file.openRead()).first;
  return digest.toString() == checksum;
}
