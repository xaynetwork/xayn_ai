import 'dart:io' show Directory, File, FileMode;

import 'package:crypto/crypto.dart' show sha256;
import 'package:flutter/services.dart' show rootBundle;
import 'package:path_provider/path_provider.dart'
    show getApplicationDocumentsDirectory;
import 'package:xayn_ai_ffi_dart/package.dart'
    show Asset, AssetType, getAssets, Fragment, SetupData;

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
    final path = await _getData(baseDiskPath.path, asset.value);
    paths.putIfAbsent(asset.key, () => path);
  }

  return SetupData(paths);
}

/// Returns the path to the data, if the data is not on disk yet
/// it will be copied from the bundle to the disk.
Future<String> _getData(String baseDiskPath, Asset asset) async {
  if (asset.fragments.isEmpty) {
    return await _copyAsset(baseDiskPath, asset);
  } else {
    return await _copyAssetFromFragments(baseDiskPath, asset);
  }
}

Future<String> _copyAsset(String baseDiskPath, Asset asset) async {
  final assetPath = joinPaths([_baseAssetsPath, asset.urlSuffix]);
  final data = await rootBundle.load(assetPath);

  final diskPath = File(joinPaths([baseDiskPath, asset.urlSuffix]));
  final diskDirPath = diskPath.parent.path;
  await Directory(diskDirPath).create(recursive: true);

  // Only write the data on disk if the file does not exist or the checksum does not match.
  // The last check is useful in case the app is closed before we can finish to write,
  // and it can be also useful during development to test with different models.
  if (!diskPath.existsSync() ||
      !await _verifyChecksum(diskPath, asset.checksum.checksumAsHex)) {
    final bytes = data.buffer.asUint8List(
      data.offsetInBytes,
      data.lengthInBytes,
    );
    await diskPath.writeAsBytes(bytes, flush: true);
  }
  return diskPath.path;
}

Future<String> _copyAssetFromFragments(String baseDiskPath, Asset asset) async {
  final diskPath = File(joinPaths([baseDiskPath, asset.urlSuffix]));
  final diskDirPath = diskPath.parent.path;
  await Directory(diskDirPath).create(recursive: true);

  if (diskPath.existsSync()) {
    if (!await _verifyChecksum(diskPath, asset.checksum.checksumAsHex)) {
      await diskPath.delete();
      await _copyFragments(diskPath, asset.fragments);
    }
  } else {
    await _copyFragments(diskPath, asset.fragments);
  }

  return diskPath.path;
}

Future<void> _copyFragments(File dest, List<Fragment> fragments) async {
  for (var fragment in fragments) {
    final fragmentPath = joinPaths([_baseAssetsPath, fragment.urlSuffix]);
    final data = await rootBundle.load(fragmentPath);
    final bytes = data.buffer.asUint8List(
      data.offsetInBytes,
      data.lengthInBytes,
    );
    await dest.writeAsBytes(bytes, mode: FileMode.append, flush: true);
  }
}

Future<bool> _verifyChecksum(File file, String checksum) async {
  final digest = await sha256.bind(file.openRead()).first;
  return digest.toString() == checksum;
}
