import 'dart:io';

import 'package:async/async.dart' show AsyncMemoizer;
import 'package:crypto/crypto.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:path_provider/path_provider.dart';
import 'package:xayn_ai_ffi_dart/package.dart' as xayn_ai
    show assets, SetupData;

import 'package:xayn_ai_ffi_dart_example/data_provider/io/downloader.dart'
    show DownloadedAsset, Downloader;

class DataProvider {
  static const _baseAssetsPath = 'packages/xayn_ai_ffi_dart/assets';
  static final AsyncMemoizer<xayn_ai.SetupData> _pathsCache = AsyncMemoizer();

  /// Prepares and returns the data that is needed to init [`XaynAi`].
  ///
  /// This function needs to be called in the main thread because it will not be allowed
  /// to access the assets from an isolate.
  ///
  /// [`baseDiskPath`] must be a path to a directory where it's possible to store the data.
  static Future<xayn_ai.SetupData> getInputData() async {
    final saveDir = await getApplicationDocumentsDirectory();
    final rubertDir = 'rubert_v0001';
    final diskDirPath = _joinPaths([saveDir.path, rubertDir]);
    await Directory(diskDirPath).create(recursive: true);

    await AssertLoader(diskDirPath).load();
    return _pathsCache.runOnce(() async => _getInputData(diskDirPath));
  }

  // This is to avoid that two calls of this function can run concurrently.
  // Doing that can lead to invalid data on the disk.
  static Future<xayn_ai.SetupData> _getInputData(String baseDiskPath) async {
    final vocab = _joinPaths([baseDiskPath, 'vocab.txt']);
    final model = _joinPaths([baseDiskPath, 'smbert.onnx']);

    return xayn_ai.SetupData(vocab, model);
  }
}

String _joinPaths(List<String> paths) {
  return paths.where((e) => e.isNotEmpty).join('/');
}

class Asset {
  final String? name;
  final String? url;
  final String? checksum;

  Asset({this.name, this.url, this.checksum});
}

class AssertLoader {
  late String _destinationDir;
  late Downloader _downloader;

  AssertLoader(String destinationDir) {
    _destinationDir = destinationDir;
    _downloader = Downloader();
  }

  Future<List<Asset>> load() async {
    final requested = createAssets();
    final toDownload = await _inspect(requested);

    final result = await _downloader.download(toDownload);
    for (var completed in result.item1) {
      if (await verifyChecksum(completed)) {
        final destinationPath = [_destinationDir, completed.asset!.name]
            .join(Platform.pathSeparator);
        print('move ${completed.asset!.name} to $destinationPath');
        await moveFile(completed.savePath!, destinationPath);
      } else {
        print(
            'checksum of ${completed.asset!.name} does not match. deleting file...');
        try {
          await completed.savePath!.delete();
        } catch (e) {
          print(e);
        }
      }
    }

    return result.item2.toList();
  }

  Future<List<Asset>> _inspect(Iterable<Asset> requested) async {
    final notInSaveDir = <Asset>[];
    for (var asset in requested) {
      final path = [_destinationDir, asset.name].join(Platform.pathSeparator);
      if (await File(path).exists() == false) {
        print('asset ${asset.name} is missing');
        notInSaveDir.add(asset);
      }
    }

    return notInSaveDir;
  }
}

Iterable<Asset> createAssets() {
  return xayn_ai.assets.map(
      (a) => Asset(name: a['name'], checksum: a['checksum'], url: a['url']));
}

// https://stackoverflow.com/a/55614133
Future<File> moveFile(File sourceFile, String newPath) async {
  try {
    // prefer using rename as it is probably faster
    return await sourceFile.rename(newPath);
  } on FileSystemException catch (_) {
    // if rename fails, copy the source file and then delete it
    final newFile = await sourceFile.copy(newPath);
    await sourceFile.delete();
    return newFile;
  }
}

Future<bool> verifyChecksum(DownloadedAsset file) async {
  final digest = await sha256.bind(file.savePath!.openRead()).first;
  return digest.toString() == file.asset!.checksum;
}