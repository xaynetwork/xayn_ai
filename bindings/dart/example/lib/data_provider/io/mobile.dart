import 'dart:io';

import 'package:async/async.dart' show AsyncMemoizer;
import 'package:crypto/crypto.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:path_provider/path_provider.dart';
import 'package:xayn_ai_ffi_dart/package.dart' as xayn_ai
    show assets, SetupData;

import 'package:xayn_ai_ffi_dart_example/data_provider/io/downloader.dart';

class SetupData implements xayn_ai.SetupData {
  final String vocab;
  final String model;

  static const _baseAssetsPath = 'packages/xayn_ai_ffi_dart/assets';
  static final AsyncMemoizer<SetupData> _pathsCache = AsyncMemoizer();

  SetupData(this.vocab, this.model);

  /// Prepares and returns the data that is needed to init [`XaynAi`].
  ///
  /// This function needs to be called in the main thread because it will not be allowed
  /// to access the assets from an isolate.
  ///
  /// [`baseDiskPath`] must be a path to a directory where it's possible to store the data.
  static Future<SetupData> getInputData() async {
    final saveDir = await getApplicationDocumentsDirectory();
    await AssertLoader(saveDir.path).load();
    return _pathsCache.runOnce(() async => _getInputData(saveDir.path));
  }

  // This is to avoid that two calls of this function can run concurrently.
  // Doing that can lead to invalid data on the disk.

  static Future<SetupData> _getInputData(String baseDiskPath) async {
    final rubertDir = 'rubert_v0001';
    final vocab = await _getData(baseDiskPath, rubertDir, 'vocab.txt');
    final model = await _getData(baseDiskPath, rubertDir, 'smbert.onnx');

    return SetupData(vocab, model);
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
