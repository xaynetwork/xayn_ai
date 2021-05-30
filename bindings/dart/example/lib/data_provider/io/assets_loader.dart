import 'dart:io';

import 'package:crypto/crypto.dart';
import 'package:xayn_ai_ffi_dart/package.dart' show Asset, AssetType;
import 'package:tuple/tuple.dart' show Tuple2;

import 'package:xayn_ai_ffi_dart_example/data_provider/io/downloader.dart'
    show DownloadedAsset, Downloader, getFilename;

class AssetsLoader {
  late String _destinationDir;
  late Downloader _downloader;

  AssetsLoader(String destinationDir) {
    _destinationDir = destinationDir;
    _downloader = Downloader();
  }

  Future<Map<AssetType, String>> load(Map<AssetType, Asset> requested) async {
    final result = await _inspect(requested);
    final assets = await _downloadMissingAssets(result.item2);
    assets.addAll(result.item1);
    return assets;
  }

  Future<Map<AssetType, String>> _downloadMissingAssets(
      Map<AssetType, Asset> missing) async {
    final downloaded = await _downloader.download(missing);
    final assets = <AssetType, String>{};
    for (var completed in downloaded) {
      if (await _verifyChecksum(completed)) {
        final destinationPath =
            [_destinationDir, completed.filename].join(Platform.pathSeparator);
        print('move ${completed.filename} to $destinationPath');
        await _moveFile(completed.savePath!, destinationPath);
        assets.putIfAbsent(completed.type!, () => destinationPath);
      } else {
        print(
            'checksum of ${completed.filename} does not match. deleting file...');
        await completed.savePath!.delete();
        throw 'checksum failed';
      }
    }
    return assets;
  }

  Future<Tuple2<Map<AssetType, String>, Map<AssetType, Asset>>> _inspect(
      Map<AssetType, Asset> assets) async {
    final missing = <AssetType, Asset>{};
    final found = <AssetType, String>{};

    for (var asset in assets.entries) {
      final filename = getFilename(asset.value.url!);
      final path = [_destinationDir, filename].join(Platform.pathSeparator);
      if (await File(path).exists() == false) {
        print('asset $filename is missing');
        missing.putIfAbsent(asset.key, () => asset.value);
      } else {
        found.putIfAbsent(asset.key, () => path);
      }
    }

    return Tuple2(found, missing);
  }
}

// https://stackoverflow.com/a/55614133
Future<File> _moveFile(File sourceFile, String newPath) async {
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

Future<bool> _verifyChecksum(DownloadedAsset file) async {
  final digest = await sha256.bind(file.savePath!.openRead()).first;
  return digest.toString() == file.asset!.checksum;
}
