import 'dart:io';

import 'package:crypto/crypto.dart';
import 'package:xayn_ai_ffi_dart/package.dart' show assets, Asset;

import 'package:xayn_ai_ffi_dart_example/data_provider/io/downloader.dart'
    show DownloadedAsset, Downloader;

class AssertLoader {
  late String _destinationDir;
  late Downloader _downloader;

  AssertLoader(String destinationDir) {
    _destinationDir = destinationDir;
    _downloader = Downloader();
  }

  Future<List<Asset>> load() async {
    final toDownload = await _inspect(assets);

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
