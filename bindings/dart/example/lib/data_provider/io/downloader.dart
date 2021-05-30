import 'dart:io';

import 'package:dio/dio.dart' show Dio, Response;
import 'package:path_provider/path_provider.dart'
    show getApplicationSupportDirectory;
import 'package:xayn_ai_ffi_dart/package.dart' show Asset, AssetType;

class DownloadedAsset {
  final AssetType? type;
  final Asset? asset;
  final String? filename;
  final File? savePath;

  DownloadedAsset({this.type, this.asset, this.filename, this.savePath});
}

class _DownloadTask {
  final AssetType? type;
  final Asset? asset;
  final String? filename;
  final File? savePath;
  final Future<Response<dynamic>>? request;

  _DownloadTask(
      {this.type, this.asset, this.filename, this.savePath, this.request});
}

class Downloader {
  late Dio _dio;

  Downloader() {
    _dio = Dio();
  }

  Future<Iterable<DownloadedAsset>> download(
      Map<AssetType, Asset> assets) async {
    final dir = await Downloader.getDownloadDirectory();

    final queue = assets.entries.map((e) {
      final filename = getFilename(e.value.url!);
      final savePath = [dir.path, filename].join(Platform.pathSeparator);
      final request = _dio.download(e.value.url!, savePath);
      return _DownloadTask(
          type: e.key,
          asset: e.value,
          filename: filename,
          savePath: File(savePath),
          request: request);
    }).toList();

    print('${queue.length} asset(s) in queue');

    // using await Future.wait(queue); does not work
    final downloadedAsset = <DownloadedAsset>[];
    for (var task in queue) {
      try {
        await task.request;
        final complete = DownloadedAsset(
            type: task.type,
            asset: task.asset,
            filename: task.filename,
            savePath: task.savePath);
        downloadedAsset.add(complete);
      } catch (e) {
        print('failed to download ${task.asset!.url} with error: $e');
      }
    }

    return downloadedAsset;
  }

  static Future<Directory> getDownloadDirectory() async {
    return await getApplicationSupportDirectory();
  }
}

String getFilename(String url) {
  return url.split('/').last;
}
