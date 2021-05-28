import 'dart:io';

import 'package:dio/dio.dart' show Dio, Response;
import 'package:tuple/tuple.dart' show Tuple2;
import 'package:path_provider/path_provider.dart'
    show getApplicationSupportDirectory;
import 'package:xayn_ai_ffi_dart_example/data_provider/io/mobile.dart' show Asset;

class DownloadedAsset {
  final Asset? asset;
  final File? savePath;

  DownloadedAsset({this.asset, this.savePath});
}

class _DownloadTask {
  final Asset? asset;
  final File? savePath;
  final Future<Response<dynamic>>? request;

  _DownloadTask({this.asset, this.savePath, this.request});
}

class Downloader {
  late Dio _dio;

  Downloader() {
    _dio = Dio();
  }

  Future<Tuple2<Iterable<DownloadedAsset>, Iterable<Asset>>> download(
      List<Asset> assets) async {
    final dir = await Downloader.getDownloadDirectory();

    final queue = assets.map((a) => () {
          final savePath = [dir.path, a.name!].join(Platform.pathSeparator);
          final request = _dio.download(a.url!, savePath);

          return _DownloadTask(
              asset: a, savePath: File(savePath), request: request);
        }());

    print('${queue.length} asset(s) in queue');

    // using await Future.wait(queue); does not work
    final failedAsset = <Asset>[];
    final downloadedAsset = <DownloadedAsset>[];
    for (var task in queue) {
      try {
        await task.request;
        final complete =
            DownloadedAsset(asset: task.asset, savePath: task.savePath);
        downloadedAsset.add(complete);
      } catch (e) {
        print('failed to download ${task.asset!.name} with error: $e');
        failedAsset.add(task.asset!);
      }
    }

    return Tuple2(downloadedAsset, failedAsset);
  }

  static Future<Directory> getDownloadDirectory() async {
    return await getApplicationSupportDirectory();
  }
}
