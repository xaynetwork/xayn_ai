import 'dart:io' show Directory, File;

import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

import 'package:path_provider/path_provider.dart'
    show getApplicationDocumentsDirectory;
import 'package:flutter/services.dart' show rootBundle;

import 'package:xayn_ai_ffi_dart/package.dart'
    show
        AssetType,
        getAssets,
        SetupData,
        History,
        Document,
        Relevance,
        UserFeedback,
        UserAction,
        DayOfWeek,
        XaynAi,
        RerankMode;

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets("failing test example", (WidgetTester tester) async {
    expect(2 + 2, equals(5));
  });

  testWidgets("xayn rerank test", (WidgetTester tester) async {
    final ai = await XaynAi.create(await getInputData());
    final outcome = ai.rerank(RerankMode.search, histories, documents);
    final faults = ai.faults();

    expect(outcome.finalRanks.length, equals(documents.length));
    documents.forEach(
        (document) => expect(outcome.finalRanks, contains(document.rank)));
    expect(faults, isNot(isEmpty));
    ai.free();
  });
}

Document mkTestDoc(String id, String title, int rank) => Document(
      id: id,
      title: title,
      snippet: 'snippet of $title',
      rank: rank,
      session: 'fcb6a685-eb92-4d36-8686-000000000000',
      queryCount: 1,
      queryId: 'fcb6a685-eb92-4d36-8686-000000000000',
      queryWords: 'query words',
      url: 'url',
      domain: 'domain',
    );

History mkTestHist(String id, Relevance relevance, UserFeedback feedback) =>
    History(
      id: id,
      relevance: relevance,
      userFeedback: feedback,
      session: 'fcb6a685-eb92-4d36-8686-000000000000',
      queryCount: 1,
      queryId: 'fcb6a685-eb92-4d36-8686-000000000000',
      queryWords: 'query words',
      day: DayOfWeek.mon,
      url: 'url',
      domain: 'domain',
      rank: 0,
      userAction: UserAction.miss,
    );

final histories = [
  mkTestHist('fcb6a685-eb92-4d36-8686-000000000000', Relevance.low,
      UserFeedback.irrelevant),
  mkTestHist('fcb6a685-eb92-4d36-8686-000000000001', Relevance.high,
      UserFeedback.relevant),
];
final documents = [
  mkTestDoc('fcb6a685-eb92-4d36-8686-000000000000', 'abc', 0),
  mkTestDoc('fcb6a685-eb92-4d36-8686-000000000001', 'def', 1),
  mkTestDoc('fcb6a685-eb92-4d36-8686-000000000002', 'ghi', 2),
];

const _baseAssetsPath = 'assets';

/// Prepares and returns the data that is needed to init [`XaynAi`].
///
/// This function needs to be called in the main thread because it will not be allowed
/// to access the assets from an isolate.
Future<SetupData> getInputData() async {
  final baseDiskPath = await getApplicationDocumentsDirectory();

  final paths = <AssetType, String>{};
  for (var asset in getAssets().entries) {
    final path = await _getData(
        baseDiskPath.path, asset.value.suffix, asset.value.checksumAsHex);
    paths.putIfAbsent(asset.key, () => path);
  }

  return SetupData(paths);
}

/// Returns the path to the data, if the data is not on disk yet
/// it will be copied from the bundle to the disk.
Future<String> _getData(
  String baseDiskPath,
  String assetSuffixPath,
  String checksum,
) async {
  final assetPath = joinPaths([_baseAssetsPath, assetSuffixPath]);
  final data = await rootBundle.load(assetPath);

  final diskPath = File(joinPaths([baseDiskPath, assetSuffixPath]));
  final diskDirPath = diskPath.parent.path;
  await Directory(diskDirPath).create(recursive: true);
  final file = File(diskPath.path);

  // Only write the data on disk if the file does not exist or the size does not match.
  // The last check is useful in case the app is closed before we can finish to write,
  // and it can be also useful during development to test with different models.
  if (!file.existsSync() || await file.length() != data.lengthInBytes) {
    final bytes =
        data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);
    await file.writeAsBytes(bytes, flush: true);
  }

  return file.path;
}

String joinPaths(List<String> paths) {
  return paths.where((e) => e.isNotEmpty).join('/');
}
