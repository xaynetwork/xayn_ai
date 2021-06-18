import 'dart:convert' show jsonDecode;

import 'package:flutter/services.dart' show AssetBundle;
import 'package:stats/stats.dart' show Stats;
import 'package:xayn_ai_ffi_dart/package.dart'
    show Document, RerankDebugCallData, RerankingOutcomes, SetupData, XaynAi;

import 'package:xayn_ai_ffi_dart_example/data_provider/data_provider.dart'
    if (dart.library.io) 'data_provider/mobile.dart'
    if (dart.library.js) 'data_provider/web.dart' show getInputData;

/// Class containing the business Logic of the the example app.
///
/// If this were a production app, this likely would be something like a
/// Provider/Reducer/Bloc potentially combined with some isolate indirections.
///
/// But for this example app this sync implementation is good enough.
class Logic {
  XaynAi _currentAi;
  final SetupData _setupData;
  final Map<String, RerankDebugCallData> _availableCallData;

  RerankDebugCallData _currentCallData;
  String _currentCallDataKey;

  Logic._(this._currentAi, this._setupData, this._availableCallData,
      this._currentCallDataKey)
      : _currentCallData = _availableCallData[_currentCallDataKey]!;

  String get currentCallDataKey => _currentCallDataKey;

  /// Creates a call data instance with the histories and documents of the current
  /// call data and an updated serialized state.
  RerankDebugCallData createUpdatedCallData() {
    final serializedState = _currentAi.serialize();
    return RerankDebugCallData(
      rerankMode: _currentCallData.rerankMode,
      histories: _currentCallData.histories,
      documents: _currentCallData.documents,
      serializedState: serializedState,
    );
  }

  /// Load the Logic instance from an asset bundle.
  ///
  /// This normally should be called with the `rootBundle`,
  /// as it expects an `AssetManifest.json` asset.
  ///
  static Future<Logic> load(AssetBundle bundle) async {
    final manifest = jsonDecode(await bundle.loadString('AssetManifest.json'))
        as Map<String, dynamic>;

    final availableCallData = <String, RerankDebugCallData>{
      for (final assetKey in manifest.keys)
        if (assetKey.startsWith('assets/call_data/') &&
            assetKey.endsWith('.json'))
          assetKey: await bundle.loadStructuredData(
              assetKey,
              (value) =>
                  Future.value(RerankDebugCallData.fromJsonString(value)))
    };

    var currentCallDataKey = 'example.json';
    if (!availableCallData.containsKey(currentCallDataKey)) {
      currentCallDataKey = availableCallData.keys.first;
    }

    final setupData = await getInputData();

    final currentAi = await XaynAi.create(
        setupData, availableCallData[currentCallDataKey]?.serializedState);

    return Logic._(currentAi, setupData, availableCallData, currentCallDataKey);
  }

  Iterable<String> availableCallDataKeys() {
    return _availableCallData.keys;
  }

  Future<void> selectCallData(String key) async {
    _currentCallData = _availableCallData[key]!;
    _currentCallDataKey = key;
    await resetXaynAiState();
  }

  Future<void> resetXaynAiState() async {
    _currentAi.free();
    _currentAi =
        await XaynAi.create(_setupData, _currentCallData.serializedState);
  }

  List<Outcome> run() {
    print('Starting Single Reranking');
    final results = _currentAi.rerank(_currentCallData.rerankMode,
        _currentCallData.histories, _currentCallData.documents);
    print('Finished Single Reranking');
    return Outcome.fromXaynAiOutcomes(_currentCallData.documents, results);
  }

  Stats benchmark() {
    print('Starting Benchmark');

    const preBenchNum = 10;
    const benchNum = 100;

    final mode = _currentCallData.rerankMode;
    final documents = _currentCallData.documents;
    final histories = _currentCallData.histories;

    // Init state with feedback loop
    _currentAi.rerank(mode, histories, documents);
    _currentAi.rerank(mode, histories, documents);

    print('Warming Up');
    // Make sure we run "hot" code to have less benchmark variety.
    // Though given the complexity of the computations and
    // given that it doesn't really involve JIT the main benefit
    // of this might be how it affects CPU frequencies.
    for (var i = 0; i < preBenchNum; i++) {
      _currentAi.rerank(mode, histories, documents);
    }

    final times = List<num>.empty(growable: true);
    for (var i = 0; i < benchNum; i++) {
      final start = DateTime.now().millisecondsSinceEpoch;
      _currentAi.rerank(mode, histories, documents);
      final end = DateTime.now().millisecondsSinceEpoch;

      times.add(end - start);

      print('Iteration: $i');
    }

    print('Finished Benchmark');
    return Stats.fromData(times).withPrecision(1);
  }

  void free() {
    _currentAi.free();
  }
}

class Outcome {
  final int finalRank;
  final double? contextValue;
  final double? qaMBertSimilarity;
  final Document document;

  Outcome(
      this.finalRank, this.contextValue, this.qaMBertSimilarity, this.document);

  static List<Outcome> fromXaynAiOutcomes(
      List<Document> inputs, RerankingOutcomes outcomes) {
    final resultList = inputs.asMap().entries.map((entry) {
      final finalRanking = outcomes.finalRanks[entry.key];
      final contextValue = outcomes.contextScores?[entry.key];
      final qaMBertSimilarity = outcomes.qaMBertSimilarities?[entry.key];
      return Outcome(
          finalRanking, contextValue, qaMBertSimilarity, entry.value);
    }).toList(growable: false);

    resultList.sort((l, r) => l.finalRank.compareTo(r.finalRank));
    return resultList;
  }
}
