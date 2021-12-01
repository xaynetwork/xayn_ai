import 'dart:convert' show jsonDecode;
import 'dart:typed_data' show Uint8List;

import 'package:flutter/material.dart' show debugPrint;
import 'package:flutter/services.dart' show AssetBundle;
import 'package:stats/stats.dart' show Stats;
import 'package:xayn_ai_ffi_dart/package.dart'
    show
        Document,
        RerankDebugCallData,
        RerankMode,
        RerankingOutcomes,
        SetupData,
        XaynAi;

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

  Logic._(
    this._currentAi,
    this._setupData,
    this._availableCallData,
    this._currentCallDataKey,
  ) : _currentCallData = _availableCallData[_currentCallDataKey]!;

  String get currentCallDataKey => _currentCallDataKey;

  String get currentRerankMode =>
      _currentCallData.rerankMode.toString().split('.').last;

  /// Creates a call data instance with the histories and documents of the current
  /// call data and an updated serialized state.
  Future<RerankDebugCallData> createUpdatedCallData() async {
    final serializedState = await _currentAi.serialize();
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

    final currentAi = await initAi(
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

  Map<String, RerankMode> availableRerankModes() {
    return {
      for (final mode in RerankMode.values)
        mode.toString().split('.').last: mode
    };
  }

  Future<void> selectRerankMode(RerankMode mode) async {
    _currentCallData.rerankMode = mode;
  }

  Future<void> resetXaynAiState() async {
    await _currentAi.free();
    _currentAi = await initAi(_setupData, _currentCallData.serializedState);
  }

  Future<List<Outcome>> run() async {
    print('Starting Single Reranking');
    final results = await _currentAi.rerank(_currentCallData.rerankMode,
        _currentCallData.histories, _currentCallData.documents);
    await _printFaults();
    print('Finished Single Reranking');
    return Outcome.fromXaynAiOutcomes(_currentCallData.documents, results);
  }

  Future<void> _printFaults() async {
    final faults = await _currentAi.faults();
    faults.forEach((fault) => debugPrint('AI FAULT: $fault', wrapWidth: 1000));
  }

  static Future<XaynAi> initAi(SetupData setupData, Uint8List? serialized) {
    if (serialized == null) {
      return XaynAi.create(setupData);
    } else {
      return XaynAi.restore(setupData, serialized);
    }
  }

  Future<Stats> benchmark() async {
    print('Starting Benchmark');

    const preBenchNum = 10;
    const benchNum = 100;

    final mode = _currentCallData.rerankMode;
    final documents = _currentCallData.documents;
    final histories = _currentCallData.histories;

    // Init state with feedback loop
    await _currentAi.rerank(mode, histories, documents);
    await _currentAi.rerank(mode, histories, documents);

    print('Warming Up');
    // Make sure we run "hot" code to have less benchmark variety.
    // Though given the complexity of the computations and
    // given that it doesn't really involve JIT the main benefit
    // of this might be how it affects CPU frequencies.
    for (var i = 0; i < preBenchNum; i++) {
      await _currentAi.rerank(mode, histories, documents);
    }

    final times = List<num>.empty(growable: true);
    for (var i = 0; i < benchNum; i++) {
      final start = DateTime.now().millisecondsSinceEpoch;
      await _currentAi.rerank(mode, histories, documents);
      final end = DateTime.now().millisecondsSinceEpoch;
      times.add(end - start);

      await _printFaults();
      print('Iteration: $i');
    }

    print('Finished Benchmark');
    return Stats.fromData(times).withPrecision(1);
  }

  Future<void> free() async {
    await _currentAi.free();
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

enum _BenchmarkStatsKind {
  none,
  pending,
  ready,
}

class BenchmarkStats {
  final _BenchmarkStatsKind _kind;
  final Stats<num>? _ready;

  BenchmarkStats.none()
      : _kind = _BenchmarkStatsKind.none,
        _ready = null;

  BenchmarkStats.ready(this._ready) : _kind = _BenchmarkStatsKind.ready;

  @override
  String toString() {
    switch (_kind) {
      case _BenchmarkStatsKind.none:
        return '-- no benchmark stats --';
      case _BenchmarkStatsKind.pending:
        return '';
      case _BenchmarkStatsKind.ready:
        return 'Min: ${_ready!.min} Median: ${_ready!.median} Max: ${_ready!.max} Avg: ${_ready!.average} Std: ${_ready!.standardDeviation}';
    }
  }
}
