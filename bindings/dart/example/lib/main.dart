import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/material.dart'
    show
        AppBar,
        BuildContext,
        Center,
        Colors,
        Column,
        Container,
        Divider,
        EdgeInsets,
        ElevatedButton,
        Expanded,
        ListView,
        MaterialApp,
        MaterialPageRoute,
        Navigator,
        Padding,
        Row,
        Scaffold,
        Spacer,
        State,
        StatefulWidget,
        StatelessWidget,
        Text,
        TextButton,
        TextStyle,
        Widget,
        debugPrint,
        runApp;

import 'package:flutter/services.dart' show rootBundle;
import 'package:stats/stats.dart' show Stats;

import 'package:xayn_ai_ffi_dart/package.dart' show FeatureHint;
import 'package:xayn_ai_ffi_dart_example/debug/print.dart'
    if (dart.library.io) 'package:xayn_ai_ffi_dart_example/debug/mobile/print.dart'
    show debugPrintLongText;
import 'package:xayn_ai_ffi_dart_example/logic.dart' show Logic, Outcome;

void main() {
  runApp(MaterialApp(title: 'XaynAi Test/Example App', home: MyApp()));
}

class MyApp extends StatefulWidget {
  @override
  _MyAppState createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  // We cannot use `late`. If we did and loading it takes long enough
  // for the application to be rendered then this can cause an exception.
  Logic? _logic;
  Stats<num>? _lastBenchmarkStats;
  List<Outcome>? _lastResults;

  @override
  void initState() {
    super.initState();

    // mobile is always loaded with multi-threading features, web only optionally
    final hint = kIsWeb ? FeatureHint.wasmSequential : null;

    debugPrint('start loading assets');
    Logic.load(rootBundle, hint).then((logic) {
      debugPrint('loaded assets');
      // Calling `setState` after it was disposed is considered an error.
      if (mounted) {
        setState(() {
          _logic = logic;
        });
      }
    });
  }

  @override
  void dispose() {
    super.dispose();
    _logic?.free();
  }

  Widget currentCallDataView(BuildContext context) {
    final name = _logic!.currentCallDataKey.split('/').last;
    return Padding(
        padding: EdgeInsets.all(10),
        child: Row(
          children: [
            Text(name),
            Spacer(),
            ElevatedButton(
              onPressed: () {
                selectCallData(context);
              },
              child: Text('Change Call Data',
                  style: TextStyle(color: Colors.white)),
            )
          ],
        ));
  }

  Widget benchmarkView() {
    final stats = _lastBenchmarkStats;

    if (stats == null) {
      return Center(
        child: Text('-- no benchmark stats --'),
      );
    }

    final min = stats.min;
    final median = stats.median;
    final max = stats.max;
    final avg = stats.average;
    final std = stats.standardDeviation;

    return Text('Min: $min Median: $median Max: $max Avg: $avg Std: $std');
  }

  Widget resultsView() {
    final results = _lastResults;

    if (results == null || results.isEmpty) {
      return Center(
        child: Text('-- no results --'),
      );
    }

    return ListView.separated(
        padding: const EdgeInsets.all(8),
        itemCount: results.length,
        itemBuilder: (BuildContext context, int listIdx) {
          final outcome = results[listIdx];
          final rank = outcome.finalRank;
          final initialRank = outcome.document.rank;
          final contextValue = outcome.contextValue?.toStringAsFixed(3);
          final qaMBertSimilarity =
              outcome.qaMBertSimilarity?.toStringAsFixed(3);

          return Container(
              height: 50,
              child: Center(
                  child: Row(children: [
                Text(
                  '$rank',
                  style: TextStyle(fontSize: 40),
                ),
                const Spacer(),
                Text(
                    'Initial Rank: $initialRank\nContext Score: $contextValue\nQA-mBert Similarity: $qaMBertSimilarity'),
                const Spacer(),
              ])));
        },
        separatorBuilder: (BuildContext context, int index) => const Divider());
  }

  Widget appView(BuildContext context) {
    if (_logic == null) {
      return Center(
        child: Text('Loading XaynAi and assets.'),
      );
    }

    return Column(
      children: [
        currentCallDataView(context),
        ElevatedButton(
          onPressed: resetAi,
          child: Text('Reset AI', style: TextStyle(color: Colors.white)),
        ),
        ElevatedButton(
          onPressed: printCallDataWithCurrentState,
          child: Text('Print updated Call Data',
              style: TextStyle(color: Colors.white)),
        ),
        ElevatedButton(
          onPressed: benchRerank,
          child:
              Text('Benchmark Rerank', style: TextStyle(color: Colors.white)),
        ),
        benchmarkView(),
        ElevatedButton(
          onPressed: singleRerank,
          child:
              Text('Run Single Rerank', style: TextStyle(color: Colors.white)),
        ),
        Expanded(child: resultsView()),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('XainAi Test/Example App'),
      ),
      body: appView(context),
    );
  }

  void singleRerank() {
    final results = _logic!.run();
    setState(() {
      _lastResults = results;
    });
  }

  void benchRerank() {
    final stats = _logic!.benchmark();

    setState(() {
      _lastBenchmarkStats = stats;
    });
  }

  void resetAi() {
    _logic!.resetXaynAiState();
    setState(() {
      _lastBenchmarkStats = null;
      _lastResults = null;
    });
  }

  void printCallDataWithCurrentState() {
    final jsonString = _logic!.createUpdatedCallData().toJsonString();
    debugPrintLongText(jsonString);
  }

  Future<void> selectCallData(BuildContext context) async {
    print('start navigation');
    final newCallDataKey = await Navigator.push(
        context,
        MaterialPageRoute<String>(
          builder: (BuildContext context) =>
              SelectCallData(_logic!.availableCallDataKeys().toList()),
        ));

    if (newCallDataKey != null) {
      await _logic!.selectCallData(newCallDataKey);
      setState(() {
        _lastResults = null;
      });
    }
  }
}

class SelectCallData extends StatelessWidget {
  final List<String> availableData;

  SelectCallData(this.availableData);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
        appBar: AppBar(title: Text('Select Call Data')),
        body: ListView.separated(
            padding: const EdgeInsets.all(8),
            itemCount: availableData.length,
            itemBuilder: (BuildContext context, int listIdx) {
              final name = availableData[listIdx];
              return Container(
                  height: 50,
                  child: Center(
                    child: TextButton(
                        onPressed: () {
                          Navigator.pop(context, name);
                        },
                        child: Text(name)),
                  ));
            },
            separatorBuilder: (BuildContext context, int index) =>
                const Divider()));
  }
}
