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

import 'package:xayn_ai_ffi_dart_example/debug/print.dart'
    if (dart.library.io) 'package:xayn_ai_ffi_dart_example/debug/mobile/print.dart'
    show debugPrintLongText;
import 'package:xayn_ai_ffi_dart_example/logic.dart'
    show BenchmarkStats, Logic, Outcome;

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
  BenchmarkStats _lastBenchmarkStats = BenchmarkStats.none();
  List<Outcome>? _lastResults;

  @override
  void initState() {
    super.initState();

    // mobile is always loaded with multi-threading features, web only optionally
    debugPrint('start loading assets');
    Logic.load(rootBundle).then((logic) {
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
  Future<void> dispose() async {
    super.dispose();
    await _logic?.free();
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

  Widget currentRerankModeView(BuildContext context) {
    return Padding(
        padding: EdgeInsets.all(10),
        child: Row(
          children: [
            Text(_logic!.currentRerankMode),
            Spacer(),
            ElevatedButton(
              onPressed: () {
                selectRerankMode(context);
              },
              child: Text('Change Rerank Mode',
                  style: TextStyle(color: Colors.white)),
            )
          ],
        ));
  }

  Widget benchmarkView() => Center(child: Text(_lastBenchmarkStats.toString()));

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
        currentRerankModeView(context),
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
        title: const Text('XaynAi Test/Example App'),
      ),
      body: appView(context),
    );
  }

  Future<void> singleRerank() async {
    final results = await _logic!.run();
    setState(() {
      _lastResults = results;
    });
  }

  Future<void> benchRerank() async {
    final stats = await _logic!.benchmark();

    setState(() {
      _lastBenchmarkStats = BenchmarkStats.ready(stats);
    });
  }

  Future<void> resetAi() async {
    await _logic!.resetXaynAiState();
    setState(() {
      _lastBenchmarkStats = BenchmarkStats.none();
      _lastResults = null;
    });
  }

  Future<void> printCallDataWithCurrentState() async {
    final callData = await _logic!.createUpdatedCallData();
    debugPrintLongText(callData.toJsonString());
  }

  Future<void> selectCallData(BuildContext context) async {
    final newCallDataKey = await Navigator.push(
        context,
        MaterialPageRoute<String>(
          builder: (BuildContext context) =>
              _SelectCallData(_logic!.availableCallDataKeys().toList()),
        ));

    if (newCallDataKey != null) {
      await _logic!.selectCallData(newCallDataKey);
      setState(() {
        _lastResults = null;
      });
    }
  }

  Future<void> selectRerankMode(BuildContext context) async {
    final availableRerankModes = _logic!.availableRerankModes();
    final newRerankMode = await Navigator.push(
        context,
        MaterialPageRoute<String>(
          builder: (BuildContext context) =>
              _SelectRerankMode(availableRerankModes.keys.toList()),
        ));

    if (newRerankMode != null) {
      await _logic!.selectRerankMode(availableRerankModes[newRerankMode]!);
      setState(() {
        _lastResults = null;
      });
    }
  }
}

class _SelectCallData extends StatelessWidget {
  final List<String> _availableData;

  _SelectCallData(this._availableData);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
        appBar: AppBar(title: Text('Select Call Data')),
        body: ListView.separated(
            padding: const EdgeInsets.all(8),
            itemCount: _availableData.length,
            itemBuilder: (BuildContext context, int listIdx) {
              final name = _availableData[listIdx];
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

class _SelectRerankMode extends StatelessWidget {
  final List<String> _availableModes;

  _SelectRerankMode(this._availableModes);

  @override
  Widget build(BuildContext context) => Scaffold(
      appBar: AppBar(title: Text('Select Rerank Mode')),
      body: ListView.separated(
          padding: const EdgeInsets.all(8),
          itemCount: _availableModes.length,
          itemBuilder: (BuildContext context, int listIdx) {
            final name = _availableModes[listIdx];
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
