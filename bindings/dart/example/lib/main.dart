import 'package:flutter/material.dart'
    show
        AppBar,
        BuildContext,
        Center,
        Colors,
        Column,
        ElevatedButton,
        MaterialApp,
        runApp,
        Scaffold,
        State,
        StatefulWidget,
        Text,
        TextStyle,
        Widget;
import 'package:stats/stats.dart' show Stats;

import 'package:xayn_ai_ffi_dart/package.dart'
    show
        createXaynAi,
        DayOfWeek,
        Document,
        Feedback,
        History,
        Relevance,
        UserAction,
        XaynAi;

import 'package:xayn_ai_ffi_dart_example/data_provider/data_provider.dart'
    if (dart.library.io) 'data_provider/mobile.dart'
    if (dart.library.js) 'data_provider/web.dart' show getInputData;

void main() {
  runApp(MyApp());
}

class MyApp extends StatefulWidget {
  @override
  _MyAppState createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  late XaynAi _ai;
  String _msg = '';
  Function()? _onBechReady;

  @override
  void initState() {
    super.initState();
    initAi();
  }

  @override
  void dispose() {
    super.dispose();
    _ai.free();
  }

  Future<void> initAi() async {
    final data = await getInputData();

    // If the widget was removed from the tree while the asynchronous platform
    // message was in flight, we want to discard the reply rather than calling
    // setState to update our non-existent appearance.
    if (!mounted) return;

    final ai = await createXaynAi(data);
    setState(() {
      _ai = ai;
      _msg = '';
      _onBechReady = benchRerank;
    });
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(
          title: const Text('XainAi Test Bench'),
        ),
        body: Center(
            child: Column(
          children: [
            ElevatedButton(
              onPressed: _onBechReady,
              child: Text('Benchmark Rerank',
                  style: TextStyle(color: Colors.white)),
            ),
            Text(_msg),
          ],
        )),
      ),
    );
  }

  void benchRerank() {
    const preBenchNum = 10;
    const benchNum = 100;

    // Init state with feedback loop
    _ai.rerank(histories, documents);
    _ai.rerank(histories, documents);

    // ensure that code and data is in cache as much as possiible.
    for (var i = 0; i < preBenchNum; i++) {
      _ai.rerank(histories, documents);
    }

    final times = List<num>.empty(growable: true);
    for (var i = 0; i < benchNum; i++) {
      final start = DateTime.now().millisecondsSinceEpoch;
      _ai.rerank(histories, documents);
      final end = DateTime.now().millisecondsSinceEpoch;

      times.add(end - start);

      print(i);
    }

    setState(() {
      _msg = stats(times);
    });
  }
}

final sesh = '00000000-0000-0000-0000-000000000010';
final query = '00000000-0000-0000-0000-000000000011';

final histories = [
  History(
      '00000000-0000-0000-0000-000000000000',
      Relevance.low,
      Feedback.irrelevant,
      sesh,
      1,
      query,
      'transport',
      DayOfWeek.mon,
      'url',
      'dom',
      8,
      UserAction.miss),
  History(
      '00000000-0000-0000-0000-000000000001',
      Relevance.medium,
      Feedback.irrelevant,
      sesh,
      1,
      query,
      'transport',
      DayOfWeek.mon,
      'url',
      'dom',
      7,
      UserAction.skip),
  History(
      '00000000-0000-0000-0000-000000000002',
      Relevance.high,
      Feedback.irrelevant,
      sesh,
      1,
      query,
      'transport',
      DayOfWeek.mon,
      'url',
      'dom',
      6,
      UserAction.skip),
  History(
      '00000000-0000-0000-0000-000000000003',
      Relevance.low,
      Feedback.irrelevant,
      sesh,
      1,
      query,
      'transport',
      DayOfWeek.mon,
      'url',
      'dom',
      5,
      UserAction.skip),
  History(
      '00000000-0000-0000-0000-000000000004',
      Relevance.medium,
      Feedback.irrelevant,
      sesh,
      1,
      query,
      'transport',
      DayOfWeek.mon,
      'url',
      'dom',
      4,
      UserAction.skip),
  History(
      '00000000-0000-0000-0000-000000000005',
      Relevance.high,
      Feedback.irrelevant,
      sesh,
      1,
      query,
      'transport',
      DayOfWeek.mon,
      'url',
      'dom',
      3,
      UserAction.skip),
  History(
      '00000000-0000-0000-0000-000000000006',
      Relevance.low,
      Feedback.relevant,
      sesh,
      1,
      query,
      'transport',
      DayOfWeek.mon,
      'url',
      'dom',
      2,
      UserAction.skip),
  History(
      '00000000-0000-0000-0000-000000000007',
      Relevance.medium,
      Feedback.relevant,
      sesh,
      1,
      query,
      'transport',
      DayOfWeek.mon,
      'url',
      'dom',
      1,
      UserAction.skip),
  History(
      '00000000-0000-0000-0000-000000000008',
      Relevance.high,
      Feedback.relevant,
      sesh,
      1,
      query,
      'transport',
      DayOfWeek.mon,
      'url',
      'dom',
      0,
      UserAction.skip),
  History(
      '00000000-0000-0000-0000-000000000009',
      Relevance.high,
      Feedback.relevant,
      sesh,
      1,
      query,
      'transport',
      DayOfWeek.mon,
      'url',
      'dom',
      9,
      UserAction.skip),
];

final documents = [
  Document('00000000-0000-0000-0000-000000000000', 'ship', 0, sesh, 1, query,
      'transport', 'url', 'dom'),
  Document('00000000-0000-0000-0000-000000000001', 'car', 1, sesh, 1, query,
      'transport', 'url', 'dom'),
  Document('00000000-0000-0000-0000-000000000002', 'auto', 2, sesh, 1, query,
      'transport', 'url', 'dom'),
  Document('00000000-0000-0000-0000-000000000003', 'flugzeug', 3, sesh, 1,
      query, 'transport', 'url', 'dom'),
  Document('00000000-0000-0000-0000-000000000004', 'plane', 4, sesh, 1, query,
      'transport', 'url', 'dom'),
  Document('00000000-0000-0000-0000-000000000005', 'vehicle', 5, sesh, 1, query,
      'transport', 'url', 'dom'),
  Document('00000000-0000-0000-0000-000000000006', 'truck', 6, sesh, 1, query,
      'transport', 'url', 'dom'),
  Document('00000000-0000-0000-0000-000000000007', 'trunk', 7, sesh, 1, query,
      'transport', 'url', 'dom'),
  Document('00000000-0000-0000-0000-000000000008', 'motorbike', 8, sesh, 1,
      query, 'transport', 'url', 'dom'),
  Document('00000000-0000-0000-0000-000000000009', 'bicycle', 9, sesh, 1, query,
      'transport', 'url', 'dom'),
];

String stats(Iterable<num> times) {
  final stats = Stats.fromData(times).withPrecision(3);

  final avg = stats.average;
  final median = stats.median;
  final min = stats.min;
  final max = stats.max;
  final std = stats.standardDeviation;

  return 'Min: $min, Max: $max, Avg: $avg, Median: $median, Std: $std';
}
