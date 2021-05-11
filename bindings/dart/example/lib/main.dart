import 'dart:async';

import 'package:flutter/material.dart' hide Feedback;

import 'package:path_provider/path_provider.dart'
    show getApplicationDocumentsDirectory;
import 'package:stats/stats.dart' show Stats;

import 'package:xayn_ai_ffi_dart/package.dart'
    show XaynAi, Document, Relevance, History, Feedback;
import 'package:xayn_ai_ffi_dart/src/mobile/reranker/data_provider.dart'
    show SetupData;

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
    final data = await getApplicationDocumentsDirectory()
        .then((dir) => SetupData.getInputData(dir.path));

    // If the widget was removed from the tree while the asynchronous platform
    // message was in flight, we want to discard the reply rather than calling
    // setState to update our non-existent appearance.
    if (!mounted) return;

    setState(() {
      _ai = XaynAi(data);
      _msg = '';
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
              onPressed: benchRerank,
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

final histories = [
  History('00000000-0000-0000-0000-000000000000', Relevance.low,
      Feedback.irrelevant),
  History('00000000-0000-0000-0000-000000000001', Relevance.medium,
      Feedback.irrelevant),
  History('00000000-0000-0000-0000-000000000002', Relevance.high,
      Feedback.irrelevant),
  History('00000000-0000-0000-0000-000000000003', Relevance.low,
      Feedback.irrelevant),
  History('00000000-0000-0000-0000-000000000004', Relevance.medium,
      Feedback.irrelevant),
  History('00000000-0000-0000-0000-000000000005', Relevance.high,
      Feedback.irrelevant),
  History(
      '00000000-0000-0000-0000-000000000006', Relevance.low, Feedback.relevant),
  History('00000000-0000-0000-0000-000000000007', Relevance.medium,
      Feedback.relevant),
  History('00000000-0000-0000-0000-000000000008', Relevance.high,
      Feedback.relevant),
  History('00000000-0000-0000-0000-000000000009', Relevance.high,
      Feedback.relevant),
];

final documents = [
  Document('00000000-0000-0000-0000-000000000000', 'ship', 0),
  Document('00000000-0000-0000-0000-000000000001', 'car', 1),
  Document('00000000-0000-0000-0000-000000000002', 'auto', 2),
  Document('00000000-0000-0000-0000-000000000003', 'flugzeug', 3),
  Document('00000000-0000-0000-0000-000000000004', 'plane', 4),
  Document('00000000-0000-0000-0000-000000000005', 'vehicle', 5),
  Document('00000000-0000-0000-0000-000000000006', 'truck', 6),
  Document('00000000-0000-0000-0000-000000000007', 'trunk', 7),
  Document('00000000-0000-0000-0000-000000000008', 'motorbike', 8),
  Document('00000000-0000-0000-0000-000000000009', 'bicycle', 9),
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
