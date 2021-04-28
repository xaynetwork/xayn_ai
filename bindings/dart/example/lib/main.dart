import 'dart:async';

import 'package:flutter/material.dart' hide Feedback;

import 'package:path_provider/path_provider.dart'
    show getApplicationDocumentsDirectory;
import 'package:stats/stats.dart' show Stats;
import 'package:xayn_ai_ffi_dart/package.dart'
    show XaynAi, Document, Relevance, History, Feedback;

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
        .then((dir) => XaynAi.inputData(dir.path));

    // If the widget was removed from the tree while the asynchronous platform
    // message was in flight, we want to discard the reply rather than calling
    // setState to update our non-existent appearance.
    if (!mounted) return;

    setState(() {
      _ai = XaynAi.fromInputData(data);
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
  History('0', Relevance.low, Feedback.irrelevant),
  History('1', Relevance.medium, Feedback.irrelevant),
  History('2', Relevance.high, Feedback.irrelevant),
  History('3', Relevance.low, Feedback.irrelevant),
  History('4', Relevance.medium, Feedback.irrelevant),
  History('5', Relevance.high, Feedback.irrelevant),
  History('6', Relevance.low, Feedback.relevant),
  History('7', Relevance.medium, Feedback.relevant),
  History('8', Relevance.high, Feedback.relevant),
  History('9', Relevance.high, Feedback.relevant),
];

final documents = [
  Document('0', 'ship', 0),
  Document('1', 'car', 1),
  Document('2', 'auto', 2),
  Document('3', 'flugzeug', 3),
  Document('4', 'plane', 4),
  Document('5', 'vehicle', 5),
  Document('6', 'truck', 6),
  Document('7', 'trunk', 7),
  Document('8', 'motorbike', 8),
  Document('9', 'bicycle', 9),
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
