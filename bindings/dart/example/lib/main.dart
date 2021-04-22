import 'dart:async';

import 'package:flutter/material.dart';

import 'package:path_provider/path_provider.dart'
    show getApplicationDocumentsDirectory;
import 'package:xayn_ai_ffi_dart/package.dart' show XaynAi;

void main() {
  runApp(MyApp());
}

class MyApp extends StatefulWidget {
  @override
  _MyAppState createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  late XaynAi _ai;

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
    });
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(
          title: const Text('Plugin example app'),
        ),
        body: Center(
          child: Text('Running on:\n'),
        ),
      ),
    );
  }
}
