import 'package:flutter/material.dart' show debugPrint;

/// `debugPrint` for long text.
void debugPrintLongText(String text) {
  debugPrint(text, wrapWidth: 80);
}
