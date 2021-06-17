import 'package:flutter/material.dart' show debugPrint;

/// Debug prints a large string.
void debugPrintLargeText(String text) {
  debugPrint(text, wrapWidth: 80);
}
