import 'package:flutter/material.dart' show debugPrint;

final _printChunkedRegex = RegExp(r'(.{0,80})');

/// `debugPrint` for long text.
///
/// Workaround for problems with string concatenation.
///
/// When using `print` or `debugPrint`, longer messages might
/// get truncated, at least when running it on an android
/// phone over adb using `flutter run`.
///
/// Using `debugPrint(test, wrapWidth: 1024)` is a workaround
/// which often works, but not always because:
///
/// - It uses full word line wrapping, and as such won't work
///   if there are any longer "words" (like base64 blobs).
///
/// - It seems (unclear) that the limit of `1024` might not
///   always be small enough.
///
/// So this is an "ad-hoc" workaround:
///
/// - We split the string into chunks of 80 characters, the
///   simplest way to do so is using the given regex.
///
/// - If this wasn't just some ad-hoc debug helper we probably
///   would implement splitting by at most 80 bytes at character
///   boundary.
///
/// - 80 is an arbitrarily chosen value which is not too large even
///   with unicode and works well with "small" terminals.
void debugPrintLongText(String text) {
  debugPrint('--- START Chunks ----');
  _printChunkedRegex
      .allMatches(text)
      .forEach((match) => debugPrint(match.group(0)!));
  debugPrint('--- END Chunks ----');
}
