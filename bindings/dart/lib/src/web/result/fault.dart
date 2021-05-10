@JS()
library fault;

import 'package:js/js.dart' show anonymous, JS;

@JS()
@anonymous
class JsFault {
  external String get message;

  external factory JsFault({
    // ignore: unused_element
    int code,
    // ignore: unused_element
    String message,
  });
}

extension ToStrings on List<JsFault> {
  /// Gets the messages of the faults.
  List<String> toStrings() => List.generate(
        length,
        (i) => this[i].message,
        growable: false,
      );
}
