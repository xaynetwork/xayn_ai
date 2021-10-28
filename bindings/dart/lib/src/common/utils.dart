abstract class ToJson {
  /// Serializes a dart object into a JSON object.
  Map<String, dynamic> toJson();
}

/// Throws an assertion error in debug mode if the left-hand side is not equal to the right-hand
/// side.
void assertEq<T>(T lhs, T rhs) {
  assert(lhs == rhs, 'equality assertion failed: $lhs != $rhs');
}

/// Throws an assertion error in debug mode if the left-hand side is equal to the right-hand side.
void assertNeq<T>(T lhs, T rhs) {
  assert(lhs != rhs, 'inequality assertion failed: $lhs == $rhs');
}
