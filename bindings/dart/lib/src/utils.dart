import 'package:flutter/foundation.dart';

/// Throws an assertion error in debug mode if the condition is false.
void debugAssert(bool cond) {
  if (!kReleaseMode) {
    if (!cond) {
      throw AssertionError('debug assertion failed');
    }
  }
}

/// Throws an assertion error in debug mode if the left-hand side is not equal to the right-hand
/// side.
void debugAssertEq<T>(T lhs, T rhs) {
  if (!kReleaseMode) {
    if (lhs != rhs) {
      throw AssertionError('debug assertion failed: $lhs != $rhs');
    }
  }
}

/// Throws an assertion error in debug mode if the left-hand side is equal to the right-hand side.
void debugAssertNeq<T>(T lhs, T rhs) {
  if (!kReleaseMode) {
    if (lhs == rhs) {
      throw AssertionError('debug assertion failed: $lhs == $rhs');
    }
  }
}
