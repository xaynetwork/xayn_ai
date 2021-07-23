package com.xayn.xayn_ai_ffi_dart_example;

import androidx.test.rule.ActivityTestRule;
import dev.flutter.plugins.integration_test.FlutterTestRunner;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;


@RunWith(FlutterTestRunner.class)
public class MainActivityTest {
  @Rule
  public ActivityTestRule<MainActivity> rule = new ActivityTestRule<>(MainActivity.class, true, false);

  @Test
  public void escapeStaticAnalysisInAws() {
    // a dummy test so that aws thinks there is at least one test
    // without it aws will skip/cancel the run
  }
}
