import Flutter
import UIKit

public class SwiftXaynAiFfiDartPlugin: NSObject, FlutterPlugin {
  public static func register(with registrar: FlutterPluginRegistrar) {
    let channel = FlutterMethodChannel(name: "xayn_ai_ffi_dart", binaryMessenger: registrar.messenger())
    let instance = SwiftXaynAiFfiDartPlugin()
    registrar.addMethodCallDelegate(instance, channel: channel)
  }

  public func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
    result("iOS " + UIDevice.current.systemVersion)
  }
}
