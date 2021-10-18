#
# To learn more about a Podspec see http://guides.cocoapods.org/syntax/podspec.html.
# Run `pod lib lint xayn_ai_ffi_dart.podspec` to validate before publishing.
#
Pod::Spec.new do |s|
  s.name             = 'xayn_ai_ffi_dart'
  s.version          = '0.0.1'
  s.summary          = 'XaynAI flutter plugin project.'
  s.description      = <<-DESC
XaynAI plugin project.
                       DESC
  s.homepage         = 'http://xayn.com'
  s.license          = { :file => '../../../LICENSE' }
  s.author           = { 'Xayn' => 'engineering@xaynet.dev' }
  s.source           = { :path => '.' }
  s.source_files = 'Classes/**/*'
  s.vendored_libraries = "**/*.a"
  s.dependency 'Flutter'
  s.platform = :ios, '9.0'

  # Flutter.framework does not contain a i386 slice.
  s.pod_target_xcconfig = { 'DEFINES_MODULE' => 'YES', 'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386' }
  s.swift_version = '5.0'
end
