/// Data that can be used to initialize [`XaynAi`].

final assets = <Asset>[];

class Asset {
  final String? name;
  final String? url;
  final String? checksum;

  Asset({this.name, this.url, this.checksum});
}

class SetupData {
  // we define the constructor here to make the linter happy
  // but it would work without it
  SetupData(dynamic vocab, dynamic model, [dynamic wasm]);
}
