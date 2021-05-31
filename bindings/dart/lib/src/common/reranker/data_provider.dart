/// Data that can be used to initialize [`XaynAi`].

final assets = <AssetType, Asset>{};

enum AssetType { vocab, smbert, qambert, ltr, wasm }

class Asset {
  final String? url;
  final String? checksum;

  Asset({this.url, this.checksum});
}

class SetupData {
  // we define the constructor here to make the linter happy
  // but it would work without it
  SetupData(dynamic vocab, dynamic model, [dynamic wasm]);
}
