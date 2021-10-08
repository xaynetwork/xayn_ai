import * as Comlink from "comlink";


// We need to wrap the ES6 Proxy (return by comlink) in an extra class because it seems that dart
// can't handle Proxy's yet. When calling a method on the Proxy Dart recursively a function which 
// results in a stack overflow.
class WXaynAi {
  constructor(worker, proxy) {
    this._worker = worker;
    this._proxy = proxy;
  }

  async rerank(mode, histories, documents) {
    return await this._proxy.rerank(
      mode,
      // dart arrays are not cloneable and therefore sending them to the worker will fail.
      Array.from(histories),
      Array.from(documents)
    );
  }

  async serialize() {
    return await this._proxy.serialize();
  }

  async faults() {
    return await this._proxy.faults();
  }

  async analytics() {
    return await this._proxy.analytics();
  }

  async syncdata_bytes() {
    return await this._proxy.syncdata_bytes();
  }

  async synchronize(bytes) {
    await this._proxy.synchronize(bytes);
  }

  async free() {
    await this._proxy.free();
    this._worker.terminate();
  }
}

async function create(
  num_threads,
  smbert_vocab,
  smbert_model,
  qambert_vocab,
  qambert_model,
  ltr_model,
  serialized
) {
  const worker = new Worker(new URL("worker.js", import.meta.url), {
    type: "module",
  });
  const proxy = await Comlink.wrap(worker);
  const ai = await proxy.create(
    num_threads,
    smbert_vocab,
    smbert_model,
    qambert_vocab,
    qambert_model,
    ltr_model,
    serialized
  );
  return new WXaynAi(worker, ai);
}

export { create, WXaynAi };
