import * as Comlink from 'comlink';

async function init_ai() {
  let multiThread = await import("./pkg/xayn_ai_ffi_wasm.js");
  await multiThread.default();
  await multiThread.initThreadPool(navigator.hardwareConcurrency);

  console.log(navigator.hardwareConcurrency)

  console.time("load_data");
  let smbert_vocab = await fetch("../data/smbert_v0000/vocab.txt");
  let smbert_model = await fetch("../data/smbert_v0000/smbert.onnx");
  let qambert_vocab = await fetch("../data/qambert_v0001/vocab.txt");
  let qambert_model = await fetch("../data/qambert_v0001/qambert.onnx");
  let ltr_model = await fetch("../data/ltr_v0000/ltr.binparams");
  let smbert_vocab_buf = new Uint8Array(await smbert_vocab.arrayBuffer());
  let smbert_model_buf = new Uint8Array(await smbert_model.arrayBuffer());
  let qambert_vocab_buf = new Uint8Array(await qambert_vocab.arrayBuffer());
  let qambert_model_buf = new Uint8Array(await qambert_model.arrayBuffer());
  let ltr_model_buf = new Uint8Array(await ltr_model.arrayBuffer());
  console.timeEnd("load_data");

  let ai = new multiThread.WXaynAi(
    smbert_vocab_buf,
    smbert_model_buf,
    qambert_vocab_buf,
    qambert_model_buf,
    ltr_model_buf,
    undefined
  );

  return Comlink.proxy(ai);
}

Comlink.expose({
  handle: init_ai(),
});
