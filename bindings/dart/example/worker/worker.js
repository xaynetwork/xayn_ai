import * as Comlink from "comlink";
import init, { initThreadPool, WXaynAi } from "./genesis.js";

async function create(
  num_threads,
  smbert_vocab,
  smbert_model,
  qambert_vocab,
  qambert_model,
  ltr_model,
  serialized
) {
  await init();
  await initThreadPool(num_threads);

  const xayn_ai = new WXaynAi(
    smbert_vocab,
    smbert_model,
    qambert_vocab,
    qambert_model,
    ltr_model,
    serialized
  );

  return Comlink.proxy(xayn_ai);
}

Comlink.expose({ create: create });
