"""
Call like `python3 extract_params.py model_in_path [params_out_dir]`.

See `xayn_ai/extract_listnet_parameters.py` for constraints.
"""

from pathlib import Path
import sys
from torch import load, Tensor
from typing import BinaryIO, Dict

CNN: Dict[str, str] = {
    "cnn2gram.cnn_list.0.weight": "conv_1/weights",
    "cnn2gram.cnn_list.0.bias": "conv_1/bias",
    "cnn2gram.cnn_list.1.weight": "conv_2/weights",
    "cnn2gram.cnn_list.1.bias": "conv_2/bias",
    "cnn2gram.cnn_list.2.weight": "conv_3/weights",
    "cnn2gram.cnn_list.2.bias": "conv_3/bias",
    "cnn2gram.cnn_list.3.weight": "conv_4/weights",
    "cnn2gram.cnn_list.3.bias": "conv_4/bias",
    "cnn2gram.cnn_list.4.weight": "conv_5/weights",
    "cnn2gram.cnn_list.4.bias": "conv_5/bias",
}

CLASSIFIER: Dict[str, str] = {
    "classifier.weight": "dense/weights",
    "classifier.bias": "dense/bias",
}

BYTE_ORDER: str = "little"

def write_integer(file: BinaryIO, integer: int):
    file.write(integer.to_bytes(8, BYTE_ORDER))

def write_string(file: BinaryIO, string: str):
    bytes = string.encode("utf-8")
    write_integer(file, len(bytes))
    file.write(bytes)

def write_tensor(file: BinaryIO, tensor: Tensor, transpose: bool):
    array = tensor.numpy()
    if transpose:
        array = array.transpose()
    array = array.astype(
        array.dtype.newbyteorder(BYTE_ORDER),
        order="C",
        casting="equiv",
        copy=False,
    )

    write_integer(file, len(array.shape))
    for value in array.shape:
        write_integer(file, value)

    write_integer(file, array.size)
    file.write(array.tobytes(order="C"))

def write_layers(file: BinaryIO, layers: Dict[str, str], state: Dict[str, Tensor], transpose: bool):
    write_integer(file, len(layers))
    for name, tensor in state.items():
        if name in layers:
            write_string(file, layers[name])
            write_tensor(file, tensor, transpose)

if __name__ == "__main__":
    model = Path(sys.argv[1]).resolve()
    if len(sys.argv) > 2:
        params = Path(sys.argv[2]).resolve()
    else:
        params = model.parent

    state = load(model)["state_dict"]
    with open(params.joinpath("cnn.binparams"), "wb") as cnn:
        write_layers(cnn, CNN, state, False)
    with open(params.joinpath("classifier.binparams"), "wb") as classifier:
        write_layers(classifier, CLASSIFIER, state, True)
