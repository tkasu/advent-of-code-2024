import numpy as np
import pandas as pd
import torch

torch.autograd.set_grad_enabled(False)
device = torch.device("cpu")

FIND_STR = "XMAS"
DECODER_MAP = dict(enumerate(FIND_STR))
ENCODER_MAP = {v: k for k, v in DECODER_MAP.items()}
FIND_TENSOR = torch.tensor(list(DECODER_MAP.keys())).to(device)
FIND_TENSOR_REVERSED = torch.flip(FIND_TENSOR, [0])


def read_input(input_file: str) -> torch.Tensor:
    pd.set_option("future.no_silent_downcasting", True)

    data_pdf = pd.read_csv(
        input_file,
        sep=" ",
        header=None,
        engine="c",
    )
    char_pdf = data_pdf[0].str.split("", expand=True).iloc[:, 1:-1]
    num_matrix = char_pdf.replace(ENCODER_MAP).values.astype(np.uint8)

    tensor = torch.tensor(num_matrix).to(device)
    return tensor
