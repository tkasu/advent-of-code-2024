import numpy as np
import pandas as pd
import torch

torch.autograd.set_grad_enabled(False)

FIND_STR = "XMAS"
DECODER_MAP = dict(enumerate(FIND_STR, start=1))
ENCODER_MAP = {v: k for k, v in DECODER_MAP.items()}


def read_input(input_file: str, device: torch.device) -> torch.Tensor:
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
