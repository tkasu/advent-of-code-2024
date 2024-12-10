import numpy as np
import pandas as pd
import torch

# INPUT_FILE = "input/day2_part1_sample.txt"
INPUT_FILE = "input/day2_part1.txt"


def read_input() -> torch.Tensor:
    data_pdf = pd.read_fwf(INPUT_FILE, header=None)
    if (
        data_pdf.shape[1] == 1
    ):  # to handle cases where there columns with varying lengths
        data_pdf = data_pdf[0].str.split("\s+", expand=True)
    data_pdf = data_pdf.astype(
        "float64"
    )  # hack as there is no nullable interger tensors in PyTorch
    tensor = torch.tensor(data_pdf.values)
    return tensor


def part1(input_tensor: torch.Tensor) -> int:
    diffs = torch.diff(input_tensor, dim=1)
    diffs_all_inc_or_dec = diffs[
        torch.all(diffs.isnan() | (diffs > 0), dim=1)
        | torch.all(diffs.isnan() | (diffs < 0), dim=1)
    ]
    abs_diffs = torch.abs(diffs_all_inc_or_dec)
    gradual_shift = torch.all(abs_diffs.isnan() | (abs_diffs <= 3), dim=1)
    return gradual_shift.sum().item()


def part2(input_tensor: torch.Tensor) -> int:
    pass


def run():
    input_tensor = read_input()
    print(input_tensor)
    part1_solution = part1(input_tensor)
    print(f"Part 1 solution: {part1_solution}")

    input_tensor = read_input()
    part2_solution = part2(input_tensor)
    print(f"Part 2 solution: {part2_solution}")


if __name__ == "__main__":
    run()
