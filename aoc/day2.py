import timeit

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
    diff_is_consistent_mask = torch.all(diffs.isnan() | (diffs > 0), dim=1) | torch.all(
        diffs.isnan() | (diffs < 0), dim=1
    )
    diff_abs_not_too_large_mask = torch.all(
        diffs.isnan() | (torch.abs(diffs) <= 3), dim=1
    )
    is_valid_mask = diff_is_consistent_mask & diff_abs_not_too_large_mask
    return is_valid_mask.sum().item()


def create_combinations_with_drops(tensor_row: torch.Tensor) -> torch.Tensor:
    """
    Create a matrix of from a signle dimensional tensor with one element dropped in each row".
    Dropped column is sequential, so column 0 is dropped for the first row, column 1 for the second row, etc.
    """
    filter_placeholder = (
        np.inf
    )  # Can't be NaN, as those are are needed for padding the tensor

    tensor_row_matrix = tensor_row.repeat(tensor_row.shape[0], 1)
    tensor_row_matrix = tensor_row_matrix.fill_diagonal_(filter_placeholder)
    tensor_row_with_drops = tensor_row_matrix[
        tensor_row_matrix != filter_placeholder
    ].reshape(tensor_row.shape[0], tensor_row.shape[0] - 1)
    return tensor_row_with_drops


def part2(input_tensor: torch.Tensor) -> int:
    rows_with_drops_tensor = torch.stack(
        [create_combinations_with_drops(row) for row in input_tensor]
    )
    diffs = torch.diff(rows_with_drops_tensor, dim=2)
    diff_is_consistent_mask = torch.all(diffs.isnan() | (diffs > 0), dim=2) | torch.all(
        diffs.isnan() | (diffs < 0), dim=2
    )
    diff_abs_not_too_large_mask = torch.all(
        diffs.isnan() | (torch.abs(diffs) <= 3), dim=2
    )
    is_valid_mask = diff_is_consistent_mask & diff_abs_not_too_large_mask
    is_valid_row = torch.any(is_valid_mask, dim=1)
    return is_valid_row.sum().item()


def run():
    input_tensor = read_input()
    part1_solution = part1(input_tensor)
    print(f"Part 1 solution: {part1_solution}")

    input_tensor = read_input()
    part2_solution = part2(input_tensor)
    print(f"Part 2 solution: {part2_solution}")

    # Benchmarking part 2
    print("Benchmarking...")
    benchmark_iterations = 100
    benchmark_s = timeit.timeit(
        lambda: part2(input_tensor), number=benchmark_iterations
    )
    print(f"Part 1 benchmark: {benchmark_s / benchmark_iterations * 1000 :.2f} ms")


if __name__ == "__main__":
    run()
