import io
import timeit

import numpy as np
import pandas as pd
import torch

# INPUT_FILE = "input/day2_part1_sample.txt"
INPUT_FILE = "input/day2_part1.txt"

torch.autograd.set_grad_enabled(False)
device = torch.device("cpu")


def read_input() -> torch.Tensor:
    max_row_len = 0
    # As we are reading the file line by line already to count the max_row_len
    # we can just save the whole file as a csv string to avoid more file I/O
    csv_string = ""
    with open(INPUT_FILE) as file:
        while line := file.readline():
            csv_string += line
            row_len = line.count(" ") + 1
            max_row_len = max(row_len, max_row_len)

    data_pdf = pd.read_csv(
        io.StringIO(csv_string),
        sep=" ",
        header=None,
        names=[str(i) for i in range(max_row_len)],
        engine="c",
        dtype=np.float64,
        dtype_backend="numpy_nullable",
    )
    tensor = torch.tensor(data_pdf.values).to(device)
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


def create_combinations_with_drops(
    tensor_row: torch.Tensor, filter_placeholder
) -> torch.Tensor:
    """
    Create a matrix of from a signle dimensional tensor where dropped value is replaced with filter_placeholder.
    Dropped column is sequential, so column 0 is dropped for the first row, column 1 for the second row, etc.

    filter_placeholder is used to fill the dropped values of the matric. Beware that this value can't be NaN,
    as those are needed for padding the tensor.
    """
    tensor_len = tensor_row.shape[0]
    tensor_row = torch.clone(tensor_row)
    row_matrix = tensor_row.repeat(tensor_len, 1)

    # tensor_row_matrix = tensor_row_matrix.fill_diagonal_(filter_placeholder)
    # Custom implemention of fill_diagonal_, as fill_diagonal_ is not supported by vmap
    diag = torch.diag(torch.full((1, tensor_len), filter_placeholder).squeeze()).to(
        device
    )
    row_matrix_drop_filled = torch.where(diag == filter_placeholder, diag, row_matrix)
    return row_matrix_drop_filled


def part2(input_tensor: torch.Tensor) -> int:
    row_len = input_tensor[0].shape[0]
    drop_placeholder = np.inf

    rows_with_drops_tensor = torch.vmap(create_combinations_with_drops)(
        input_tensor, filter_placeholder=drop_placeholder
    )
    rows_with_drops_tensor = rows_with_drops_tensor[
        rows_with_drops_tensor != drop_placeholder
    ].reshape(-1, row_len, row_len - 1)

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
        lambda: part2(read_input()), number=benchmark_iterations
    )
    print(f"Part 2 benchmark: {benchmark_s / benchmark_iterations * 1000 :.2f} ms")


if __name__ == "__main__":
    run()
