import functools
import timeit

import torch

from aoc.day4_parsing import read_input, DECODER_MAP

# INPUT_FILE = "input/day4_part1_sample.txt"
INPUT_FILE = "input/day4_part1.txt"

torch.autograd.set_grad_enabled(False)
device = torch.device("cpu")

FIND_TENSOR = torch.tensor(list(DECODER_MAP.keys())).to(device)
FIND_TENSOR_REVERSED = torch.flip(FIND_TENSOR, [0])


def count_matching_tensors(
    input_tensor: torch.Tensor, find_tensor: torch.Tensor
) -> int:
    """
    Matches the last dimension of input_tensor with find_tensor, find tensor should be 1D.
    """
    match_mask = torch.all(
        input_tensor == find_tensor,
        dim=-1,
    )
    return match_mask.count_nonzero().item()


def diagonal_unfold(input_tensor: torch.Tensor, size: int, step: int) -> torch.Tensor:
    """
    Similar than torch.unfold, but for rolling diagonals.
    """
    diag_tensors = []
    for offset in range(-1 * input_tensor.shape[0], input_tensor.shape[1] + 1):
        diag_tensor = torch.diagonal(input_tensor, offset=offset)
        if diag_tensor.shape[0] >= size:
            diag_tensors.extend(diag_tensor.unfold(dimension=0, size=size, step=step))
    return torch.stack(diag_tensors)


def part1(input_tensor: torch.Tensor) -> int:
    horizontal_windows = input_tensor.unfold(dimension=1, size=4, step=1)
    vertical_windows = input_tensor.T.unfold(dimension=1, size=4, step=1)

    diag_horizontal_windows = diagonal_unfold(input_tensor, size=4, step=1)
    diag_vertical_windows = diagonal_unfold(input_tensor.rot90(), size=4, step=1)

    # For debugging
    counts = [
        count_matching_tensors(horizontal_windows, FIND_TENSOR),
        count_matching_tensors(horizontal_windows, FIND_TENSOR_REVERSED),
        count_matching_tensors(vertical_windows, FIND_TENSOR),
        count_matching_tensors(vertical_windows, FIND_TENSOR_REVERSED),
        count_matching_tensors(diag_horizontal_windows, FIND_TENSOR),
        count_matching_tensors(diag_horizontal_windows, FIND_TENSOR_REVERSED),
        count_matching_tensors(diag_vertical_windows, FIND_TENSOR),
        count_matching_tensors(diag_vertical_windows, FIND_TENSOR_REVERSED),
    ]
    # print(counts)

    count = functools.reduce(lambda x, y: x + y, counts)

    return count


def run():
    input_tensor = read_input(INPUT_FILE, device=device)
    part1_solution = part1(input_tensor)
    print(f"Part 1 solution: {part1_solution}")

    print("Benchmarking...")
    benchmark_iterations = 100
    benchmark_s = timeit.timeit(
        lambda: part1(read_input(INPUT_FILE, device=device)),
        number=benchmark_iterations,
    )
    print(f"Part 1 benchmark: {benchmark_s / benchmark_iterations * 1000 :.2f} ms")


if __name__ == "__main__":
    run()
