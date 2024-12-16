import timeit

import torch

from aoc.day4_parsing import read_input, ENCODER_MAP

# INPUT_FILE = "input/day4_part1_sample.txt"
INPUT_FILE = "input/day4_part1.txt"

torch.autograd.set_grad_enabled(False)
device = torch.device("cpu")

ENCODER_WITH_EMPTY = ENCODER_MAP.copy()
ENCODER_WITH_EMPTY["."] = 0

TARGET_MATRIX = [
    ["M", ".", "S"],
    [".", "A", "."],
    ["M", ".", "S"],
]

TARGET_TENSOR = torch.tensor(
    [[ENCODER_WITH_EMPTY[c] for c in row] for row in TARGET_MATRIX]
).to(device)

TARGET_TENSORS = [
    TARGET_TENSOR,
    TARGET_TENSOR.rot90(),
    TARGET_TENSOR.rot90().rot90(),
    TARGET_TENSOR.rot90().rot90().rot90(),
]

TARGET_MASK = torch.tensor(
    [
        [1, 0, 1],
        [0, 1, 0],
        [1, 0, 1],
    ]
).to(device)


def unfold_subrectagles(input_m: torch.Tensor, rectangle_r: int) -> torch.Tensor:
    """
    Unfold [N, M] tensor to [B, rectangle_r, rectangle_r],
    where rectangles are all the posible sub-rectangles within input_m.
    """
    rectangle_start_indices = torch.tensor(
        [
            (i, j)
            for i in range(input_m.shape[0] - rectangle_r + 1)
            for j in range(input_m.shape[1] - rectangle_r + 1)
        ]
    )
    start_indices_rows = rectangle_start_indices[:, :1]
    start_indices_cols = rectangle_start_indices[:, 1:]

    base_m = (
        torch.arange(rectangle_r).repeat(rectangle_r).view(rectangle_r, rectangle_r)
    )
    base_m_rows = base_m.T
    base_m_cols = base_m

    row_indices = base_m_rows + start_indices_rows.expand(
        -1, rectangle_r * rectangle_r
    ).view(-1, rectangle_r, rectangle_r)
    col_indices = base_m_cols + start_indices_cols.expand(
        -1, rectangle_r * rectangle_r
    ).view(-1, rectangle_r, rectangle_r)
    return input_m[row_indices, col_indices]


def part2(input_tensor: torch.Tensor) -> int:
    target_rectangle_n = TARGET_TENSOR.shape[0]

    rectangles = unfold_subrectagles(input_tensor, target_rectangle_n) * TARGET_MASK

    count = 0
    for target_tensor in TARGET_TENSORS:
        count += torch.all(rectangles == target_tensor, dim=(1, 2)).sum().item()

    return count


def run():
    input_tensor = read_input(INPUT_FILE, device=device)
    part1_solution = part2(input_tensor)
    print(f"Part 2 solution: {part1_solution}")

    print("Benchmarking...")
    benchmark_iterations = 100
    benchmark_s = timeit.timeit(
        lambda: part2(read_input(INPUT_FILE, device=device)),
        number=benchmark_iterations,
    )
    print(f"Part 2 benchmark: {benchmark_s / benchmark_iterations * 1000 :.2f} ms")


if __name__ == "__main__":
    run()
