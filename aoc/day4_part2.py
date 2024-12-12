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


def match_sub_m(sub_m: torch.Tensor) -> bool:
    sub_m_mask = sub_m * TARGET_MASK
    for target_tensor in TARGET_TENSORS:
        if torch.all(sub_m_mask == target_tensor):
            return True
    return False


def part2(input_tensor: torch.Tensor) -> int:
    target_rectangle_n = TARGET_TENSOR.shape[0]

    count = 0
    for i in range(input_tensor.shape[0] - target_rectangle_n + 1):
        for j in range(input_tensor.shape[1] - target_rectangle_n + 1):
            sub_m = input_tensor[i : i + target_rectangle_n, j : j + target_rectangle_n]
            if match_sub_m(sub_m):
                count += 1

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
