import pandas as pd
import torch

# INPUT_FILE = "input/day1_part1_sample.txt"
INPUT_FILE = "input/day1_part1.txt"


def read_input() -> torch.Tensor:
    data_pdf = pd.read_csv(INPUT_FILE, sep="\s+", header=None)
    tensor = torch.tensor(data_pdf.values)
    return tensor


def part1(input_tensor: torch.Tensor) -> int:
    first_column_sorted = torch.sort(input_tensor[:, 0]).values
    second_column_sorted = torch.sort(input_tensor[:, 1]).values
    distances = torch.abs(second_column_sorted - first_column_sorted)
    distance_sum = distances.sum()
    return distance_sum


def part2(input_tensor: torch.Tensor) -> int:
    first_column = input_tensor[:, 0]

    second_column_counts = torch.stack(
        torch.unique(input_tensor[:, 1], return_counts=True)
    )

    def get_scaling_factor(
        x: int,
    ) -> int:  # Not efficient, but could not figure out a vectorized operation (yet)
        count_tensor = second_column_counts[1][second_column_counts[0] == x]
        return count_tensor.item() if count_tensor.nelement() > 0 else 0

    scaling_factors = torch.clone(first_column).apply_(get_scaling_factor)
    return torch.sum(first_column * scaling_factors)


def run():
    input_tensor = read_input()
    part1_solution = part1(input_tensor)
    print(f"Part 1 solution: {part1_solution}")

    input_tensor = read_input()
    part2_solution = part2(input_tensor)
    print(f"Part 2 solution: {part2_solution}")


if __name__ == "__main__":
    run()
