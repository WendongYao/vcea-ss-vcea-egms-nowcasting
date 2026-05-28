from __future__ import annotations

from tgrs_experiments.pipeline import run_minimal_e0_to_e8


def main() -> None:
    outputs = run_minimal_e0_to_e8()
    print("实验完成，报告路径：")
    print(outputs["report"])


if __name__ == "__main__":
    main()
