from __future__ import annotations

from tgrs_experiments.supplements import run_mandatory_supplements


def main() -> None:
    outputs = run_mandatory_supplements()
    print("补充实验完成，报告路径：")
    print(outputs["report"])


if __name__ == "__main__":
    main()
