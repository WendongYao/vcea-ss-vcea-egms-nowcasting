from tgrs_experiments.staticval_ea_study import run_staticval_ea_study


if __name__ == "__main__":
    summary = run_staticval_ea_study()
    print(summary["report"])
