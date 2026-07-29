from pathlib import Path

import pandas as pd

from mi_proyecto_pipeline.pipeline import entrenar_y_evaluar


def main():
    data_path = Path(__file__).resolve().parent / "transactions_demo.csv"
    df = pd.read_csv(data_path)

    _, accuracy, report = entrenar_y_evaluar(df, target_column="is_large_transaction")

    print(f"Precisión obtenida en la ejecución externa: {accuracy:.2f}")
    print("\nReporte de clasificación:\n")
    print(report)


if __name__ == "__main__":
    main()
