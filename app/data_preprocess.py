import re
import pandas as pd

from app.config import (
    DATASET_PATH,
    DATASET_PROCESSED_PATH,
    MODELS_DIR,
)


def limpiar_texto(texto: str) -> str:
    if pd.isna(texto):
        return ""
    texto = str(texto).strip().lower()
    texto = texto.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    texto = re.sub(r"\s+", " ", texto)
    return texto


def preparar_dataset(df: pd.DataFrame) -> pd.DataFrame:
    columnas_requeridas = ["subject", "body", "language", "business_type"]
    faltantes = [col for col in columnas_requeridas if col not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas: {faltantes}")

    # Solo tickets en español
    df = df.copy()
    df["language"] = df["language"].astype(str).str.strip().str.lower()
    df = df[df["language"] == "es"].copy()

    # Rellenar nulos
    df["subject"] = df["subject"].fillna("")
    df["body"] = df["body"].fillna("")
    df["business_type"] = df["business_type"].fillna("")

    # Concatenar subject + body
    df["texto"] = (
        df["subject"].astype(str).apply(limpiar_texto)
        + " "
        + df["body"].astype(str).apply(limpiar_texto)
    )

    df["texto"] = df["texto"].apply(limpiar_texto)
    df["area"] = df["business_type"].astype(str).apply(limpiar_texto)

    # Eliminar vacíos
    df = df[(df["texto"] != "") & (df["area"] != "")].copy()

    # Eliminar duplicados exactos
    df = df.drop_duplicates(subset=["texto", "area"])

    return df[["texto", "area"]].copy()


def filtrar_clases_pocas_muestras(
    df: pd.DataFrame, min_muestras: int = 5
) -> pd.DataFrame:
    conteo = df["area"].value_counts()
    clases_validas = conteo[conteo >= min_muestras].index
    return df[df["area"].isin(clases_validas)].copy()


if __name__ == "__main__":
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Cargando dataset desde: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)

    print("Columnas detectadas:")
    print(df.columns.tolist())

    df = preparar_dataset(df)
    df = filtrar_clases_pocas_muestras(df, min_muestras=5)

    print("\nTotal de registros después de limpieza:", len(df))
    print("\nDistribución de clases:")
    print(df["area"].value_counts())

    if len(df) < 20:
        raise ValueError("Muy pocos datos tras la limpieza.")
    else:
        print("\nDataset preparado correctamente.")
        df.to_csv(DATASET_PROCESSED_PATH, index=False)
