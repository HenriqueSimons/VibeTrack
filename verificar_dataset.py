from pathlib import Path
import pandas as pd

caminho = Path("data") / "spotify-tracks-dataset-detailed.csv"

if not caminho.exists():
    print("ERRO: arquivo CSV não encontrado.")
    print("Caminho procurado:", caminho.resolve())
else:
    df = pd.read_csv(caminho)

    print("\n=== VIBETRACK: ANÁLISE INICIAL ===")
    print("\nQuantidade de linhas e colunas:")
    print(df.shape)

    print("\nNomes das colunas:")
    print(df.columns.tolist())

    print("\nPrimeiras 5 músicas:")
    print(df.head())

    print("\nTipos de dados:")
    print(df.dtypes)

    print("\nValores ausentes por coluna:")
    print(df.isnull().sum())

    print("\nQuantidade de linhas duplicadas:")
    print(df.duplicated().sum())
