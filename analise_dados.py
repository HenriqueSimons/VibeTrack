
from pathlib import Path
import pandas as pd

# Caminho do conjunto de dados
caminho = Path("data") / "spotify-tracks-dataset-detailed.csv"

# Características musicais que serão utilizadas no modelo
caracteristicas = [
    "danceability",
    "energy",
    "valence",
    "acousticness",
    "instrumentalness",
    "tempo"
]

# Carregar o dataset
df = pd.read_csv(caminho)

print("=== ANÁLISE E PREPARAÇÃO DOS DADOS ===")
print(f"\nRegistros originais: {len(df)}")

# Remover duplicatas
df = df.drop_duplicates().copy()
print(f"Registros após remover duplicatas: {len(df)}")

# Remover linhas sem nome de música ou artista
df = df.dropna(subset=["track_name", "artists"]).copy()
print(f"Registros após verificar nomes ausentes: {len(df)}")

# Garantir que as características selecionadas sejam numéricas
for coluna in caracteristicas:
    df[coluna] = pd.to_numeric(df[coluna], errors="coerce")

# Remover registros sem todas as características necessárias
df = df.dropna(subset=caracteristicas).copy()

print(f"Registros disponíveis para o modelo: {len(df)}")

print("\n=== ESTATÍSTICAS DAS CARACTERÍSTICAS ===")
print(df[caracteristicas].describe().round(3))

print("\n=== VERIFICAÇÃO DE VALORES ===")
print("Existem valores ausentes nas características?")
print(df[caracteristicas].isnull().sum())

print("\n=== EXEMPLOS DE MÚSICAS ===")
print(df[["artists", "track_name"] + caracteristicas].head())

# Salvar uma versão limpa para a próxima etapa
saida = Path("data") / "spotify_tracks_limpo.csv"
df.to_csv(saida, index=False)

print(f"\nDataset limpo salvo em: {saida}")
