
import pandas as pd
import joblib

# Carregar o dataset limpo
df = pd.read_csv("data/spotify_tracks_limpo.csv")

# Carregar o modelo treinado e o padronizador
pacote = joblib.load("models/modelo_vibetrack.joblib")

modelo = pacote["modelo"]
scaler = pacote["scaler"]
caracteristicas = pacote["caracteristicas"]

# Preparar os dados
X = df[caracteristicas]
X_scaled = scaler.transform(X)

# Identificar a qual grupo cada música pertence
df["grupo"] = modelo.predict(X_scaled)

print("=== ANÁLISE DOS GRUPOS DO VIBETRACK ===")

# Mostrar quantidade de músicas em cada grupo
print("\nQuantidade de músicas por grupo:")
print(df["grupo"].value_counts().sort_index())

# Calcular médias das características de cada grupo
medias = df.groupby("grupo")[caracteristicas].mean()

print("\n=== PERFIL MÉDIO DOS GRUPOS ===")
print(medias.round(3).to_string())

# Mostrar exemplos de músicas de cada grupo
for grupo in sorted(df["grupo"].unique()):
    print(f"\n=== GRUPO {grupo} ===")

    exemplos = df[df["grupo"] == grupo][
        ["track_name", "artists", "track_genre"]
    ].head(5)

    print(exemplos.to_string(index=False))

# Salvar dados com identificação dos grupos
df.to_csv("data/musicas_com_grupos.csv", index=False)

# Salvar estatísticas dos grupos
medias.to_csv("perfil_grupos.csv")

print("\nArquivos salvos com sucesso!")
print("data/musicas_com_grupos.csv")
print("perfil_grupos.csv")
