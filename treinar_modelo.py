
from pathlib import Path
import time
import joblib
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Caminhos
caminho_dados = Path("data") / "spotify_tracks_limpo.csv"
pasta_modelos = Path("models")
pasta_modelos.mkdir(exist_ok=True)

# Características musicais usadas pelo modelo
caracteristicas = [
    "danceability",
    "energy",
    "valence",
    "acousticness",
    "instrumentalness",
    "tempo"
]

# Carregar os dados limpos
df = pd.read_csv(caminho_dados)

# Selecionar somente as características numéricas
X = df[caracteristicas].copy()

print("=== TREINAMENTO DO VIBETRACK ===")
print(f"Registros: {len(X)}")
print(f"Características: {caracteristicas}")

# Padronizar as variáveis para equilibrar suas escalas
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Usar uma amostra fixa para comparar o silhouette score.
# Isso reduz o tempo de cálculo com mais de 100 mil registros.
amostra_tamanho = min(10000, len(X_scaled))

rng = np.random.default_rng(42)
indices_amostra = rng.choice(
    len(X_scaled),
    size=amostra_tamanho,
    replace=False
)

X_amostra = X_scaled[indices_amostra]

resultados = []
melhor_k = None
melhor_score = float("-inf")
melhor_modelo = None

for k in [4, 6, 8]:
    print(f"\nTreinando K-means com k={k}...")
    inicio = time.time()

    modelo = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    modelo.fit(X_scaled)

    # Avaliar a separação dos grupos na mesma amostra
    rotulos_amostra = modelo.predict(X_amostra)
    score = silhouette_score(
        X_amostra,
        rotulos_amostra
    )

    # Distribuição dos registros pelos grupos
    tamanhos = pd.Series(modelo.labels_).value_counts().sort_index()

    tempo_treino = time.time() - inicio

    resultados.append({
        "k": k,
        "silhouette_score": score,
        "tempo_treino_segundos": tempo_treino,
        "menor_grupo": int(tamanhos.min()),
        "maior_grupo": int(tamanhos.max())
    })

    print(f"Silhouette score: {score:.4f}")
    print(f"Tempo de treinamento: {tempo_treino:.1f} segundos")
    print("Quantidade de músicas por grupo:")
    print(tamanhos.sort_index().to_dict())

    if score > melhor_score:
        melhor_score = score
        melhor_k = k
        melhor_modelo = modelo

# Exibir comparação
df_resultados = pd.DataFrame(resultados)
print("\n=== COMPARAÇÃO DOS MODELOS ===")
print(df_resultados.round(4).to_string(index=False))

# Salvar resultados da comparação
df_resultados.to_csv(
    "resultados_modelos.csv",
    index=False
)

# Salvar modelo escolhido e scaler juntos
pacote = {
    "modelo": melhor_modelo,
    "scaler": scaler,
    "caracteristicas": caracteristicas,
    "numero_grupos": melhor_k
}

joblib.dump(
    pacote,
    pasta_modelos / "modelo_vibetrack.joblib"
)

print("\n=== MODELO SELECIONADO ===")
print(f"Quantidade de grupos: {melhor_k}")
print(f"Silhouette score: {melhor_score:.4f}")
print("Modelo salvo em: models/modelo_vibetrack.joblib")
print("Comparação salva em: resultados_modelos.csv")
