
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Configuração da página
st.set_page_config(
    page_title="VibeTrack",
    page_icon="🎵",
    layout="wide"
)



# Nomes dos grupos identificados na análise
nomes_grupos = {
    0: "Intensidade e Emoção",
    1: "Ritmo Acelerado",
    2: "Acústico e Tranquilo",
    3: "Instrumental e Sereno",
    4: "Instrumental Energético",
    5: "Animado e Dançante"
}

# Carregar modelo e dados somente quando necessário
@st.cache_resource
def carregar_modelo():
    return joblib.load("models/modelo_vibetrack.joblib")

@st.cache_data
def carregar_musicas():
    return pd.read_csv("data/musicas_com_grupos.csv")

pacote = carregar_modelo()
df = carregar_musicas()

modelo = pacote["modelo"]
scaler = pacote["scaler"]
caracteristicas = pacote["caracteristicas"]

# Título
st.title("🎵 VibeTrack")
st.subheader("Explorador Inteligente de Músicas")

st.write(
    "Descubra seu perfil musical e encontre músicas semelhantes "
    "com inteligência artificial."
)

st.divider()

st.header("Configure suas preferências musicais")

col1, col2 = st.columns(2)

with col1:
    danceability = st.slider(
        "Dançabilidade",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.01
    )

    energy = st.slider(
        "Energia",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.01
    )

    valence = st.slider(
        "Positividade musical",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.01
    )

with col2:
    acousticness = st.slider(
        "Característica acústica",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.01
    )

    instrumentalness = st.slider(
        "Instrumentalidade",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.01
    )

    tempo = st.slider(
        "Ritmo (BPM)",
        min_value=0.0,
        max_value=244.0,
        value=120.0,
        step=1.0
    )

st.divider()

if st.button("🎧 Descobrir minha vibe", type="primary"):

    # Preparar a entrada na ordem usada no treinamento
    entrada = pd.DataFrame([{
        "danceability": danceability,
        "energy": energy,
        "valence": valence,
        "acousticness": acousticness,
        "instrumentalness": instrumentalness,
        "tempo": tempo
    }])

    entrada = entrada[caracteristicas]

    # Usar o mesmo scaler do treinamento
    entrada_scaled = scaler.transform(entrada)

    # Prever grupo musical
    grupo = int(modelo.predict(entrada_scaled)[0])

    nome_grupo = nomes_grupos.get(
        grupo,
        f"Grupo {grupo}"
    )

    st.success(f"Sua vibe é: {nome_grupo}")

    st.write(
        f"Suas preferências foram associadas ao grupo {grupo} "
        "pelo modelo K-means."
    )

    # Selecionar músicas do grupo previsto
    musicas_grupo = df[df["grupo"] == grupo].copy()

    # Calcular proximidade entre entrada e músicas
    X_musicas = musicas_grupo[caracteristicas]

    musicas_scaled = scaler.transform(X_musicas)

    distancias = np.linalg.norm(
        musicas_scaled - entrada_scaled[0],
        axis=1
    )

    musicas_grupo["distancia"] = distancias

    # Selecionar músicas mais próximas
    recomendacoes = (
        musicas_grupo
        .sort_values("distancia")
        .drop_duplicates(
            subset=["track_name", "artists"]
        )
        .head(10)
    )

    st.subheader("🎶 Músicas recomendadas")

    for _, musica in recomendacoes.iterrows():
        st.markdown(
            f"**{musica['track_name']}** — "
            f"{musica['artists']}"
        )

        st.caption(
            f"Gênero no dataset: {musica['track_genre']}"
        )

    st.info(
        "As recomendações são calculadas pela proximidade "
        "das características musicais. "
        "O sistema não analisa o áudio diretamente."
    )

st.divider()

st.caption(
    "© 2026 VibeTrack — Desenvolvido Henrique Simon, Tainara Beck e Willer Tiago Cesti Dos Reis"
)
