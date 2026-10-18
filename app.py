
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================================
# CONFIGURAÇÃO DA PÁGINA
# ==========================================

st.set_page_config(
    page_title="VibeTrack | Identidade Sonora",
    page_icon="🎵",
    layout="wide"
)

# ==========================================
# IDENTIDADES SONORAS
# ==========================================

identidades = {
    0: {
        "nome": "Intensidade e Emoção",
        "descricao": (
            "Seu perfil apresenta músicas intensas, "
            "energéticas e com uma atmosfera mais emocional."
        )
    },
    1: {
        "nome": "Explorador de Ritmos",
        "descricao": (
            "Seu perfil combina alta energia com ritmos "
            "acelerados e bastante movimento."
        )
    },
    2: {
        "nome": "Alma Acústica",
        "descricao": (
            "Seu perfil valoriza músicas acústicas, "
            "mais suaves e de energia moderada ou baixa."
        )
    },
    3: {
        "nome": "Espírito Sereno",
        "descricao": (
            "Seu perfil apresenta músicas tranquilas, "
            "instrumentais e predominantemente acústicas."
        )
    },
    4: {
        "nome": "Energia Instrumental",
        "descricao": (
            "Seu perfil combina músicas instrumentais "
            "com bastante energia e intensidade."
        )
    },
    5: {
        "nome": "Vibração Positiva",
        "descricao": (
            "Seu perfil apresenta músicas dançantes, "
            "energéticas e com sonoridade mais positiva."
        )
    }
}

nomes_caracteristicas = {
    "danceability": "Dançabilidade",
    "energy": "Energia",
    "valence": "Positividade",
    "acousticness": "Característica acústica",
    "instrumentalness": "Instrumentalidade",
    "tempo": "Ritmo"
}

# ==========================================
# CARREGAMENTO DOS ARQUIVOS
# ==========================================

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

# ==========================================
# CABEÇALHO
# ==========================================

st.title("🎵 VibeTrack")
st.subheader("Descubra sua Identidade Sonora")

st.write(
    "Explore suas preferências musicais e encontre "
    "músicas semelhantes utilizando inteligência artificial."
)

st.divider()

# ==========================================
# ENTRADA DE DADOS
# ==========================================

st.header("Configure suas preferências musicais")

col1, col2 = st.columns(2)

with col1:

    danceability = st.slider(
        "Dançabilidade",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.01,
        help="Indica o quanto uma música é adequada para dançar."
    )

    energy = st.slider(
        "Energia",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.01,
        help="Representa a intensidade percebida da música."
    )

    valence = st.slider(
        "Positividade musical",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.01,
        help="Indica uma sonoridade mais positiva ou negativa."
    )

with col2:

    acousticness = st.slider(
        "Característica acústica",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.01,
        help="Indica o grau de característica acústica."
    )

    instrumentalness = st.slider(
        "Instrumentalidade",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.01,
        help="Indica a presença de conteúdo instrumental."
    )

    tempo = st.slider(
        "Ritmo (BPM)",
        min_value=0.0,
        max_value=244.0,
        value=120.0,
        step=1.0,
        help="Andamento musical em batidas por minuto."
    )

st.divider()

# ==========================================
# IDENTIFICAÇÃO DA IDENTIDADE SONORA
# ==========================================

if st.button(
    "🎧 Descobrir minha Identidade Sonora",
    type="primary"
):

    # Organizar os dados informados
    entrada = pd.DataFrame([{
        "danceability": danceability,
        "energy": energy,
        "valence": valence,
        "acousticness": acousticness,
        "instrumentalness": instrumentalness,
        "tempo": tempo
    }])

    # Garantir a mesma ordem usada no treinamento
    entrada = entrada[caracteristicas]

    # Aplicar a padronização do treinamento
    entrada_escalada = scaler.transform(entrada)

    # Identificar o grupo
    grupo = int(modelo.predict(entrada_escalada)[0])

    identidade = identidades[grupo]

    # ======================================
    # 1. IDENTIDADE SONORA
    # ======================================

    st.header("Sua Identidade Sonora")

    st.success(f"🎵 {identidade['nome']}")

    st.write(identidade["descricao"])

    # ======================================
    # 2. IMPRESSÃO DIGITAL MUSICAL
    # ======================================

    st.subheader("Sua Impressão Digital Musical")

    valores_grafico = entrada.iloc[0].copy()

    # Converter BPM para escala de 0 a 1
    # somente para exibição no gráfico
    valores_grafico["tempo"] = (
        valores_grafico["tempo"] / 244.0
    )

    grafico = pd.DataFrame({
        "Característica": [
            nomes_caracteristicas[c]
            for c in caracteristicas
        ],
        "Intensidade": [
            float(valores_grafico[c])
            for c in caracteristicas
        ]
    })

    # Gráfico de barras original
    st.bar_chart(
        grafico.set_index("Característica")
    )

    st.caption(
        "O ritmo foi convertido para uma escala de 0 a 1 "
        "apenas para visualização."
    )

    # ======================================
    # 3. RECOMENDAÇÕES MUSICAIS
    # ======================================

    st.subheader("Músicas que combinam com você")

    # Selecionar músicas do grupo identificado
    musicas_grupo = df[
        df["grupo"] == grupo
    ].copy()

    # Padronizar as características das músicas
    musicas_escaladas = scaler.transform(
        musicas_grupo[caracteristicas]
    )

    # Calcular a distância entre as preferências
    # informadas e as músicas do grupo
    musicas_grupo["distancia"] = np.linalg.norm(
        musicas_escaladas - entrada_escalada[0],
        axis=1
    )

    # Selecionar as 10 músicas mais semelhantes
    recomendacoes = (
        musicas_grupo
        .sort_values("distancia")
        .drop_duplicates(
            subset=["track_name", "artists"]
        )
        .head(10)
    )

    for _, musica in recomendacoes.iterrows():

        with st.container(border=True):

            st.markdown(
                f"**🎶 {musica['track_name']}**"
            )

            st.write(
                f"Artista: {musica['artists']}"
            )

            st.caption(
                f"Gênero: {musica['track_genre']}"
            )

            # Identificar a característica
            # mais próxima da preferência informada
            diferencas = {}

            for caracteristica in caracteristicas:

                diferenca = abs(
                    float(entrada.iloc[0][caracteristica])
                    - float(musica[caracteristica])
                )

                # Ajustar a escala do BPM
                if caracteristica == "tempo":
                    diferenca = diferenca / 244.0

                diferencas[caracteristica] = diferenca

            mais_proxima = min(
                diferencas,
                key=diferencas.get
            )

            st.caption(
                "Por que essa música? "
                "Ela possui uma característica próxima "
                "da sua preferência em "
                f"{nomes_caracteristicas[mais_proxima].lower()}."
            )

    # ======================================
    # 4. EXPLORE O OPOSTO
    # ======================================

    st.subheader("Explore o Oposto")

    st.write(
        "Descubra músicas de um perfil sonoro "
        "diferente daquele identificado para você."
    )

    # Encontrar o centroide mais distante
    distancias_centroides = np.linalg.norm(
        modelo.cluster_centers_
        - entrada_escalada[0],
        axis=1
    )

    grupo_oposto = int(
        np.argmax(distancias_centroides)
    )

    st.info(
        "Perfil alternativo: "
        f"{identidades[grupo_oposto]['nome']}"
    )

    # Selecionar músicas do perfil alternativo
    musicas_opostas = df[
        df["grupo"] == grupo_oposto
    ].copy()

    # Padronizar características
    musicas_opostas_escaladas = scaler.transform(
        musicas_opostas[caracteristicas]
    )

    # Calcular proximidade com o centroide
    # do grupo alternativo
    musicas_opostas["distancia_centro"] = np.linalg.norm(
        musicas_opostas_escaladas
        - modelo.cluster_centers_[grupo_oposto],
        axis=1
    )

    # Selecionar cinco músicas representativas
    sugestoes_opostas = (
        musicas_opostas
        .sort_values("distancia_centro")
        .drop_duplicates(
            subset=["track_name", "artists"]
        )
        .head(5)
    )

    for _, musica in sugestoes_opostas.iterrows():

        st.write(
            f"🎵 **{musica['track_name']}** "
            f"— {musica['artists']}"
        )

    st.caption(
        "O perfil alternativo é escolhido pela distância "
        "entre suas preferências e os centros dos grupos. "
        "Isso não significa que as músicas sejam opostas "
        "em todas as características."
    )

# ==========================================
# RODAPÉ
# ==========================================

st.divider()

st.caption(
    "© 2026 VibeTrack — Desenvolvido por "
    "Henrique Simon, Tainara Beck e Willer Tiago Cesti Dos Reis."
)
