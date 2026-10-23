
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

modo_entrada = st.radio(
    "Como deseja informar suas preferências?",
    ["Controles deslizantes", "Entrada manual"],
    horizontal=True,
    key="modo_entrada",
)

campos_entrada = {
    "danceability": ("Dançabilidade", 0.60, "Indica o quanto uma música é adequada para dançar."),
    "energy": ("Energia", 0.70, "Representa a intensidade percebida da música."),
    "valence": ("Positividade musical", 0.60, "Indica uma sonoridade mais positiva ou negativa."),
    "acousticness": ("Característica acústica", 0.30, "Indica o grau de característica acústica."),
    "instrumentalness": ("Instrumentalidade", 0.10, "Indica a probabilidade de ausência de vocais."),
    "tempo": ("Ritmo (BPM)", 120.0, "Andamento musical em batidas por minuto."),
}

valores_informados = {}
col1, col2 = st.columns(2)

for indice, (chave, (rotulo, padrao, ajuda)) in enumerate(campos_entrada.items()):
    with (col1 if indice < 3 else col2):
        if modo_entrada == "Controles deslizantes":
            valores_informados[chave] = st.slider(
                rotulo,
                min_value=0.0,
                max_value=244.0 if chave == "tempo" else 1.0,
                value=padrao,
                step=1.0 if chave == "tempo" else 0.01,
                help=ajuda,
                key=f"slider_{chave}",
            )
        else:
            valores_informados[chave] = st.text_input(
                rotulo,
                value=f"{padrao:.0f}" if chave == "tempo" else f"{padrao:.2f}",
                help=ajuda,
                key=f"manual_{chave}",
            )

# Explicações disponíveis sem alterar os controles da interface.
with st.expander("ℹ️ Como preencher os campos?"):
    st.markdown("""
    Ajuste os controles de acordo com o tipo de música que você procura.
    Os cinco primeiros campos usam valores de **0 a 1**:

    - **Dançabilidade:** 0 indica pouca adequação para dança; 1 indica muita adequação.
    - **Energia:** 0 representa menor intensidade percebida; 1, maior intensidade.
    - **Positividade musical:** valores baixos sugerem uma sonoridade mais negativa ou melancólica; valores altos, uma sonoridade mais positiva.
    - **Característica acústica:** valores altos indicam maior confiança de que a gravação é acústica.
    - **Instrumentalidade:** valores altos indicam maior probabilidade de a faixa não conter vocais.
    - **Ritmo (BPM):** andamento estimado em **batidas por minuto**. Nesta aplicação, o controle vai de **0 a 244 BPM**.

    No modo **Controles deslizantes**, os limites são aplicados automaticamente.
    No modo **Entrada manual**, valores vazios, não numéricos ou fora dos limites são
    identificados antes da inferência. Você pode usar vírgula ou ponto decimal.
    """)

st.divider()

# ==========================================
# IDENTIFICAÇÃO DA IDENTIDADE SONORA
# ==========================================

if st.button(
    "🎧 Descobrir minha Identidade Sonora",
    type="primary"
):

    # Validar as duas modalidades antes de enviar dados ao modelo.
    limites = {chave: (0.0, 244.0 if chave == "tempo" else 1.0)
               for chave in campos_entrada}
    entrada_validada = {}
    erros = []

    for chave, (rotulo, _, _) in campos_entrada.items():
        valor_bruto = valores_informados[chave]
        texto = str(valor_bruto).strip()

        if not texto:
            erros.append(f"O campo '{rotulo}' é obrigatório.")
            continue

        try:
            valor = float(texto.replace(",", "."))
        except ValueError:
            erros.append(f"O campo '{rotulo}' deve conter um número válido.")
            continue

        if not np.isfinite(valor):
            erros.append(f"O campo '{rotulo}' deve conter um número finito.")
            continue

        minimo, maximo = limites[chave]
        if not minimo <= valor <= maximo:
            erros.append(f"O campo '{rotulo}' deve estar entre {minimo:g} e {maximo:g}.")
            continue

        entrada_validada[chave] = valor

    if erros:
        st.error("Foram encontrados problemas na entrada:")
        for erro in erros:
            st.write(f"• {erro}")
        st.stop()

    # Manter exatamente a ordem de características usada no treinamento.
    entrada = pd.DataFrame([entrada_validada])[caracteristicas]

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
# LIMITAÇÕES DA SOLUÇÃO
# ==========================================

with st.expander("ℹ️ Limitações do VibeTrack"):
    st.markdown("""
    - O VibeTrack identifica **semelhanças entre características musicais** usando agrupamento K-means. As identidades sonoras são nomes interpretativos dos grupos, não classificações definitivas de gênero musical ou personalidade.
    - As sugestões dependem das faixas presentes no conjunto de dados utilizado. O aplicativo não consulta um catálogo musical em tempo real.
    - O sistema não considera histórico de reprodução, avaliações pessoais nem preferências anteriores do visitante.
    - A qualidade dos resultados depende dos dados e das seis características analisadas. Músicas parecidas nesses atributos podem soar diferentes para uma pessoa.
    - O perfil alternativo representa um grupo matematicamente distante, não necessariamente o oposto em todos os aspectos musicais.
    """)

# ==========================================
# RODAPÉ
# ==========================================

st.divider()

st.caption(
    "© 2026 VibeTrack — Desenvolvido por "
    "Henrique Simon, Tainara Beck e Willer Tiago Cesti Dos Reis."
)
