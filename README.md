# VibeTrack — Explorador Inteligente de Músicas

Projeto desenvolvido para a disciplina **Tópicos Avançados em Inteligência Artificial**, sob orientação do **Prof. Dr. Márcio Andrey Teixeira**.

**Aplicação publicada:** https://vibetracktrabalho.streamlit.app/  
**Repositório:** https://github.com/HenriqueSimons/VibeTrack

## 1. Tema, proposta e objetivo

**Tema escolhido pelo grupo:** exploração de perfis musicais por meio de aprendizado de máquina não supervisionado.

A proposta do VibeTrack foi transformar atributos numéricos de músicas em uma experiência interativa: em vez de apenas mostrar resultados de um algoritmo, o sistema permite que a pessoa experimente combinações de características sonoras e descubra perfis e sugestões musicais. A escolha do tema musical, a proposta de identidades sonoras e a apresentação acessível dos resultados são decisões de concepção do projeto.

O VibeTrack é uma aplicação web que utiliza aprendizado de máquina não supervisionado para identificar perfis sonoros a partir de seis características informadas pelo usuário. O sistema atribui as preferências a um agrupamento, apresenta uma descrição interpretativa do perfil e recomenda músicas com características próximas. O público-alvo são pessoas interessadas em explorar músicas e conhecer perfis musicais semelhantes às suas preferências.

## 2. Funcionalidades e decisões do projeto

Além da implementação do algoritmo, o grupo definiu como apresentar os resultados para pessoas sem conhecimento técnico de aprendizado de máquina. Por isso, a aplicação usa nomes interpretativos para os agrupamentos, gráficos e descrições em português. Também foram incorporadas duas formas de entrada de dados e a opção de explorar um perfil diferente do identificado.

- Entrada de preferências por controles deslizantes ou preenchimento manual.
- Validação de campos obrigatórios, números inválidos e limites permitidos.
- Identificação de uma das seis identidades sonoras.
- Gráfico das características informadas.
- Recomendação de até dez músicas do grupo identificado, com explicação de semelhança.
- **Explore o Oposto:** apresentação de um perfil alternativo e cinco sugestões musicais.
- Informações sobre os campos, unidades e limitações da solução.

## 3. Dataset e preparação

**Fonte:** Spotify Tracks Dataset, disponibilizado em https://github.com/sai-chaitanya-reddy/spotify-tracks-dataset . O arquivo original utilizado é `spotify-tracks-dataset-detailed.csv`.

O dataset original contém **114.000 registros e 20 colunas**. A preparação removeu **450 linhas duplicadas** e **uma linha sem identificação musical**, resultando em **113.549 registros**. As seis características usadas pelo modelo são:

| Atributo | Significado | Intervalo na aplicação |
| --- | --- | --- |
| `danceability` | Adequação para dança | 0 a 1 |
| `energy` | Intensidade percebida | 0 a 1 |
| `valence` | Positividade musical | 0 a 1 |
| `acousticness` | Característica acústica | 0 a 1 |
| `instrumentalness` | Predominância instrumental | 0 a 1 |
| `tempo` | Andamento musical em batidas por minuto (BPM) | 0 a 244 |

As variáveis foram padronizadas com `StandardScaler` antes do agrupamento. Por se tratar de aprendizado não supervisionado, **não há variável-alvo**.

## 4. Treinamento e avaliação

Foram comparadas três configurações de `KMeans` (`k=4`, `k=6` e `k=8`), com `random_state=42` e `n_init=10`. O treinamento utilizou os 113.549 registros tratados. O índice de silhueta foi calculado sobre uma amostra fixa de 10.000 registros.

| Número de grupos | Índice de silhueta |
| --- | ---: |
| 4 | 0,219445 |
| **6** | **0,221659** |
| 8 | 0,199395 |

A configuração de **seis grupos** apresentou o maior índice entre as opções testadas e foi escolhida para o deploy. O valor próximo de 0,22 indica separação limitada entre os agrupamentos; os perfis são aproximações exploratórias, não classificações definitivas.

| Grupo | Nome interpretativo |
| --- | --- |
| 0 | Intensidade e Emoção |
| 1 | Explorador de Ritmos |
| 2 | Alma Acústica |
| 3 | Espírito Sereno |
| 4 | Energia Instrumental |
| 5 | Vibração Positiva |

## 5. Tecnologias

Python, Streamlit, pandas, NumPy, scikit-learn, joblib, Google Colab e GitHub.

**Importante:** o arquivo `requirements.txt` entregue deve informar **versões testadas** das bibliotecas. Confirme-as no ambiente usado para gerar e carregar o modelo; não considere uma lista sem versões como definitiva.

## 6. Como executar a aplicação localmente

1. Obtenha o projeto pelo repositório ou pelo ZIP da entrega.
2. Instale Python e abra um terminal na pasta do projeto.
3. Crie e ative um ambiente virtual (recomendado):

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

4. Instale as dependências:

   ```powershell
   python -m pip install -r requirements.txt
   ```

5. Confirme a presença dos arquivos `models/modelo_vibetrack.joblib` e `data/musicas_com_grupos.csv`.
6. Execute:

   ```powershell
   python -m streamlit run app.py
   ```

7. Acesse o endereço local exibido pelo terminal, normalmente `http://localhost:8501`.

O aplicativo **carrega o modelo e o padronizador salvos**; cada consulta executa apenas uma inferência, sem novo treinamento.

## 7. Como reproduzir o treinamento

O arquivo original `spotify-tracks-dataset-detailed.csv` pode ser obtido no repositório da fonte indicado na seção 3. Preserve o mesmo arquivo/versão para reproduzir os resultados.

- Os scripts de análise, preparação, treinamento e análise dos grupos devem acompanhar a entrega.
- O notebook `VibeTrack_IA.ipynb` demonstra a leitura do dataset original, a limpeza, a padronização, a comparação das configurações do K-means, a análise dos grupos, o salvamento do modelo e a inferência.
- O pacote `.joblib` contém o modelo K-means, o `StandardScaler`, a lista ordenada de características e o número de grupos.

## 8. Testes realizados

Foram executados cinco casos na aplicação publicada:

| Caso | Tipo | Resultado observado |
| --- | --- | --- |
| 01 | Válido — perfil positivo | Grupo 5 — Vibração Positiva |
| 02 | Válido — perfil tranquilo/acústico | Grupo 3 — Espírito Sereno |
| 03 | Válido — perfil energético/acelerado | Grupo 1 — Explorador de Ritmos |
| 04 | Inválido — dançabilidade igual a 12 | Mensagem de erro; nenhuma nova inferência |
| 05 | Limítrofe — combinação de valores extremos permitidos | Grupo 4 — Energia Instrumental |

As capturas de tela e a descrição completa das entradas, resultados esperados, resultados obtidos e comentários devem acompanhar o ZIP ou o relatório. Uma entrada com valores `0,60 / 0,70 / 0,60 / 0,30 / 0,10 / 120 BPM` produziu o grupo 5 tanto no Colab quanto no aplicativo publicado.

## 9. Limitações

- O K-means sempre associa uma entrada válida a um dos seis grupos, mesmo quando a combinação é incomum.
- Os nomes das identidades sonoras são interpretações humanas dos perfis médios.
- A separação entre os grupos, medida pela silhueta, é relativamente baixa.
- O sistema utiliza apenas seis características e não considera histórico pessoal ou avaliações dos usuários.
- As recomendações dependem do dataset fornecido e não de um catálogo atualizado em tempo real.

## 10. Integrantes e contribuições

- **Henrique Simon — Desenvolvimento e implementação:** programação da aplicação em Python e Streamlit, integração do modelo K-means, funcionalidades de recomendação, interface e publicação no GitHub e Streamlit Cloud.
- **Tainara Beck — Desenvolvimento e análise de dados:** participação no planejamento, preparação e análise do dataset, avaliação das configurações do K-means, interpretação dos agrupamentos, definição de funcionalidades e melhorias na experiência do usuário.
- **Willer Tiago Cesti Dos Reis — Documentação e testes:** organização da documentação acadêmica, estruturação do relatório, descrição da metodologia e funcionalidades, organização dos casos de teste, registro dos resultados e revisão do material de entrega.

As descrições de contribuição devem ser revisadas pelos integrantes para refletir as atividades efetivamente realizadas.

## 11. Processo de desenvolvimento e uso de IA generativa

O VibeTrack não foi produzido exclusivamente por ferramentas de IA generativa. O trabalho envolveu definição da proposta, escolha e estudo do dataset, preparação dos dados, experimentação com diferentes valores de `k`, interpretação dos resultados, implementação e publicação da aplicação, além de testes práticos com entradas válidas, inválidas e limítrofes.

**Decisões e atividades do projeto:** escolha do tema musical; definição das seis características de entrada; comparação de `k=4`, `k=6` e `k=8`; escolha de seis agrupamentos com base nos resultados; atribuição de nomes interpretativos aos perfis; construção das funcionalidades de Identidade Sonora e Explore o Oposto; criação de entradas por controles deslizantes e por digitação; validação dos campos; e execução de cinco casos de teste no aplicativo publicado.

**Apoio de IA generativa:** o ChatGPT foi utilizado como ferramenta auxiliar para discutir alternativas de implementação, esclarecer dúvidas de programação, sugerir melhorias, revisar textos e apoiar a organização da documentação. As sugestões foram adaptadas ao contexto do VibeTrack e verificadas durante a execução e os testes. O uso dessa ferramenta não substituiu o treinamento do modelo, a comparação dos resultados nem a verificação do funcionamento da aplicação.

O grupo assume a responsabilidade pela revisão, pela correção técnica e pela apresentação do trabalho. Esta seção deve ser conferida pelos integrantes para garantir que descreva fielmente o processo realizado.

## 12. Referências

- **Spotify Tracks Dataset:** https://github.com/sai-chaitanya-reddy/spotify-tracks-dataset
- **scikit-learn — KMeans:** https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
- **scikit-learn — StandardScaler:** https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
- **scikit-learn — silhouette_score:** https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html
- **Streamlit — Documentação:** https://docs.streamlit.io/
- **pandas — Documentação:** https://pandas.pydata.org/docs/
- **NumPy — Documentação:** https://numpy.org/doc/
- **joblib — Documentação:** https://joblib.readthedocs.io/

---
