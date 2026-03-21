# Ciencia-de-dados
Projeto de Alexandre Nóbrega

Visão Geral

Este projeto foca-se na análise e previsão de atrasos de voos utilizando técnicas de machine learning. Através do uso de dados históricos de voos, o objetivo é identificar os principais fatores que contribuem para atrasos e construir modelos preditivos capazes de estimar se um voo irá sofrer atraso.

Objetivos
Analisar padrões e tendências nos atrasos de voos
Realizar limpeza e pré-processamento dos dados
Criar novas variáveis (feature engineering) para melhorar o desempenho dos modelos
Aplicar análise estatística e testes de hipóteses
Construir e avaliar modelos preditivos
Identificar as variáveis mais relevantes que influenciam os atrasos

Descrição do Dataset

O dataset contém informação detalhada sobre voos, incluindo:

Data do voo (FL_DATE)
Companhia aérea (AIRLINE)
Aeroporto de origem e destino (ORIGIN, DEST)
Horários de partida e chegada
Distância e duração do voo
Causas de atraso (transportadora, meteorologia, NAS, etc.)

O dataset não está incluído neste repositório devido ao seu tamanho.

Pode ser descarregado em:
https://www.kaggle.com/datasets/patrickzel/flight-delay-and-cancellation-dataset-2019-2023/data

Estrutura do Projeto
projeto de CD/
│
├── data/                 # Dataset (não incluído no repositório)
│
├── python/               # Código principal (Jupyter + scripts Python)
│   ├── outputs/          # Resultados gerados (análise de features, etc.)
│
├── outputs/plots/        # Visualizações da análise exploratória (EDA)
│
├── README.md
└── Report

Metodologia
1. Preparação dos Dados
Remoção de voos cancelados
Tratamento de valores em falta
Eliminação de variáveis com fuga de informação (data leakage)
2. Engenharia de Features
Variáveis temporais (mês, dia da semana, hora)
Features baseadas em rotas
Tráfego e taxas de atraso por aeroporto
Padrões de atraso por companhia aérea
Features de interação entre variáveis
3. Análise Exploratória de Dados (EDA)
Análise de distribuições
Análise de correlações
Padrões de atraso por companhia, tempo e distância
4. Análise de Features
PCA (Análise de Componentes Principais)
Informação Mútua (Mutual Information)
Testes estatísticos (ANOVA, Kruskal-Wallis, t-test)
5. Modelação
Modelos de machine learning para previsão de atrasos
Avaliação da importância das variáveis
