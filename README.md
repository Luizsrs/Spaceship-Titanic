# Spaceship Titanic - Machine Learning Classification

[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-blue)](https://www.kaggle.com/competitions/spaceship-titanic)
[![Python](https://img.shields.io/badge/Python-3.x-green)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Enterprise-orange)](https://scikit-learn.org/)

Este repositório contém a solução completa para o desafio **Spaceship Titanic** do Kaggle. O objetivo principal é prever quais passageiros foram transportados para uma dimensão alternativa após uma colisão com uma anomalia espacial, utilizando técnicas avançadas de Engenharia de Atributos (Feature Engineering) e Modelagem Preditiva.

---

## O Problema de Negócio (Contexto)
Diferente do desastre histórico, este é um problema de classificação binária ambientado no ano de 2912. O conjunto de dados contém registros de cerca de 8.700 passageiros do cargueiro espacial *Spaceship Titanic*. A variável alvo a ser predita é `Transported` (True/False).

---

## O Pipeline de Ciência de Dados

O projeto foi construído seguindo um pipeline rigoroso focado em evitar vazamento de dados (*data leakage*) e garantir a robustez estatística do modelo.

### 1. Engenharia de Atributos (Feature Engineering)
Antes do tratamento de dados, extraí sinal estatístico de variáveis complexas:
* **Divisão da Cabine:** A string original `Cabin` (ex: `B/0/P`) foi desmembrada em três novos atributos: `Deck`, `Num` e `Side` (Bordo/Estibordo), permitindo ao modelo capturar padrões de localização física na nave.
* **Agregação de Gastos:** Criamos o atributo `TotalSpent` somando todas as facilidades pagas (`RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck`), gerando uma métrica de poder aquisitivo e comportamento do passageiro.

### 2. Limpeza e Tratamento de Dados Nulos
Em vez de simplesmente descartar registros ausentes (o que reduziria o poder preditivo), aplicamos regras de negócio explícitas:
* **Lógica do Sono Criogênico:** Passageiros confirmados em `CryoSleep` tiveram seus gastos nulos preenchidos com `0.0`, dado que clientes confinados não consomem serviços.
* **Imputação por Moda/Mediana:** Variáveis categóricas residuais foram preenchidas com a moda e variáveis numéricas com a mediana para neutralizar o impacto de outliers.

### 3. Codificação Categórica (Encoding)
* Aplicamos **One-Hot Encoding** (`pd.get_dummies`) com o parâmetro `drop_first=True` nas variáveis nominais (`HomePlanet`, `Destination`, `Deck`, `Side`). Isso elimina a redundância matemática (multicolinearidade) e converte textos em vetores puramente numéricos.
* Tipos booleanos nativos (`VIP`, `CryoSleep`) foram explicitamente convertidos para inteiros (`0` e `1`).

### 4. Validação e Treinamento do Modelo
* **Estratégia de Validação:** Divisão Holdout (80% treino / 20% validação) utilizando `train_test_split` com semente aleatória fixa para garantir a reprodutibilidade dos experimentos.
* **Algoritmo:** Utilizou-se o **Random Forest Classifier** (`scikit-learn`), escolhido pela sua excelente capacidade de lidar com dados estruturados mistos e robustez natural contra *overfitting*.

---

## Como Executar o Projeto

1. Clone o repositório:
   ```bash
   git clone (https://github.com/Luizsrs/Spaceship-Titanic.git)
