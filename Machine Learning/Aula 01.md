# Aula 01 – Introdução ao Machine Learning (ML)

**Carga horária:** 4 horas
**Público-alvo:** Iniciantes em Machine Learning
**Pré-requisitos:** Lógica de programação e Python básico

---

# Objetivos da Aula

Ao final desta aula, você será capaz de:

✅ Entender o que é Machine Learning.

✅ Diferenciar programação tradicional de Machine Learning.

✅ Conhecer os principais tipos de aprendizado.

✅ Entender o conceito de treinamento e previsão.

✅ Criar seu primeiro modelo utilizando Scikit-Learn.

---

# 1. O que é Machine Learning?

Machine Learning (Aprendizado de Máquina) é uma área da Inteligência Artificial que permite que computadores aprendam padrões a partir dos dados sem serem explicitamente programados para cada situação.

## Analogia

Imagine uma criança aprendendo a identificar cachorros.

Você não entrega uma lista enorme de regras:

❌ Tem 4 patas

❌ Tem rabo

❌ Late

Porque existem exceções.

Em vez disso, você mostra centenas de fotos:

🐶 Cachorro

🐶 Cachorro

🐶 Cachorro

A criança começa a identificar padrões.

Machine Learning funciona da mesma forma.

---

# 2. Programação Tradicional x Machine Learning

## Programação Tradicional

### Entrada

* Dados
* Regras

### Resultado

* Resposta

Exemplo:

```python
idade = 20

if idade >= 18:
    print("Maior de idade")
else:
    print("Menor de idade")
```

O programador escreveu as regras.

---

## Machine Learning

### Entrada

* Dados
* Respostas corretas

### Resultado

* Regras aprendidas

Exemplo:

| Idade | Comprou |
| ----- | ------- |
| 18    | Sim     |
| 20    | Sim     |
| 15    | Não     |
| 16    | Não     |

O algoritmo encontra sozinho os padrões.

---

# 3. Onde usamos Machine Learning?

Machine Learning está presente diariamente:

### Netflix

Sugestão de filmes.

### Spotify

Sugestão de músicas.

### Instagram

Conteúdo recomendado.

### YouTube

Vídeos sugeridos.

### Gmail

Filtro de spam.

### Bancos

Detecção de fraudes.

### Medicina

Auxílio em diagnósticos.

---

# 4. Principais Tipos de Aprendizado

## Aprendizado Supervisionado

Possui respostas corretas.

### Exemplo

Queremos prever:

* Compra ou não compra.
* Aprova ou reprova.
* Tem doença ou não.

Dados:

| Idade | Comprou |
| ----- | ------- |
| 20    | Sim     |
| 30    | Sim     |
| 15    | Não     |

A coluna "Comprou" é a resposta.

---

## Aprendizado Não Supervisionado

Não possui resposta.

O computador tenta encontrar grupos.

### Exemplo

Clientes de uma loja.

O algoritmo pode descobrir:

Grupo A → Jovens

Grupo B → Adultos

Grupo C → Idosos

Sem ninguém informar isso.

---

## Aprendizado por Reforço

Aprende através de recompensas.

### Exemplo

Jogos.

O sistema recebe pontos quando toma boas decisões.

Foi assim que várias IAs aprenderam a jogar:

* Xadrez
* Go
* Atari

---

# 5. Como o Computador Aprende?

Imagine que queremos prever se uma pessoa fará uma compra.

Dados:

| Idade | Salário | Comprou |
| ----- | ------- | ------- |
| 18    | 2000    | Não     |
| 25    | 4000    | Sim     |
| 35    | 6000    | Sim     |
| 20    | 1800    | Não     |

---

## Etapa 1

Fornecemos os dados.

---

## Etapa 2

O algoritmo procura padrões.

Exemplo:

"Pessoas com salários maiores tendem a comprar."

---

## Etapa 3

O modelo aprende.

---

## Etapa 4

Recebe novos dados.

| Idade | Salário |
| ----- | ------- |
| 30    | 5000    |

---

## Etapa 5

Realiza uma previsão.

Resultado:

```text
Provavelmente comprará.
```

---

# 6. Conceitos Fundamentais

## Dataset

Conjunto de dados utilizado para treinar o modelo.

Exemplo:

| Idade | Salário | Comprou |
| ----- | ------- | ------- |
| 20    | 2000    | Não     |
| 30    | 5000    | Sim     |

---

## Features

Características utilizadas para prever.

Neste caso:

```text
Idade
Salário
```

---

## Target

Resposta desejada.

```text
Comprou
```

---

# 7. Conhecendo o Scikit-Learn

É a principal biblioteca de Machine Learning em Python.

Instalação:

```python
pip install scikit-learn
```

Importação:

```python
from sklearn.tree import DecisionTreeClassifier
```

---

# 8. Nosso Primeiro Modelo

## Passo 1 – Criar os dados

```python
import pandas as pd

dados = {
    "idade": [18, 22, 25, 30, 35, 40],
    "salario": [1500, 2500, 3000, 4500, 5000, 6000],
    "comprou": [0, 0, 1, 1, 1, 1]
}

df = pd.DataFrame(dados)

print(df)
```

---

## Passo 2 – Separar entradas e saídas

```python
X = df[["idade", "salario"]]

y = df["comprou"]
```

### Explicação

X representa as informações usadas para prever.

```python
idade
salario
```

y representa a resposta.

```python
comprou
```

---

## Passo 3 – Criar o modelo

```python
from sklearn.tree import DecisionTreeClassifier

modelo = DecisionTreeClassifier()
```

### O que é isso?

Estamos criando uma Árvore de Decisão.

Ela aprende fazendo perguntas:

```text
Salário > 3000?
      |
      Sim
      |
Comprou
```

---

## Passo 4 – Treinar

```python
modelo.fit(X, y)
```

### Analogia

Imagine um professor mostrando exercícios resolvidos para um aluno.

O aluno estuda os exemplos.

O método:

```python
fit()
```

faz exatamente isso.

---

## Passo 5 – Fazer previsão

```python
nova_pessoa = [[28, 3500]]

resultado = modelo.predict(nova_pessoa)

print(resultado)
```

Saída:

```text
[1]
```

---

## Interpretando

```python
if resultado[0] == 1:
    print("Comprará")
else:
    print("Não comprará")
```

---

# 9. Fluxo Completo do Machine Learning

```text
Coleta dos Dados
        ↓
Limpeza dos Dados
        ↓
Treinamento
        ↓
Avaliação
        ↓
Previsão
        ↓
Melhoria do Modelo
```

---

# 10. Atividade Prática

Crie um modelo para prever aprovação.

Dados:

| Horas Estudo | Aprovado |
| ------------ | -------- |
| 1            | 0        |
| 2            | 0        |
| 3            | 0        |
| 4            | 1        |
| 5            | 1        |
| 6            | 1        |

Pergunta:

Uma pessoa que estudou 5 horas será aprovada?

---

# Exercícios

## Exercício 1

Explique com suas palavras:

* O que é Machine Learning?

---

## Exercício 2

Qual a diferença entre:

* Programação tradicional
* Machine Learning

---

## Exercício 3

Classifique:

a) Detecção de spam

b) Recomendação de filmes

c) Agrupamento de clientes

Como:

* Supervisionado
* Não supervisionado

---

## Exercício 4

Identifique:

| Idade | Salário | Comprou |
| ----- | ------- | ------- |
| 25    | 3000    | Sim     |

Quais são:

* Features?
* Target?

---

## Exercício 5

Complete:

```python
from sklearn.tree import __________

modelo = __________()

modelo.fit(X, y)
```

---

# Desafio da Aula

Crie um modelo para prever se um aluno será aprovado utilizando:

* Nota 1
* Nota 2

Como entrada.

E:

* Aprovado (0 ou 1)

Como saída.

---

# Resumo

Nesta aula aprendemos:

✅ O que é Machine Learning

✅ Diferença entre programação e ML

✅ Tipos de aprendizado

✅ Dataset, Features e Target

✅ Biblioteca Scikit-Learn

✅ Árvore de Decisão

✅ Método `fit()`

✅ Método `predict()`

✅ Primeiro modelo de Machine Learning

Na próxima aula, o foco será **NumPy e Pandas para preparação de dados**, etapa essencial para qualquer projeto de Machine Learning.
