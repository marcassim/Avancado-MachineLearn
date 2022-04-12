import os
from io import open
from random import random

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

# delimitando o tamanho máximo das colunas
pd.options.display.max_columns = 150

# carregando a base de dados retirada
# do site https://www.kaggle.com/datasets/tunguz/big-five-personality-test
data = pd.read_csv("data-final.csv", sep="\t")

# excluindo as colunas não utilizadas (da 50 até 110)
data.drop(data.columns[50:110], axis=1, inplace=True)

# Analisando as estatísticas da base de dados
pd.options.display.float_format = "{:.2f}".format

# muitos registros tinham o valor ZERO, causando ruido na análise
# retirando os valores ZERO
data = data[(data > 0.00).all(axis=1)]

# instanciando métodos Keans e o Visualizer
kmeans = KMeans()

# k é o número de parâmetros a ser estudado
visualizer = KElbowVisualizer(kmeans, k=(2, 10))

# Utilizando uma amostra padrão de 5000 registros, testar qual
# seria a melhor divisão de dados para a análise utlizando o método ELBOW
# from random import random
# data_sample = data.sample(n=5000, random_state=1)
# visualizer.fit(data_sample)
# visualizer.poof()

# Atribuindo os registros aos grupos
kmeans = KMeans(n_clusters=5)
k_fit = kmeans.fit(data)

# inserindo os rótulos dos clusters no dataframe
predicoes = k_fit.labels_
data["Clusters"] = predicoes

# Agrupando os registros por grupos
data.groupby("Clusters").mean()

# Selecionando as colunas de cada grupo
col_list = list(data)
ext = col_list[0:10]
est = col_list[10:20]
agr = col_list[20:30]
csn = col_list[30:40]
opn = col_list[40:50]

# Somando e tirando as médias dos valores de cada grupo
data_clusters = data_soma = pd.DataFrame()
data_soma["Extrovertido"] = data[ext].sum(axis=1) / 10
data_soma["Neurótico"] = data[est].sum(axis=1) / 10
data_soma["Agradável"] = data[agr].sum(axis=1) / 10
data_soma["Consciente"] = data[csn].sum(axis=1) / 10
data_soma["Receptivo"] = data[opn].sum(axis=1) / 10
data_soma["clusters"] = predicoes

# Visualizando as médias
data_clusters = data_soma.groupby("clusters").mean()

# criando os gráficos de cada grupo
plt.figure(figsize=(22, 3))
for i in range(0, 5):
    plt.subplot(1, 5, i + 1)
    plt.bar(data_clusters.columns, data_clusters.iloc[:, i], color="green", alpha=0.2)
    plt.plot(data_clusters.columns, data_clusters.iloc[:, i], color="red")
    plt.title("Grupo" + str(i))
    plt.xticks(rotation=45)
    plt.ylim(0, 4)

data[:0].to_excel("perguntas.xlsx", index=False)

meus_dados = pd.read_excel("perguntas.xlsx")

grupo_personalidade = k_fit.predict(meus_dados)[0]
print("Meu grupo de personalidade é:", grupo_personalidade)
