# -*- coding: utf-8 -*-

# Importando as bibliotecas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

#Lendo a base de dados
df = pd.read_excel('/content/idh-fortaleza.xlsx')
df

#Visualizando as 5 primeiras linhas
df.head()

#Visualizando o total de linhas e colunas
df.shape

#Visualizando o tipo de dados para cada coluna
df.dtypes

#Lista das colunas a serem convertidas
colunas_numericas = ['IDH-Educação', 'IDH-Longevidade', 'IDH-Renda', 'IDH']

#Convertendo as colunas numéricas para o tipo float
for coluna in colunas_numericas:
    df = df.replace('-', np.nan)
    df[coluna] = df[coluna].astype(float)

#Visualizando o tipo de dados para cada coluna novamente
df.dtypes

#Verificando a quantidade de valores nulos em cada coluna
df.isnull().sum()

#Removendo linhas com valores vazios
df.dropna(axis=0, inplace=True)
df

#Analisando correlações entre as variáveis
df.corr()

#Verificando as estatísticas resumidas dos dados numéricos
df.describe()

#Gráfico que compara o IDH entre os diferentes bairros de Fortaleza
plt.figure(figsize=(12, 6))
sns.barplot(x=df['Bairros'], y=df['IDH'], color='blue')
plt.xlabel('Bairros')
plt.ylabel('IDH')
plt.title('Comparação do IDH por Bairro')
plt.xticks(rotation=90)  #Rotaciona os rótulos dos bairros em 90 graus
plt.tight_layout()  #Ajusta a distribuição dos elementos no gráfico
plt.show()

#Mapa de Calor do IDH de Fortaleza
plt.figure(figsize=(10, 8))
heatmap_data = df.pivot(index='Bairros', columns='Regional', values='IDH')
sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True, fmt=".2f", linewidths=0.5)
plt.xlabel('Regional')
plt.ylabel('Bairros')
plt.title('Mapa de Calor do IDH de Fortaleza')
plt.show()

#Selecionando as colunas relevantes para a clusterização
X = df[['IDH-Educação', 'IDH-Longevidade', 'IDH-Renda']]

#Método do cotovelo
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), wcss)
plt.title("Método do Cotovelo")
plt.xlabel("Número de Clusters")
plt.ylabel("WCSS")
plt.show()

#Treinando o modelo com um número escolhido de clusters
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)

#Coeficiente de silhueta
silhouette_avg = silhouette_score(X, kmeans.labels_)
print("Coeficiente de silhueta:", silhouette_avg)

y_kmeans

#Adicionando a coluna de clusters ao DataFrame original
df['cluster'] = pd.DataFrame(y_kmeans)
df

cluster1 = df[df["cluster"] == 1]
cluster1

from mpl_toolkits.mplot3d import Axes3D

# Criar a figura e o eixo 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Definir as cores para cada cluster
colors = ['red', 'blue', 'green', 'yellow', 'orange']

# Preencher valores ausentes com um valor padrão
df.fillna(0, inplace=True)

# Visualizando os clusters em 3D
for cluster in df['cluster'].unique():
    cluster_data = df[df['cluster'] == cluster]
    ax.scatter(cluster_data['IDH-Renda'], cluster_data['IDH-Educação'], cluster_data['IDH-Longevidade'], c=colors[int(cluster)-1], label=f'Cluster {cluster}')

# Configurar o gráfico 3D
ax.set_xlabel('IDH-Renda')
ax.set_ylabel('IDH-Educação')
ax.set_zlabel('IDH-Longevidade')
ax.set_title('Clusters de Bairros em Fortaleza')

# Criar uma legenda para cada cluster
ax.legend(title='Clusters')
plt.show()


#Este gráfico representa a clusterização dos bairros de Fortaleza com base em suas características socioeconômicas.
#Cada ponto no gráfico representa um bairro e está posicionado em um espaço tridimensional, onde os eixos representam as seguintes variáveis:
# -Eixo X: IDH-Renda (Índice de Desenvolvimento Humano relacionado à renda)
# -Eixo Y: IDH-Educação (Índice de Desenvolvimento Humano relacionado à educação)
# -Eixo Z: IDH-Longevidade (Índice de Desenvolvimento Humano relacionado à longevidade)
#Os pontos são coloridos de acordo com o cluster ao qual pertencem. Cada cor representa um cluster distinto.
#Através dessa visualização, é possível identificar padrões e segmentações dos bairros de Fortaleza com base em suas características socioeconômicas.

"""#Estatísticas descritivas dos Clusters"""

cluster1 = df[df["cluster"] == 1]
cluster1.describe()

cluster2 = df[df["cluster"] == 2]
cluster2.describe()

cluster3 = df[df["cluster"] == 3]
cluster3.describe()

cluster4 = df[df["cluster"] == 4]
cluster4.describe()

# Médias dos indicadores socioeconômicos por cluster
means = df.groupby('cluster')[['IDH-Educação', 'IDH-Longevidade', 'IDH-Renda']].mean()

# Gráfico de barras comparando as médias
means.plot(kind='bar', figsize=(10, 6))
plt.xlabel('Cluster')
plt.ylabel('Média')
plt.title('Média dos Indicadores Socioeconômicos por Cluster')
plt.legend(['IDH-Educação', 'IDH-Longevidade', 'IDH-Renda'])
plt.show()

# Matriz de correlação
correlation_matrix = df[['IDH-Educação', 'IDH-Longevidade', 'IDH-Renda']].corr()

# Mapa de calor
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Matriz de Correlação')
plt.show()


#O mapa de calor da matriz de correlação apresenta as relações entre as variáveis de IDH-Educação, IDH-Longevidade e IDH-Renda.
#As cores no mapa representam o grau de correlação entre as variáveis, sendo o vermelho para correlações positivas e o azul para correlações negativas.
#No mapa, podemos observar que a maior correlação positiva ocorre entre IDH-Educação e IDH-Renda, evidenciada pelo tom vermelho intenso.
#Isso indica que existe uma relação direta entre essas duas variáveis, sugerindo que regiões com maior IDH-Educação também tendem a ter uma maior IDH-Renda.
#Por outro lado, a correlação entre IDH-Longevidade e as outras variáveis é mais fraca, como indicado pelos tons mais claros de azul.
#Isso indica que a relação entre a longevidade e os outros indicadores pode ser menos direta ou influenciada por outros fatores.

# Análise de componentes principais (PCA)
from sklearn.decomposition import PCA

pca = PCA(n_components=3)
pca.fit(X)

# Variância explicada por cada componente
explained_variance = pca.explained_variance_ratio_
print('Variância explicada por cada componente:', explained_variance)

# Componentes principais
components = pd.DataFrame(pca.components_, columns=X.columns)
print('Componentes principais:')
print(components)

#Neste trecho de código, realizamos uma Análise de Componentes Principais (PCA) como uma técnica de redução de dimensionalidade.
#O PCA busca identificar combinações lineares das variáveis originais (no caso, as características socioeconômicas) que explicam a maior parte da variância nos dados.

"""# Gráficos de Distribuição dos Bairros em Fortaleza por Cluster
Em relação ao IDH-Longevidade, IDH-Renda e IDH-Educação.
"""

# Definir o número de clusters desejado
n_clusters = 5

# Executar o algoritmo de clustering (K-means)
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df['cluster'] = kmeans.fit_predict(df[['IDH-Renda', 'IDH-Educação']])

# Filtrar e selecionar 5 bairros de cada cluster
selected_bairros = []
for cluster in range(n_clusters):
    cluster_bairros = df[df['cluster'] == cluster].nlargest(5, 'IDH-Renda')
    selected_bairros.append(cluster_bairros)

df_selected = pd.concat(selected_bairros)

# Plotar o gráfico com os bairros selecionados
plt.figure(figsize=(10, 8))
scatter = plt.scatter(df_selected['IDH-Renda'], df_selected['IDH-Educação'], c=df_selected['cluster'], cmap='Set1')

# Adicionar rótulos aos pontos com espaçamento
for i, row in df_selected.iterrows():
    plt.text(row['IDH-Renda'], row['IDH-Educação'], row['Bairros'], fontsize=8, ha='center', va='center')

plt.xlabel('IDH-Renda')
plt.ylabel('IDH-Educação')
plt.title('Distribuição dos Bairros em Fortaleza por Cluster')
legend_elements = scatter.legend_elements()[0]
legend_labels = ['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3']
plt.legend(legend_elements, legend_labels, title='Clusters')

plt.show()

# Definir o número de clusters desejado
n_clusters = 5

# Executar o algoritmo de clustering (K-means)
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df['cluster'] = kmeans.fit_predict(df[['IDH', 'IDH-Longevidade']])

# Filtrar e selecionar 5 bairros de cada cluster
selected_bairros = []
for cluster in range(n_clusters):
    cluster_bairros = df[df['cluster'] == cluster].nlargest(5, 'IDH')
    selected_bairros.append(cluster_bairros)

df_selected = pd.concat(selected_bairros)

# Plotar o gráfico com os bairros selecionados
plt.figure(figsize=(10, 8))
scatter = plt.scatter(df_selected['IDH'], df_selected['IDH-Longevidade'], c=df_selected['cluster'], cmap='Set1')

# Adicionar rótulos aos pontos com espaçamento
for i, row in df_selected.iterrows():
    plt.text(row['IDH'], row['IDH-Longevidade'], row['Bairros'], fontsize=8, ha='center', va='center')

plt.xlabel('IDH')
plt.ylabel('IDH-Longevidade')
plt.title('Distribuição dos Bairros em Fortaleza por Cluster')
legend_elements = scatter.legend_elements()[0]
legend_labels = ['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3']
plt.legend(legend_elements, legend_labels, title='Clusters')

plt.show()