# Representação, redução de dimensão ou algoritmo?

**O que determina a qualidade do clustering sobre features de modelos pré-treinados**

*Proposta de trabalho final · Aprendizado de Máquina Não Supervisionado · UFG · novembro de 2026*

## 1. Motivação

Clustering de imagens deixou de aprender representação e agrupamento ao mesmo tempo e passou a agrupar features congeladas de modelos pré-treinados. Nesse cenário, K-means sobre as features virou o baseline padrão: métodos recentes partem dele e o superam por margens de 6 a 12 pontos de acurácia \[1, 2\].

Na prática, quem aplica clustering faz três escolhas: a **representação** (que modelo extrai as features), a **redução de dimensão** (se e como reduzir antes de agrupar) e o **algoritmo**. A literatura costuma reportar a melhor combinação medida com rótulos. Em um problema não supervisionado real, porém, não há rótulos para escolher a configuração. Este trabalho mede o peso de cada escolha e testa se é possível escolher bem sem rótulos.

## 2. Perguntas de pesquisa e hipóteses

**QP1.** Quanto da variação na qualidade do clustering é explicado pela representação, pela redução de dimensão e pelo algoritmo?

**QP2.** Métricas internas (Silhouette, Davies-Bouldin, Calinski-Harabasz) ordenam as configurações de forma coerente com métricas externas (ARI, NMI, ACC)? Ou seja, dá para escolher a configuração sem rótulos?

**QP3.** Aplicar UMAP antes do clustering melhora o resultado, e para quais algoritmos?

Hipóteses:

- **H1.** A representação explica mais variância do ARI do que o algoritmo, como sugerem \[1, 2\].

- **H2.** Dentro de uma mesma representação, as métricas internas se correlacionam com as externas. Entre representações diferentes, a correlação cai, porque os índices internos dependem da geometria e da dimensão do espaço em que são calculados.

- **H3.** O ganho de ARI com UMAP é maior para HDBSCAN do que para K-means, em linha com \[6\].

## 3. Trabalhos relacionados

**Clustering com modelos pré-treinados.** Adaloglou et al. \[1\] treinam cabeças de clustering sobre 17 modelos pré-treinados e superam o K-means em 6,1% no ImageNet e 12,2% no CIFAR-100. Gadetsky et al. \[2\] propõem o TURTLE, que busca a rotulação que induz classificadores de margem máxima no espaço de features e usa o K-means como baseline. Em texto, Petukhova et al. \[12\] chegam a um padrão parecido com embeddings de LLMs. Nenhum desses trabalhos analisa como escolher a configuração sem rótulos.

**Redução de dimensão antes do clustering.** Herrmann et al. \[6\] mostram que UMAP antes do DBSCAN melhora a detecção de clusters e reduz a sensibilidade a parâmetros. Kobak e Linderman \[7\] mostram que a inicialização, e não o algoritmo em si, explica boa parte da preservação de estrutura global em UMAP e t-SNE. Chari e Pachter \[8\] argumentam que embeddings 2D distorcem demais os dados para servir à análise exploratória. Lause et al. \[9\] contestam: com métricas de vizinhança e de classe, os embeddings preservam estrutura relevante. Este trabalho testa essa questão de forma quantitativa, usando clustering como tarefa final.

## 4. Metodologia

### 4.1 Dados

CIFAR-10 \[13\], CIFAR-100 (avaliado com as 20 superclasses e com as 100 classes, para testar granularidade) e STL-10 \[14\]. Usamos o conjunto de teste de cada um (10.000, 10.000 e 8.000 imagens): assim n é igual para todos os métodos e o aglomerativo, de memória O(n²), continua viável. Os rótulos são usados apenas na avaliação.

### 4.2 Grade experimental

| **Fator**           | **Níveis**                                                                                                                 |
|---------------------|----------------------------------------------------------------------------------------------------------------------------|
| Representação       | Pixels (32×32, piso); ResNet-50 supervisionada no ImageNet \[15\] (referência); DINOv2 ViT-B/14 \[3\]; CLIP ViT-B/16 \[4\] |
| Redução de dimensão | Nenhuma; PCA (50 componentes); UMAP \[5\] (10 dimensões, n_neighbors ∈ {15, 50}, inicialização espectral)                  |
| Algoritmo           | K-means (k-means++, 10 inicializações); GMM (covariância diagonal); aglomerativo (Ward); HDBSCAN \[10\] (sem k)            |
| Repetições          | 5 seeds por configuração                                                                                                   |

Features vêm do token CLS ou do pooling final, com normalização L2. K-means, GMM e aglomerativo recebem k igual ao número de classes. São 4 conjuntos de rótulos × 4 representações × 4 reduções × 4 algoritmos × 5 seeds, ou 1.280 execuções. A extração de features roda uma única vez na GPU (RTX 4090); o resto roda em CPU sobre arrays em cache.

### 4.3 Métricas

- **Externas (com rótulos):** ARI \[11\], a métrica principal por ser corrigida para o acaso; NMI; e ACC com casamento húngaro entre clusters e classes.

- **Internas (sem rótulos):** Silhouette \[16\], Davies-Bouldin \[17\] e Calinski-Harabasz \[18\], calculadas no espaço em que o clustering foi feito.

- **HDBSCAN:** o ruído conta como um cluster próprio nas métricas externas, e a cobertura (fração de pontos atribuídos) é reportada à parte.

### 4.4 Análises

- **A1 (QP1):** ANOVA do ARI com representação, redução e algoritmo como fatores, por conjunto de dados; o tamanho de efeito de cada fator é reportado como η².

- **A2 (QP2):** τ de Kendall entre o ranking por métrica interna e o ranking por ARI, dentro de cada representação e entre todas as configurações. Medimos também o *arrependimento*: ARI da melhor configuração menos ARI da configuração escolhida pela métrica interna.

- **A3 (QP3):** variação de ARI com e sem UMAP por algoritmo, desvio entre seeds e sensibilidade a n_neighbors.

- **A4 (qualitativa):** matrizes de confusão das melhores configurações, classes mais confundidas e visualizações 2D.

## 5. Limitações

- K-means, GMM e aglomerativo recebem o k verdadeiro, o que é uma vantagem irreal. A análise A2 indica se as métricas internas compensariam isso.

- A ResNet-50 foi treinada com rótulos do ImageNet, então entra como referência, não como método não supervisionado.

- DINOv2 e CLIP foram pré-treinados em grandes coleções da web que podem conter imagens parecidas com as dos datasets avaliados.

- Os resultados valem para imagens naturais e não se generalizam automaticamente para outros domínios.

## 6. Cronograma

| **Semana**         | **Entregas**                                                                     |
|--------------------|----------------------------------------------------------------------------------|
| 1 (9 a 15/nov)     | Repositório e pipeline, extração e cache das features, baseline K-means validado |
| 2 (16 a 22/nov)    | Grade completa, tabela única de resultados                                       |
| 3 (23 a 29/nov)    | Análises A1 a A4, figuras e ablações                                             |
| 4 (30/nov a 6/dez) | Escrita no formato de artigo e slides da apresentação                            |

## 7. Repositório

O código será público e servirá de portfólio. O pipeline tem quatro estágios com cache (extração, redução, clustering, avaliação), configurados por YAML. Os resultados ficam numa tabela única, com uma linha por execução, e as figuras são geradas por script a partir dela. Seeds fixas, ambiente travado e um comando único no README garantem a reprodução.

## Referências

1.  ADALOGLOU, N.; MICHELS, F.; KALISCH, H.; KOLLMANN, M. Exploring the Limits of Deep Image Clustering using Pretrained Models. *BMVC*, 2023.

2.  GADETSKY, A.; JIANG, Y.; BRBIĆ, M. Let Go of Your Labels with Unsupervised Transfer. *ICML*, 2024.

3.  OQUAB, M. et al. DINOv2: Learning Robust Visual Features without Supervision. *Transactions on Machine Learning Research*, 2024.

4.  RADFORD, A. et al. Learning Transferable Visual Models From Natural Language Supervision. *ICML*, 2021.

5.  McINNES, L.; HEALY, J.; MELVILLE, J. UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. arXiv:1802.03426, 2018.

6.  HERRMANN, M.; KAZEMPOUR, D.; SCHEIPL, F.; KRÖGER, P. Enhancing cluster analysis via topological manifold learning. *Data Mining and Knowledge Discovery*, v. 38, 2024.

7.  KOBAK, D.; LINDERMAN, G. C. Initialization is critical for preserving global data structure in both t-SNE and UMAP. *Nature Biotechnology*, v. 39, p. 156-157, 2021.

8.  CHARI, T.; PACHTER, L. The specious art of single-cell genomics. *PLoS Computational Biology*, v. 19, n. 8, e1011288, 2023.

9.  LAUSE, J.; BERENS, P.; KOBAK, D. The art of seeing the elephant in the room: 2D embeddings of single-cell data do make sense. *PLoS Computational Biology*, v. 20, n. 10, e1012403, 2024.

10. CAMPELLO, R. J. G. B.; MOULAVI, D.; SANDER, J. Density-Based Clustering Based on Hierarchical Density Estimates. *PAKDD*, 2013.

11. HUBERT, L.; ARABIE, P. Comparing partitions. *Journal of Classification*, v. 2, p. 193-218, 1985.

12. PETUKHOVA, A.; MATOS-CARVALHO, J. P.; FACHADA, N. Text clustering with large language model embeddings. arXiv:2403.15112, 2024.

13. KRIZHEVSKY, A. Learning Multiple Layers of Features from Tiny Images. Relatório técnico, University of Toronto, 2009.

14. COATES, A.; NG, A.; LEE, H. An Analysis of Single-Layer Networks in Unsupervised Feature Learning. *AISTATS*, 2011.

15. HE, K.; ZHANG, X.; REN, S.; SUN, J. Deep Residual Learning for Image Recognition. *CVPR*, 2016.

16. ROUSSEEUW, P. J. Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. *Journal of Computational and Applied Mathematics*, v. 20, p. 53-65, 1987.

17. DAVIES, D. L.; BOULDIN, D. W. A Cluster Separation Measure. *IEEE TPAMI*, v. 1, n. 2, p. 224-227, 1979.

18. CALIŃSKI, T.; HARABASZ, J. A dendrite method for cluster analysis. *Communications in Statistics*, v. 3, n. 1, p. 1-27, 1974.
