# RFM Clustering

## Objective

- To indicate the profitability of customers
- To group and focus studying on loyal and profitable customers

## Methodology

1. **RFM Matrix**: This step involves creating an RFM (Recency, Frequency, Monetary) matrix that summarizes customer behavior. The RFM matrix is used to quantify the value of a customer to the business. 
    - The recency score indicates how recently the customer made a purchase.
    - The frequency score indicates how often they make purchases.
    - The monetary score indicates how much money they spend.
2. **Hopkins Test**: This step involves performing a Hopkins test to validate the clustering. The Hopkins test is a statistical test used to determine whether a dataset is suitable for clustering analysis. In this step, a random sample of points is taken from the dataset and compared to a uniform distribution of points. If the dataset is suitable for clustering, the distance between the random points and the uniform distribution will be significantly different.
3. **Modeling**: This step involves clustering the customers based on the RFM matrix using the K-means algorithm. The goal is to group customers with similar behavior together. The number of clusters is determined using the elbow method or another clustering validation method.
4. **Model Evaluation**: This step involves evaluating the clustering model. The evaluation can be done using various metrics such as silhouette score, within-cluster sum of squares, or other cluster validation methods. The results of the evaluation can be used to fine-tune the model to improve its accuracy.

## Findings

- One group has high RFM score which is labeled as loyal customer.
    - The further studies would concentrate to observe this group of customer
- One group has a potential to be a loyal customer. because they have high monetary score but with low frequency score.
    - To intiate some promotion or survey to understand their preference to increase the retention rate.

## Data Processing

### RFM matrix

The code produces the RFM matrix, which summarizes customer behavior. It extracts the customer ID, revenue, invoice number, and invoice date from the dataset, then groups the data by customer ID.

For each customer, the sum of revenue, the number of unique invoice numbers, and the maximum invoice date would be obtained for "monetary", "frequency", and "recency" indicators. For recency, it would be subtracted by the maximum invoice date from the invoice date for each transaction.

The bottom and roof variables are used to filter out outliers. They are Boolean arrays that indicate whether the monetary and frequency values are greater than the 5th percentile and less than the 95th percentile, respectively.

| customerid | monetary | frequency | recency |
| --- | --- | --- | --- |
| 12747 | 4196.01 | 11 | 1 |
| 12749 | 4090.88 | 5 | 3 |
| 12820 | 942.34 | 4 | 2 |
| 12822 | 948.88 | 2 | 70 |
| 12823 | 1759.5 | 5 | 74 |

```python
# produce RFM matrix 
rfm_raw = dataset.get(['customerid', 'revenue', 'invoiceno', 'invoicedate'])
rfm_matrix = (rfm_raw
    .groupby(['customerid'])
    .agg(monetary = ('revenue', 'sum'),
        frequency = ('invoiceno', 'nunique'),
        recency = ('invoicedate', 'max'))
)       
rfm_matrix['recency'] = (rfm_matrix.recency - rfm_matrix.recency.max()).dt.days
bottom = (rfm_matrix.get(['monetary', 'frequency']) > rfm_matrix.get(['monetary', 'frequency']).quantile(0.05)).all(axis=1)
roof = (rfm_matrix < rfm_matrix.quantile(0.95)).all(axis=1)
rfm_matrix = rfm_matrix.loc[(bottom & roof), :]
```

### RFM Scoring

This step would create scores for each customer based on their recency, frequency, and monetary values with spliting into 7 bins. The resulting scores are added to the `rfm_matrix` dataframe as `r_score`, `f_score`, and `m_score`. This helps to quantify the value of a customer to the business and is used later in the clustering process.

| customerid | r_score | f_score | m_score |
| --- | --- | --- | --- |
| 12749 | 6 | 2 | 6 |
| 12822 | 2 | 0 | 3 |
| 12823 | 2 | 2 | 5 |
| 12827 | 6 | 0 | 1 |
| 12829 | 0 | 0 | 0 |

```python
qcut = 7
rfm_matrix = rfm_matrix.assign(
    r_score = pd.qcut(rfm_matrix.recency, qcut, duplicates='drop').cat.codes,
    f_score = pd.qcut(rfm_matrix.frequency, qcut, duplicates = 'drop').cat.codes,
    m_score = pd.qcut(rfm_matrix.monetary, qcut, duplicates = 'drop').cat.codes
)
```

### Clustering Testing

```python
#importing the required libraries
import numpy as np
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from random import sample
from numpy.random import uniform

# function to compute hopkins's statistic for the dataframe X
def hopkins_statistic(X):
    
    X=X.values  #convert dataframe to a numpy array
    sample_size = int(X.shape[0]*0.05) #0.05 (5%) based on paper by Lawson and Jures
    
    #a uniform random sample in the original data space
    X_uniform_random_sample = uniform(X.min(axis=0), X.max(axis=0) ,(sample_size , X.shape[1]))
    
    #a random sample of size sample_size from the original data X
    random_indices=sample(range(0, X.shape[0], 1), sample_size)
    X_sample = X[random_indices]
   
    
    #initialise unsupervised learner for implementing neighbor searches
    neigh = NearestNeighbors(n_neighbors=2)
    nbrs=neigh.fit(X)
    
    #u_distances = nearest neighbour distances from uniform random sample
    u_distances , u_indices = nbrs.kneighbors(X_uniform_random_sample , n_neighbors=2)
    u_distances = u_distances[: , 0] #distance to the first (nearest) neighbour
    
    #w_distances = nearest neighbour distances from a sample of points from original data X
    w_distances , w_indices = nbrs.kneighbors(X_sample , n_neighbors=2)
    #distance to the second nearest neighbour (as the first neighbour will be the point itself, with distance = 0)
    w_distances = w_distances[: , 1]
    
 
    
    u_sum = np.sum(u_distances)
    w_sum = np.sum(w_distances)
    
    #compute and return hopkins' statistic
    H = u_sum/ (u_sum + w_sum)
    return H

#compute hopkins statistic for the dataframe rfm_matrix
sum([hopkins_statistic(rfm_matrix.get(['r_score', 'f_score', 'm_score'])) for i in range(100)]) / 100
```

### Modeling (K-Means)

```python
# produce the elbow plot
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import silhouette_score 

# get the inertia of each k
train_data = rfm_matrix.get(['r_score', 'f_score', 'm_score'])
kmeans = lambda n_cluster: KMeans(n_clusters=n_cluster, random_state=123, n_init='auto').fit(train_data)
score_matrix = [silhouette_score(train_data, kmeans(n).labels_) for n in range(2, 11)]

# plot the elbow plot
plt.figure(figsize=(10, 5))
sns.lineplot(x=range(2, len(score_matrix)+2), y=score_matrix)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Plot')
plt.show()
```

![6b23f310-0b7b-4b96-898a-898f685326c2.png](RFM%20Clustering%20391e432790e84abfa775617e6ea7b54b/6b23f310-0b7b-4b96-898a-898f685326c2.png)

| cluster | r_score | f_score | m_score | cluster count |
| --- | --- | --- | --- | --- |
| 0 | 4.56214 | 2.99426 | 5.18547 | 523 |
| 1 | 1.11392 | 0.146474 | 0.886076 | 553 |
| 2 | 1.42675 | 0.906582 | 4.22718 | 471 |
| 3 | 4.62478 | 0.363002 | 2.03665 | 573 |

![0caffb44-ff69-4059-8df9-d49478fae305.png](RFM%20Clustering%20391e432790e84abfa775617e6ea7b54b/0caffb44-ff69-4059-8df9-d49478fae305.png)

![9f5104e4-6536-4c07-8c45-5c286ef15924.png](RFM%20Clustering%20391e432790e84abfa775617e6ea7b54b/9f5104e4-6536-4c07-8c45-5c286ef15924.png)

![6346d480-4d3a-46b1-8780-eb1f6f9ec3e8.png](RFM%20Clustering%20391e432790e84abfa775617e6ea7b54b/6346d480-4d3a-46b1-8780-eb1f6f9ec3e8.png)