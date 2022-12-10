# Fashion-MNIST
I would like to share my ML project on the [fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset. Further informations on the dataset can be found via [Zalando Research](https://github.com/zalandoresearch/fashion-mnist) and [Kaggle](https://www.kaggle.com/zalando-research/fashionmnist)

## Data Exploration
Fashion-MNIST is a balanced dataset of 10 classes (labels), consiting of 60,000 images as the training set and 6,000 images as the testing set. 

| Label | Description |
| --- | --- |
|  0  | T-shirt/top |
|  1  | Trouser |
|  2  | Pullover |
|  3  | Dress |
|  4  | Coat |
|  5  | Sandal |
|  6  | Shirt |
|  7  | Sneaker |
|  8  | Bag |
|  9  | Ankle boot |

Each image is a 28x28 pixel grayscale image, which can be directly flatten to 784 features.
<img src="./figures/data_visualization.png">

Data distribution of 10 randomly-selected data were ploted, showing the non-gaussian distributions. Furthermore, as shown in the [notebook](), the data fails the normality test (p-values = 0), confirming the non-gaussian behaviour.
<img src="./figures/data_distributions.png">

## Clustering

### 2D t-SNE
<img src="./figures/2D-t-SNT_tune_perplexity.png">

### 2D UMAP
<img src="./figures/2D-UMAP_tune_n_neighbors.png">
<img src="./figures/2D-UMAP_tune_min_dist">

### Comparison of t-SNE and UMAP
<img src="./figures/2D-T-SNE_vs_2D-UMAP.png">

### Manifold Learning: UMAP Clustering with Trustworthiness Scores
<img src="./figures/manifold_learning_2D-UMAP.png">

## Classification

### Logistic Regression
<img src="./figures/confusion_matrix_Logistic.png">

### XGBoost
<img src="./figures/confusion_matrix_XGBoost.png">

### CatBoost
<img src="./figures/confusion_matrix_CatBoost.png">

## Requirements
- TensorFlow
- Scikit-learn
- cuDF - GPU DataFrames
- cuML - GPU Machine Learning Algorithms

## References
[1] [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)<br>
[2] [RAPIDS-cuML](https://github.com/rapidsai/cuml)
[3] [scikit-learn](https://github.com/scikit-learn/scikit-learn)
[5] [UMAP](https://github.com/lmcinnes/umap)
[4] [XGBoost](https://github.com/dmlc/xgboost)
[4] [CatBoost](https://github.com/catboost)

## Author
Kanokkorn Pimcharoen
