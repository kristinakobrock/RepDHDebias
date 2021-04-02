# RepDHDebias
**Re-implementation of [Wang et al. (2020): "Double-Hard Debias: Tailoring Word Embeddings for Gender Bias Mitigation"](https://arxiv.org/abs/2005.00965)**  
This project is a re-implementation of Wang et al. (2020). It aimed to replicate the proposed **Double-Hard Debias** algorithm and to reproduce the evaluation results reported in the original paper.

## Requirements
Python 3

## Data
All required datasets are provided except for the Double-Hard debiased embedding obtained by the original authors (needed for the evaluation). Please download the file `glove_dhd.p` [here](www.cs.virginia.edu/~tw8cb/word_embeddings/) and save it to the folder `Code` without changing the file name.

## Code
Please find all code in this folder. 
* [`Debias_Glove.ipynb`:](Debias_Glove.ipynb) This notebook loads the required datasets and executes our implementation of **Double-Hard Debias**. The obtained Double-Hard debiased embedding is saved in  `debiased.zip`. The [pickle](https://docs.python.org/3/library/pickle.html) module is needed for reading it out.
* `Evaluations.ipynb`: This notebook loads the baseline datasets, the Double-Hard debiased embedding obtained by the original authors and the Double-Hard debiased embedding obtained by our implementation of the algorithm. The embeddings are evaluated based on standard word embedding tasks: **WEAT** ([Caliskan et al.](https://arxiv.org/abs/1608.07187)), the **Neighborhood Metric** ([Gonen & Goldberg 2019](https://arxiv.org/abs/1903.03862)) the **MSR analogy task** ([Mikolov et al. 2013a](https://www.aclweb.org/anthology/N13-1090/)) and the **Google word analogy task** ([Mikolov et al. 2013b](https://arxiv.org/abs/1301.3781v3)).
* `Frequency_in_Glove.ipynb`: Please view this notebook for our re-implementation of the pilot study conducted by Wang et al. (2020) in the **Motivation** part of their paper. It is designed to motivate the main change from Hard Debias to Double-Hard Debias, i.e. removing the frequency direction of the embedding additionally to removing its gender direction.

## Report
You can find the full report in this folder.
