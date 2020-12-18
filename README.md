This fork is used for our pre specialication project where we experimented with extending the LightGCN recommendation framework with price and category features.
The repository is forked from https://github.com/kuandeng/LightGCN.
These changes can be ran by runnning LightGCN with the parameters --alg_type pas --adj_type adj_with_cp or  --alg_type ngcfpas --adj_type adj_with_cp.
PAS stands for price aware simple and is a simple extension that adds categories and prices to the adjacency matrix and embeddings.
