# LightGCN
This is our Tensorflow implementation for the paper:

>Xiangnan He, Kuan Deng ,Xiang Wang, Yan Li, Yongdong Zhang, Meng Wang(2020). LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation, [Paper in arXiv](https://arxiv.org/abs/2002.02126).

Contributors: Dr. Xiangnan He (staff.ustc.edu.cn/~hexn/), Kuan Deng, Yingxin Wu.

## Introduction
In this work, we aim to simplify the design of GCN to make it more concise and appropriate for recommendation. We propose a new model named LightGCN, including only the most essential component in GCN—neighborhood aggregation—for collaborative filtering

## Environment Requirement
The code has been tested running under Python 3.6.5. The required packages are as follows:
* tensorflow == 1.11.0
* numpy == 1.14.3
* scipy == 1.1.0
* sklearn == 0.19.1
* cython == 0.29.15
## C++ evaluator
We have implemented C++ code to output metrics during and after training, which is more efficient than python evaluator. Compile the code first with the following command.
```
python setup.py build_ext --inplace
```
After execution, the C++ code will run by default.

## Examples to run a 3-layer LightGCN
The instruction of commands has been clearly stated in the codes (see the parser function in LightGCN/utility/parser.py).
### Gowalla dataset
* Command
```
python LightGCN.py --dataset gowalla --regs [1e-4] --embed_size 64 --layer_size [64,64,64] --lr 0.001 --batch_size 4096 --epoch 1000
```
* Output log :
```
eval_score_matrix_foldout with cpp
n_users=29858, n_items=40981
n_interactions=1027370
n_train=810128, n_test=217242, sparsity=0.00084
      ...
Epoch 1 [30.3s]: train==[0.46925=0.46911 + 0.00014]
Epoch 2 [27.1s]: train==[0.21866=0.21817 + 0.00048]
      ...
Epoch 1000 [51.4s + 11.1s]: test==[0.13277=0.12649 + 0.00627 + 0.00000], recall=[0.18123], precision=[0.05618], ndcg=[0.15553]
Best Iter=[45]@[28563.4]	recall=[0.18198], precision=[0.05613], ndcg=[0.15523]
```


### Yelp2018 dataset
* Command
```
python LightGCN.py --dataset yelp2018 --regs [1e-4] --embed_size 64 --layer_size [64,64,64] --lr 0.001 --batch_size 4096 --epoch 1000
```
* Output log :
```
eval_score_matrix_foldout with cpp
n_users=31668, n_items=38048
n_interactions=1561406
n_train=1237259, n_test=324147, sparsity=0.00130
    ...
Epoch 1 [56.5s]: train==[0.33843=0.33815 + 0.00028]
Epoch 2 [53.1s]: train==[0.16253=0.16192 + 0.00061]
    ...
Epoch 840 [91.6s + 10.5s]: test==[0.17441=0.16502 + 0.00939 + 0.00000], recall=[0.06398], precision=[0.02880], ndcg=[0.05265]
Early stopping is trigger at step: 5 log:0.06398
Best Iter=[36]@[47231.3]	recall=[0.06446], precision=[0.02890], ndcg=[0.05279]
```
### Amazon-book dataset
* Command
```
python LightGCN.py --dataset amazon-book --regs [1e-4] --embed_size 64 --layer_size [64,64,64] --lr 0.001 --batch_size 4096 --epoch 1000
```
* Output log :
```
eval_score_matrix_foldout with cpp
n_users=52643, n_items=91599
n_interactions=2984108
n_train=2380730, n_test=603378, sparsity=0.00062
    ...
Epoch 1 [53.2s]: train==[0.57471=0.57463 + 0.00008]
Epoch 2 [47.3s]: train==[0.31518=0.31478 + 0.00040]
    ...
Epoch 1000 [123.4s + 20.6s]: test==[0.20296=0.19421 + 0.00875 + 0.00000], recall=[0.04171], precision=[0.01725], ndcg=[0.03224]
Best Iter=[49]@[51723.6]	recall=[0.04171], precision=[0.01725], ndcg=[0.03224]
```
NOTE : the duration of training and testing depends on the running environment.
## Dataset
We provide three processed datasets: Gowalla, Yelp2018 and Amazon-book.
* `train.txt`
  * Train file.
  * Each line is a user with her/his positive interactions with items: userID\t a list of itemID\n.

* `test.txt`
  * Test file (positive instances).
  * Each line is a user with her/his positive interactions with items: userID\t a list of itemID\n.
  * Note that here we treat all unobserved interactions as the negative instances when reporting performance.
  
* `user_list.txt`
  * User file.
  * Each line is a triplet (org_id, remap_id) for one user, where org_id and remap_id represent the ID of the user in the original and our datasets, respectively.
  
* `item_list.txt`
  * Item file.
  * Each line is a triplet (org_id, remap_id) for one item, where org_id and remap_id represent the ID of the item in the original and our datasets, respectively.

## Efficiency Improvements:
  * Parallelized sampling on CPU
  * C++ evaluation for top-k recommendation

