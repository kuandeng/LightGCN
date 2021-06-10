# LightGCN
This is our Tensorflow implementation for our SIGIR 2020 paper:

>Xiangnan He, Kuan Deng ,Xiang Wang, Yan Li, Yongdong Zhang, Meng Wang(2020). LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation, [Paper in arXiv](https://arxiv.org/abs/2002.02126).

Contributors: Dr. Xiangnan He (staff.ustc.edu.cn/~hexn/), Kuan Deng, Yingxin Wu.

(We also provide Pytorch implementation for LightGCN : https://github.com/gusye1234/LightGCN-PyTorch. Contributors: Jianbai Ye.)

## Introduction
In this work, we aim to simplify the design of GCN to make it more concise and appropriate for recommendation. We propose a new model named LightGCN, including only the most essential component in GCN—neighborhood aggregation—for collaborative filtering.

## Environment Requirement
The code has been tested running under Python 3.6.5. The required packages are as follows:
* tensorflow == 1.11.0
* numpy == 1.14.3
* scipy == 1.1.0
* sklearn == 0.19.1
* cython == 0.29.15
## C++ evaluator
We have implemented C++ code to output metrics during and after training, which is much more efficient than python evaluator. It needs to be compiled first using the following command. 
```
python setup.py build_ext --inplace
```
After compilation, the C++ code will run by default instead of Python code.

## Examples to run a 3-layer LightGCN
The instruction of commands has been clearly stated in the codes (see the parser function in LightGCN/utility/parser.py).
### Gowalla dataset
* Command
```
python LightGCN.py --dataset gowalla --regs [1e-4] --embed_size 64 --layer_size [64,64,64] --lr 0.001 --batch_size 2048 --epoch 1000
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
Epoch 879 [81.6s + 31.3s]: test==[0.13271=0.12645 + 0.00626 + 0.00000], recall=[0.18201], precision=[0.05601], ndcg=[0.15555]
Early stopping is trigger at step: 5 log:0.18201370537281036
Best Iter=[38]@[32829.6]	recall=[0.18236], precision=[0.05607], ndcg=[0.15539]
```


### Yelp2018 dataset
* Command
```
python LightGCN.py --dataset yelp2018 --regs [1e-4] --embed_size 64 --layer_size [64,64,64] --lr 0.001 --batch_size 2048 --epoch 1000
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
Epoch 679 [104.6s + 12.9s]: test==[0.17217=0.16289 + 0.00929 + 0.00000], recall=[0.06359], precision=[0.02874], ndcg=[0.05240]
Early stopping is trigger at step: 5 log:0.06359195709228516
Best Iter=[28]@[42815.0]	recall=[0.06367], precision=[0.02868], ndcg=[0.05236]
```
### Amazon-book dataset
* Command
```
python LightGCN.py --dataset amazon-book --regs [1e-4] --embed_size 64 --layer_size [64,64,64] --lr 0.001 --batch_size 8192 --epoch 1000
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
Epoch 779 [181.7s + 79.0s]: test==[0.20300=0.19434 + 0.00866 + 0.00000], recall=[0.04120], precision=[0.01703], ndcg=[0.03186]
Early stopping is trigger at step: 5 log:0.04119725897908211
Best Iter=[33]@[49875.4]	recall=[0.04123], precision=[0.01710], ndcg=[0.03189]
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

=======
