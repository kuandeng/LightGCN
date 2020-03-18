# LightGCN
This is our Tensorflow implementation for the paper:

>Xiangnan He, Kuan Deng ,Xiang Wang, Yan Li, Yongdong Zhang, Meng Wang(2020). LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation, [Paper in arXiv](https://arxiv.org/abs/2002.02126).

Contributors: Kuan Deng, Yingxin Wu, Xiangnan He

## Introduction
In this work, we aim to simplify the design of GCN to make it more concise and appropriate for recommendation. We propose a new model named LightGCN,including only the most essential component in GCN—neighborhood aggregation—for collaborative filtering

## Environment Requirement
The code has been tested running under Python 3.6.5. The required packages are as follows:
* tensorflow == 1.11.0
* numpy == 1.14.3
* scipy == 1.1.0
* sklearn == 0.19.1

## Example to Run the Codes
The instruction of commands has been clearly stated in the codes (see the parser function in LightGCN/utility/parser.py).
* Gowalla dataset
```
python LightGCN.py --dataset gowalla --regs [1e-4] --embed_size 64 --layer_size [64,64,64,64] --lr 0.001 --batch_size 2048 --epoch 1000
```
* Yelp2018 dataset
```
python LightGCN.py --dataset yelp2018 --regs [1e-4] --embed_size 64 --layer_size [64,64,64,64] --lr 0.001 --batch_size 2048 --epoch 1000
```
* Amazon-book dataset
```
python LightGCN.py --dataset amazon-book --regs [1e-4] --embed_size 64 --layer_size [64,64,64,64] --lr 0.001 --batch_size 1024 --epoch 200 
```

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

## Improvement
parallelize sampling on CPU

=======


