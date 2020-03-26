'''
Created on Oct 10, 2018
Tensorflow Implementation of Neural Graph Collaborative Filtering (NGCF) model in:
Wang Xiang et al. Neural Graph Collaborative Filtering. In SIGIR 2019.

@author: Xiang Wang (xiangwang@u.nus.edu)
'''
from utility.parser import parse_args
from utility.load_data import *
from evaluator import eval_score_matrix_foldout
import multiprocessing
import heapq
import numpy as np
cores = multiprocessing.cpu_count() // 2

args = parse_args()

data_generator = Data(path=args.data_path + args.dataset, batch_size=args.batch_size)
USR_NUM, ITEM_NUM = data_generator.n_users, data_generator.n_items
N_TRAIN, N_TEST = data_generator.n_train, data_generator.n_test

BATCH_SIZE = args.batch_size


def test(sess, model, users_to_test, drop_flag=False, train_set_flag=0):
    # B: batch size
    # N: the number of items
    top_show = np.sort(model.Ks)
    max_top = max(top_show)
    result = {'precision': np.zeros(len(model.Ks)), 'recall': np.zeros(len(model.Ks)), 'ndcg': np.zeros(len(model.Ks))}

    u_batch_size = BATCH_SIZE

    test_users = users_to_test
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1
    
    count = 0
    all_result = []
    item_batch = range(ITEM_NUM)
    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_batch = test_users[start: end]
        if drop_flag == False:
            rate_batch = sess.run(model.batch_ratings, {model.users: user_batch,
                                                        model.pos_items: item_batch})
        else:
            rate_batch = sess.run(model.batch_ratings, {model.users: user_batch,
                                                        model.pos_items: item_batch,
                                                        model.node_dropout: [0.] * len(eval(args.layer_size)),
                                                        model.mess_dropout: [0.] * len(eval(args.layer_size))})
        rate_batch = np.array(rate_batch)# (B, N)
        test_items = []
        if train_set_flag == 0:
            for user in user_batch:
                test_items.append(data_generator.test_set[user])# (B, #test_items)
                
            # set the ranking scores of training items to -inf,
            # then the training items will be sorted at the end of the ranking list.    
            for idx, user in enumerate(user_batch):
                    train_items_off = data_generator.train_items[user]
                    rate_batch[idx][train_items_off] = -np.inf
        else:
            for user in user_batch:
                test_items.append(data_generator.train_items[user])
        
        batch_result = eval_score_matrix_foldout(rate_batch, test_items, max_top)#(B,k*metric_num), max_top= 20
        count += len(batch_result)
        all_result.append(batch_result)
        
    
    assert count == n_test_users
    all_result = np.concatenate(all_result, axis=0)
    final_result = np.mean(all_result, axis=0)  # mean
    final_result = np.reshape(final_result, newshape=[5, max_top])
    final_result = final_result[:, top_show-1]
    final_result = np.reshape(final_result, newshape=[5, len(top_show)])
    result['precision'] += final_result[0]
    result['recall'] += final_result[1]
    result['ndcg'] += final_result[3]
    return result
               
            








