'''
Created on Oct 10, 2018
Tensorflow Implementation of Neural Graph Collaborative Filtering (NGCF) model in:
Wang Xiang et al. Neural Graph Collaborative Filtering. In SIGIR 2019.

@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import numpy as np
import random as rd
import scipy.sparse as sp
from time import time

class business_model:
    def __init__(self, id, categories, price):
        self.id = id
        self.categories = categories
        self.price = price  

class Data(object):
    def __init__(self, path, batch_size):
        self.path = path
        self.batch_size = batch_size

        train_file = path + '/train.txt'
        test_file = path + '/test.txt'
        item_file = path + '/item_list.txt'

        self.n_users, self.n_items = 0, 0
        self.n_cat, self.n_price = 0, 0
        self.n_train, self.n_test = 0, 0
        self.neg_pools = {}
        self.price_list = []
        self.cat_list = []
        self.business_list = []

        self.exist_users = []
        #read item_list containing category and prices
        with open(item_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    if(l[-1] not in self.price_list and len(l) > 3): #Checks if price exists in list.
                        self.price_list.append(l[-1])
                        self.n_price += 1
                    self.item_categories = []
                    for i in range(2, len(l)): #The first two elements are item name and item id. The last i price
                        if(i != len(l) and l[i] not in self.cat_list):
                            self.cat_list.append(l[i])
                            self.n_cat += 1
                        self.item_categories.append(l[i])
                    self.business_list.append(business_model(l[1], self.item_categories, l[-1]))
        print('n_price: ' + str(self.n_price))
        print('n_cat: ' + str(self.n_cat))
        print('cat_list: ' + str(len(self.cat_list)))
        print('price_list: ' + str(len(self.price_list)))
        
        #read training file
        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    self.exist_users.append(uid)
                    self.n_items = max(self.n_items, max(items))
                    self.n_users = max(self.n_users, uid)
                    self.n_train += len(items)
        #read test file
        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')[1:]]
                    except Exception:
                        continue
                    self.n_items = max(self.n_items, max(items))
                    self.n_test += len(items)
        self.n_items += 1
        self.n_users += 1
        self.print_statistics()
		#matrix size num_users X num_items
        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        self.train_items, self.test_set = {}, {}
        with open(train_file) as f_train:
            with open(test_file) as f_test:
                for l in f_train.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    items = [int(i) for i in l.split(' ')]
                    uid, train_items = items[0], items[1:]

                    for i in train_items:
                        self.R[uid, i] = 1.
                        
                    self.train_items[uid] = train_items
                    
                for l in f_test.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')]
                    except Exception:
                        continue
                    
                    uid, test_items = items[0], items[1:]
                    self.test_set[uid] = test_items

    def get_adj_mat(self):
        try:
            t1 = time()
            adj_mat = sp.load_npz(self.path + '/s_adj_mat.npz')
            norm_adj_mat = sp.load_npz(self.path + '/s_norm_adj_mat.npz')
            mean_adj_mat = sp.load_npz(self.path + '/s_mean_adj_mat.npz')
            adj_mat_with_cp = sp.load_npz(self.path + '/s_adj_mat_with_cp.npz')
            print('already load adj matrix', adj_mat.shape, time() - t1)
        
        except Exception:
            adj_mat, norm_adj_mat, mean_adj_mat, adj_mat_with_cp = self.create_adj_mat()
            sp.save_npz(self.path + '/s_adj_mat.npz', adj_mat)
            sp.save_npz(self.path + '/s_norm_adj_mat.npz', norm_adj_mat)
            sp.save_npz(self.path + '/s_mean_adj_mat.npz', mean_adj_mat)
            sp.save_npz(self.path + '/s_adj_mat_with_cp.npz', adj_mat_with_cp)

        try:
            pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')
        except Exception:
            adj_mat=adj_mat
            rowsum = np.array(adj_mat.sum(1))
            d_inv = np.power(rowsum, -0.5).flatten()
            
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj = d_mat_inv.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat_inv)
            print('generate pre adjacency matrix.')
            pre_adj_mat = norm_adj.tocsr()
            sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)
            
        return adj_mat, norm_adj_mat, mean_adj_mat,pre_adj_mat, adj_mat_with_cp

    def create_adj_mat(self):
        t1 = time()
		#adj matrix = num_users+num_items X num_users+num_items
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.R.tolil()
        # prevent memory from overflowing
        # indexing works by [start_row:end_row+1 , start_col: EMPTY] -> EMPTY means rest of the coloumn in this instance
        for i in range(5):                                                                  # splits it in 5 parts and assigns part of the U x I matrix R to the adj matrix
            adj_mat[int(self.n_users*i/5.0):int(self.n_users*(i+1.0)/5), self.n_users:] =\
            R[int(self.n_users*i/5.0):int(self.n_users*(i+1.0)/5)]                          # Only iterates throuh users since items never interact
            adj_mat[self.n_users:,int(self.n_users*i/5.0):int(self.n_users*(i+1.0)/5)] =\
            R[int(self.n_users*i/5.0):int(self.n_users*(i+1.0)/5)].T                        # adj matrix is U+I x U+I size
        adj_mat = adj_mat.todok()
        print('already create adjacency matrix', adj_mat.shape, time() - t1)
        
        t2 = time()
        def adj_with_cat_and_price(adj): # Creates a U+I+C+P x U+I+C+P size matrix
            size_of_ui_matrix = self.n_users + self.n_items
            size_of_uic_matrix = self.n_users + self.n_items + self.n_cat
            size_of_uicp_matrix = self.n_users + self.n_items + self.n_price + self.n_cat
            adj_mat_with_cat_and_price = sp.dok_matrix((size_of_uicp_matrix, size_of_uicp_matrix), dtype=np.float32)
            adj_mat_with_cat_and_price = adj_mat_with_cat_and_price.tolil()
            print(adj_mat_with_cat_and_price._shape)

            for i in range (5):
                adj_mat_with_cat_and_price[int(self.n_users*i/5.0):int(self.n_users*(i+1)/5.0), self.n_users:size_of_ui_matrix] =\
                adj[int(self.n_users*i/5.0):int(self.n_users*(i+1)/5.0), self.n_users:]
                adj_mat_with_cat_and_price[self.n_users:size_of_ui_matrix,int(self.n_users*i/5.0):int(self.n_users*(i+1)/5.0)] =\
                adj[int(self.n_users*i/5.0):int(self.n_users*(i+1)/5.0), self.n_users:].T
                print("iteration: ", i)
            print(adj_mat_with_cat_and_price._shape)

            adj_connection_value = 1.0
            for row in range(0, self.n_items):      # Iterates through the category list of each item to find connections
                categories = self.business_list[row].categories
                
                for cat in categories:  # works for single and multiple categories pr item
                    index = 0
                    for all_cat in self.cat_list:
                        if(cat == all_cat):
                            adj_mat_with_cat_and_price[self.n_users + row, size_of_ui_matrix + index] = adj_connection_value # displace by n_users so that we give the correct item the category 
                            adj_mat_with_cat_and_price[size_of_ui_matrix + index, self.n_users + row] = adj_connection_value # Insures symmetry in adj matrix
                        index += 1
                price = self.business_list[row].price
                index = 0
                for p in self.price_list:
                    if p == price:
                        adj_mat_with_cat_and_price[self.n_users + row, size_of_uic_matrix + index] = adj_connection_value
                        adj_mat_with_cat_and_price[size_of_uic_matrix + index, self.n_users + row] = adj_connection_value

                        break
                    index += 1   
            return adj_mat_with_cat_and_price


        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        def check_adj_if_equal(adj):
            dense_A = np.array(adj.todense())
            degree = np.sum(dense_A, axis=1, keepdims=False)

            temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
            print('check normalized adjacency matrix whether equal to this laplacian matrix.')
            return temp
        
        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        mean_adj_mat = normalized_adj_single(adj_mat)
        adj_with_cp = adj_with_cat_and_price(adj_mat) 

        print('already normalize adjacency matrix', time() - t2)
        return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr(), adj_with_cp.tocsr()
        
    def negative_pool(self):
        t1 = time()
        for u in self.train_items.keys():
            neg_items = list(set(range(self.n_items)) - set(self.train_items[u]))
            pools = [rd.choice(neg_items) for _ in range(100)]
            self.neg_pools[u] = pools
        print('refresh negative pools', time() - t1)

    def sample(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]


        def sample_pos_items_for_u(u, num):
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_id = np.random.randint(low=0, high=self.n_items,size=1)[0]
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        def sample_neg_items_for_u_from_pools(u, num):
            neg_items = list(set(self.neg_pools[u]) - set(self.train_items[u]))
            return rd.sample(neg_items, num)

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items
    
    def sample_test(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.test_set.keys(), self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]

        def sample_pos_items_for_u(u, num):
            pos_items = self.test_set[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                if neg_id not in (self.test_set[u]+self.train_items[u]) and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items
    
        def sample_neg_items_for_u_from_pools(u, num):
            neg_items = list(set(self.neg_pools[u]) - set(self.train_items[u]))
            return rd.sample(neg_items, num)

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items
    
    
    
    
    
    
    def get_num_users_items(self):
        return self.n_users, self.n_items

    def print_statistics(self):
        print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        print('n_interactions=%d' % (self.n_train + self.n_test))
        print('n_train=%d, n_test=%d, sparsity=%.5f' % (self.n_train, self.n_test, (self.n_train + self.n_test)/(self.n_users * self.n_items)))


    def get_sparsity_split(self):
        try:
            split_uids, split_state = [], []
            lines = open(self.path + '/sparsity.split', 'r').readlines()

            for idx, line in enumerate(lines):
                if idx % 2 == 0:
                    split_state.append(line.strip())
                    print(line.strip())
                else:
                    split_uids.append([int(uid) for uid in line.strip().split(' ')])
            print('get sparsity split.')

        except Exception:
            split_uids, split_state = self.create_sparsity_split()
            f = open(self.path + '/sparsity.split', 'w')
            for idx in range(len(split_state)):
                f.write(split_state[idx] + '\n')
                f.write(' '.join([str(uid) for uid in split_uids[idx]]) + '\n')
            print('create sparsity split.')

        return split_uids, split_state



    def create_sparsity_split(self):
        all_users_to_test = list(self.test_set.keys())
        user_n_iid = dict()

        # generate a dictionary to store (key=n_iids, value=a list of uid).
        for uid in all_users_to_test:
            train_iids = self.train_items[uid]
            test_iids = self.test_set[uid]

            n_iids = len(train_iids) + len(test_iids)

            if n_iids not in user_n_iid.keys():
                user_n_iid[n_iids] = [uid]
            else:
                user_n_iid[n_iids].append(uid)
        split_uids = list()

        # split the whole user set into four subset.
        temp = []
        count = 1
        fold = 4
        n_count = (self.n_train + self.n_test)
        n_rates = 0

        split_state = []
        for idx, n_iids in enumerate(sorted(user_n_iid)):
            temp += user_n_iid[n_iids]
            n_rates += n_iids * len(user_n_iid[n_iids])
            n_count -= n_iids * len(user_n_iid[n_iids])

            if n_rates >= count * 0.25 * (self.n_train + self.n_test):
                split_uids.append(temp)

                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' %(n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)

                temp = []
                n_rates = 0
                fold -= 1

            if idx == len(user_n_iid.keys()) - 1 or n_count == 0:
                split_uids.append(temp)

                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' % (n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)



        return split_uids, split_state
