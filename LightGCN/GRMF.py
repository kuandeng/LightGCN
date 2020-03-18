'''
Created on Oct 10, 2018
Tensorflow Implementation of Neural Graph Collaborative Filtering (NGCF) model in:
Wang Xiang et al. Neural Graph Collaborative Filtering. In SIGIR 2019.

@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import tensorflow as tf
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from utility.helper import *
from utility.batch_test import *

class NGCF(object):
    def __init__(self, data_config, pretrain_data):
        # argument settings
        self.model_type = 'GRMF'
        self.adj_type = args.adj_type
        self.alg_type = args.alg_type

        self.pretrain_data = pretrain_data
        self.degree=data_config['degree']
        self.degree_k2=data_config['degree_k2']
        self.degree_norm=tf.sqrt(tf.constant(self.degree))
        self.degree_norm_L1=tf.constant(self.degree)
        self.degree_norm_L1_k2=tf.constant(self.degree_k2)
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']

        self.n_fold = 100

        self.norm_adj = data_config['norm_adj']
        self.plain_adj = data_config['plain_adj']
        self.n_nonzero_elems = self.norm_adj.count_nonzero()

        self.lr = args.lr

        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size

        self.weight_size = eval(args.layer_size)
        self.n_layers = len(self.weight_size)

        self.is_norm=args.is_norm
        self.regs = eval(args.regs)
        self.decay = self.regs[0]
        self.g_decay=self.regs[1]
        self.log_dir=self.create_model_str()
        self.verbose = args.verbose

        '''
        *********************************************************
        Create Placeholder for Input Data & Dropout.
        '''
        # placeholder definition
        self.users = tf.placeholder(tf.int32, shape=(None,))
        self.pos_items = tf.placeholder(tf.int32, shape=(None,))
        self.neg_items = tf.placeholder(tf.int32, shape=(None,))

        # dropout: node dropout (adopted on the ego-networks);
        #          ... since the usage of node dropout have higher computational cost,
        #          ... please use the 'node_dropout_flag' to indicate whether use such technique.
        #          message dropout (adopted on the convolution operations).
        self.node_dropout_flag = args.node_dropout_flag
        self.node_dropout = tf.placeholder(tf.float32, shape=[None])
        self.mess_dropout = tf.placeholder(tf.float32, shape=[None])
        
        
        
        with tf.name_scope('TRAIN_LOSS'):
            self.train_loss = tf.placeholder(tf.float32)
            tf.summary.scalar('train_loss', self.train_loss)
            self.train_mf_loss = tf.placeholder(tf.float32)
            tf.summary.scalar('train_mf_loss', self.train_mf_loss)
            self.train_emb_loss = tf.placeholder(tf.float32)
            tf.summary.scalar('train_emb_loss', self.train_emb_loss)
            self.train_reg_loss = tf.placeholder(tf.float32)
            tf.summary.scalar('train_reg_loss', self.train_reg_loss)
        self.merged_train_loss = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, 'TRAIN_LOSS'))
        
        
        with tf.name_scope('TRAIN_ACC'):
            self.train_rec20 = tf.placeholder(tf.float32)
            tf.summary.scalar('train_rec20', self.train_rec20)
            self.train_rec100 = tf.placeholder(tf.float32)
            tf.summary.scalar('train_rec100', self.train_rec100)
            self.train_ndcg20 = tf.placeholder(tf.float32)
            tf.summary.scalar('train_ndcg20', self.train_ndcg20)
            self.train_ndcg100 = tf.placeholder(tf.float32)
            tf.summary.scalar('train_ndcg100', self.train_ndcg100)
        self.merged_train_acc = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, 'TRAIN_ACC'))

        with tf.name_scope('TEST_LOSS'):
            self.test_loss = tf.placeholder(tf.float32)
            tf.summary.scalar('test_loss', self.test_loss)
            self.test_mf_loss = tf.placeholder(tf.float32)
            tf.summary.scalar('test_mf_loss', self.test_mf_loss)
            self.test_emb_loss = tf.placeholder(tf.float32)
            tf.summary.scalar('test_emb_loss', self.test_emb_loss)
            self.test_reg_loss = tf.placeholder(tf.float32)
            tf.summary.scalar('test_reg_loss', self.test_reg_loss)
        self.merged_test_loss = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, 'TEST_LOSS'))

        with tf.name_scope('TEST_ACC'):
            self.test_rec20 = tf.placeholder(tf.float32)
            tf.summary.scalar('test_rec20', self.test_rec20)
            self.test_rec100 = tf.placeholder(tf.float32)
            tf.summary.scalar('test_rec100', self.test_rec100)
            self.test_ndcg20 = tf.placeholder(tf.float32)
            tf.summary.scalar('test_ndcg20', self.test_ndcg20)
            self.test_ndcg100 = tf.placeholder(tf.float32)
            tf.summary.scalar('test_ndcg100', self.test_ndcg100)
        self.merged_test_acc = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, 'TEST_ACC'))
        """
        *********************************************************
        Create Model Parameters (i.e., Initialize Weights).
        """
        # initialization of model parameters
        self.weights = self._init_weights()

        """
        *********************************************************
        Compute Graph-based Representations of all users & items via Message-Passing Mechanism of Graph Neural Networks.
        Different Convolutional Layers:
            1. ngcf: defined in 'Neural Graph Collaborative Filtering', SIGIR2019;
            2. gcn:  defined in 'Semi-Supervised Classification with Graph Convolutional Networks', ICLR2018;
            3. gcmc: defined in 'Graph Convolutional Matrix Completion', KDD2018;
        """
        if self.alg_type in ['ngcf']:
            self.ua_embeddings, self.ia_embeddings = self._create_ngcf_embed()

        elif self.alg_type in ['gcn']:
            self.ua_embeddings, self.ia_embeddings = self._create_gcn_embed()

        elif self.alg_type in ['gcmc']:
            self.ua_embeddings, self.ia_embeddings = self._create_gcmc_embed()

        """
        *********************************************************
        Establish the final representations for user-item pairs in batch.
        """
        self.u_g_embeddings = tf.nn.embedding_lookup(self.ua_embeddings, self.users)
        self.pos_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.pos_items)
        self.neg_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.neg_items)

        """
        *********************************************************
        Inference for the testing phase.
        """
        self.batch_ratings = tf.matmul(self.u_g_embeddings, self.pos_i_g_embeddings, transpose_a=False, transpose_b=True)

        """
        *********************************************************
        Generate Predictions & Optimize via BPR loss.
        """
        self.mf_loss, self.emb_loss, self.reg_loss = self.create_bpr_loss(self.u_g_embeddings,
                                                                          self.pos_i_g_embeddings,
                                                                          self.neg_i_g_embeddings)
        self.loss = self.mf_loss + self.emb_loss+self.reg_loss

        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
        self.Laplace_cos=self.create_Laplace_norm()
        self.Laplace_pre=self.create_Laplace_norm1()
        self.Laplace_norm=self.create_Laplace_norm2()
        self.Laplace_pre_k2=self.create_Laplace_norm3()
        self.Laplace_cos_k2_u,self.Laplace_cos_k2_i=self.create_Laplace_norm4()
    def create_model_str(self):
        str1 = '/'+args.dataset
        str1 +='/is_norm_'+str(args.is_norm)+'/lr_' + str(self.lr) + '/reg_' + str(self.decay)+'/g_reg_'+str(self.g_decay)
        return str1


    def _init_weights(self):
        all_weights = dict()

        initializer = tf.contrib.layers.xavier_initializer()

        if self.pretrain_data is None:
            all_weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]), name='user_embedding')
            all_weights['item_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]), name='item_embedding')
            print('using xavier initialization')
        else:
            all_weights['user_embedding'] = tf.Variable(initial_value=self.pretrain_data['user_embed'], trainable=True,
                                                        name='user_embedding', dtype=tf.float32)
            all_weights['item_embedding'] = tf.Variable(initial_value=self.pretrain_data['item_embed'], trainable=True,
                                                        name='item_embedding', dtype=tf.float32)
            print('using pretrained initialization')

        self.weight_size_list = [self.emb_dim] + self.weight_size

        for k in range(self.n_layers):
            all_weights['W_gc_%d' %k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k+1]]), name='W_gc_%d' % k)
            all_weights['b_gc_%d' %k] = tf.Variable(
                initializer([1, self.weight_size_list[k+1]]), name='b_gc_%d' % k)

            all_weights['W_bi_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_bi_%d' % k)
            all_weights['b_bi_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_bi_%d' % k)

            all_weights['W_mlp_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k+1]]), name='W_mlp_%d' % k)
            all_weights['b_mlp_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k+1]]), name='b_mlp_%d' % k)

        return all_weights

    def _split_A_hat(self, X):
        A_fold_hat = []

        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold -1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat

    def _split_A_hat_node_dropout(self, X):
        A_fold_hat = []

        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold -1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len

            # A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
            temp = self._convert_sp_mat_to_sp_tensor(X[start:end])
            n_nonzero_temp = X[start:end].count_nonzero()
            A_fold_hat.append(self._dropout_sparse(temp, 1 - self.node_dropout[0], n_nonzero_temp))

        return A_fold_hat

    def _create_ngcf_embed(self):
        # Generate a set of adjacency sub-matrix.
        
        return self.weights['user_embedding'], self.weights['item_embedding']

    def _create_gcn_embed(self):
        A_fold_hat = self._split_A_hat(self.norm_adj)
        embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)


        all_embeddings = [embeddings]

        for k in range(0, self.n_layers):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], embeddings))

            embeddings = tf.concat(temp_embed, 0)
            embeddings = tf.nn.leaky_relu(tf.matmul(embeddings, self.weights['W_gc_%d' %k]) + self.weights['b_gc_%d' %k])
            embeddings = tf.nn.dropout(embeddings, 1 - self.mess_dropout[k])

            all_embeddings += [embeddings]

        all_embeddings = tf.concat(all_embeddings, 1)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings

    def _create_gcmc_embed(self):
        A_fold_hat = self._split_A_hat(self.norm_adj)

        embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)

        all_embeddings = []

        for k in range(0, self.n_layers):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], embeddings))
            embeddings = tf.concat(temp_embed, 0)
            # convolutional layer.
            embeddings = tf.nn.leaky_relu(tf.matmul(embeddings, self.weights['W_gc_%d' % k]) + self.weights['b_gc_%d' % k])
            # dense layer.
            mlp_embeddings = tf.matmul(embeddings, self.weights['W_mlp_%d' %k]) + self.weights['b_mlp_%d' %k]
            mlp_embeddings = tf.nn.dropout(mlp_embeddings, 1 - self.mess_dropout[k])

            all_embeddings += [mlp_embeddings]
        all_embeddings = tf.concat(all_embeddings, 1)

        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings

    
    def create_Laplace_norm(self):
        A_fold_hat = self._split_A_hat(self.plain_adj)
        self.ua_embeddings, self.ia_embeddings = self._create_ngcf_embed()
        embedding=tf.concat([self.ua_embeddings,self.ia_embeddings],0)
        norm_embedding=tf.nn.l2_normalize(embedding, axis=1)
        temp_embed = []
        for f in range(self.n_fold):
            temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f],norm_embedding))

        # sum messages of neighbors.
        temp = tf.concat(temp_embed, 0)
        temp=tf.multiply(temp,norm_embedding)
        temp=tf.reduce_sum(tf.reduce_sum(temp,axis=1,keepdims=False))
        temp1=tf.constant(self.n_nonzero_elems,dtype=tf.float32)
        return (temp1-temp)/temp1
        
    
    def create_Laplace_norm1(self):
        A_fold_hat = self._split_A_hat(self.plain_adj)
        self.ua_embeddings, self.ia_embeddings = self._create_ngcf_embed()
        embedding=tf.concat([self.ua_embeddings,self.ia_embeddings],0)
        temp_embed = []
        for f in range(self.n_fold):
            temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f],embedding))

        # sum messages of neighbors.
        temp = tf.concat(temp_embed, 0)
        temp=tf.multiply(temp,embedding)
        temp=tf.reduce_sum(temp)
        temp1=tf.multiply(tf.multiply(embedding,embedding),self.degree_norm_L1)
        temp1=tf.reduce_sum(temp1)
        return (temp1-temp)/temp1
    
    def create_Laplace_norm2(self):
        A_fold_hat = self._split_A_hat(self.norm_adj)
        self.ua_embeddings, self.ia_embeddings = self._create_ngcf_embed()
        embedding=tf.concat([self.ua_embeddings,self.ia_embeddings],0)
        temp_embed = []
        for f in range(self.n_fold):
            temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f],embedding))

        # sum messages of neighbors.
        temp = tf.concat(temp_embed, 0)
        temp=tf.multiply(temp,embedding)
        temp=tf.reduce_sum(temp)
        temp1=tf.multiply(embedding,embedding)
        temp1=tf.reduce_sum(temp1)
        return (temp1-temp)/temp1
    
    
    def create_Laplace_norm3(self):
        A_fold_hat = self._split_A_hat(self.norm_adj)
        self.ua_embeddings, self.ia_embeddings = self._create_ngcf_embed()
        embedding=tf.concat([self.ua_embeddings,self.ia_embeddings],0)
        temp=embedding
        for i in range(2):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f],temp))
            temp = tf.concat(temp_embed, 0)
        temp=tf.multiply(temp,embedding)
        temp=tf.reduce_sum(temp)
        temp1=tf.multiply(tf.multiply(embedding,embedding),self.degree_norm_L1_k2)
        temp1=tf.reduce_sum(temp1)
        return (temp1-temp)/temp1
    
    def create_Laplace_norm4(self):
        A_fold_hat = self._split_A_hat(self.norm_adj)
        self.ua_embeddings, self.ia_embeddings = self._create_ngcf_embed()
        embedding=tf.nn.l2_normalize(tf.concat([self.ua_embeddings,self.ia_embeddings],0),axis=1)
        temp=embedding
        for i in range(2):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f],temp))
            temp = tf.concat(temp_embed, 0)
        temp=tf.multiply(temp,embedding)
        temp_u,temp_i=tf.split(temp, [self.n_users, self.n_items], 0)
        temp_u=tf.reduce_sum(temp_u)
        temp_i=tf.reduce_sum(temp_i)
        temp1=tf.multiply(tf.multiply(embedding,embedding),self.degree_norm_L1_k2)
        temp1_u,temp1_i=tf.split(temp1, [self.n_users, self.n_items], 0)
        temp1_u=tf.reduce_sum(temp1_u)
        temp1_i=tf.reduce_sum(temp1_i)
        return (temp1_u-temp_u),temp1_i-temp_i
    
    
    
    
    
    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)

        regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items)
        regularizer = regularizer/self.batch_size
        
        # In the first version, we implement the bpr loss via the following codes:
        # We report the performance in our paper using this implementation.
#         maxi = tf.log(tf.nn.sigmoid(pos_scores - neg_scores))
#         mf_loss = tf.negative(tf.reduce_mean(maxi))
        
        ## In the second version, we implement the bpr loss via the following codes to avoid 'NAN' loss during training:
        ## However, it will change the training performance and training performance.
        ## Please retrain the model and do a grid search for the best experimental setting.
        mf_loss = tf.reduce_mean(tf.nn.softplus(-(pos_scores - neg_scores)))
        

        emb_loss = self.decay * regularizer
        if self.is_norm==0:
            reg_loss = self.g_decay*tf.reduce_mean(tf.reduce_sum(tf.square(users-pos_items),axis=1,keepdims=False))
        else:
            norm1=tf.nn.embedding_lookup(self.degree_norm,self.users)
            norm2=tf.nn.embedding_lookup(self.degree_norm,self.pos_items+self.n_users)
            temp1=tf.divide(users,norm1)
            temp2=tf.divide(pos_items,norm2)
            reg_loss = self.g_decay*tf.reduce_mean(tf.reduce_sum(tf.square(temp1-temp2),axis=1,keepdims=False))
        
        return mf_loss, emb_loss, reg_loss

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def _dropout_sparse(self, X, keep_prob, n_nonzero_elems):
        """
        Dropout for sparse tensors.
        """
        noise_shape = [n_nonzero_elems]
        random_tensor = keep_prob
        random_tensor += tf.random_uniform(noise_shape)
        dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
        pre_out = tf.sparse_retain(X, dropout_mask)

        return pre_out * tf.div(1., keep_prob)

def load_pretrained_data():
    pretrain_path = '%spretrain/%s/%s.npz' % (args.proj_path, args.dataset, 'embedding')
    try:
        pretrain_data = np.load(pretrain_path)
        print('load the pretrained embeddings.')
    except Exception:
        pretrain_data = None
    return pretrain_data

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items

    """
    *********************************************************
    Generate the Laplacian matrix, where each entry defines the decay factor (e.g., p_ui) between two connected nodes.
    """
    plain_adj, norm_adj, mean_adj,pre_adj = data_generator.get_adj_mat()
    config['plain_adj'] = plain_adj
    degree = np.array(plain_adj.sum(1))
    print(degree)
    print(degree.shape)
    config['degree'] = degree
    config['degree_k2']=np.array(pre_adj.sum(1))
    if args.adj_type == 'plain':
        config['norm_adj'] = plain_adj
        print('use the plain adjacency matrix')

    elif args.adj_type == 'norm':
        config['norm_adj'] = norm_adj
        print('use the normalized adjacency matrix')
    
    elif args.adj_type == 'gcmc':
        config['norm_adj'] = mean_adj
        print('use the gcmc adjacency matrix')
    elif args.adj_type=='ngcf_pre':
        config['norm_adj']=ngcf_pre_adj
        print('use the ngcf_pre adjcency matrix')
    elif args.adj_type=='pre':
        config['norm_adj']=pre_adj
        print('use the pre adjcency matrix')


    else:
        config['norm_adj'] = mean_adj + sp.eye(mean_adj.shape[0])
        print('use the mean adjacency matrix')
    t0 = time()

    if args.pretrain == -1:
        pretrain_data = load_pretrained_data()
    else:
        pretrain_data = None

    model = NGCF(data_config=config, pretrain_data=pretrain_data)

    """
    *********************************************************
    Save the model parameters.
    """
    saver = tf.train.Saver()

    if args.save_flag == 1:
        layer = '-'.join([str(l) for l in eval(args.layer_size)])
        weights_save_path = '%sweights/%s/%s/%s/l%s_r%s' % (args.weights_path, args.dataset, model.model_type, layer,
                                                            str(args.lr), '-'.join([str(r) for r in eval(args.regs)]))
        ensureDir(weights_save_path)
        save_saver = tf.train.Saver(max_to_keep=1)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    """
    *********************************************************
    Reload the pretrained model parameters.
    """
    if args.pretrain == 1:
        layer = '-'.join([str(l) for l in eval(args.layer_size)])

        pretrain_path = '%sweights/%s/%s/%s/l%s_r%s' % (args.weights_path, args.dataset, model.model_type, layer,
                                                        str(args.lr), '-'.join([str(r) for r in eval(args.regs)]))


        ckpt = tf.train.get_checkpoint_state(os.path.dirname(pretrain_path + '/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('load the pretrained model parameters from: ', pretrain_path)

            # *********************************************************
            # get the performance from pretrained model.
            if args.report != 1:
                users_to_test = list(data_generator.test_set.keys())
                ret = test(sess, model, users_to_test, drop_flag=True)
                cur_best_pre_0 = ret['recall'][0]

                pretrain_ret = 'pretrained model recall=[%.5f, %.5f], precision=[%.5f, %.5f], hit=[%.5f, %.5f],' \
                               'ndcg=[%.5f, %.5f]' % \
                               (ret['recall'][0], ret['recall'][-1],
                                ret['precision'][0], ret['precision'][-1],
                                ret['hit_ratio'][0], ret['hit_ratio'][-1],
                                ret['ndcg'][0], ret['ndcg'][-1])
                print(pretrain_ret)
        else:
            sess.run(tf.global_variables_initializer())
            cur_best_pre_0 = 0.
            print('without pretraining.')

    else:
        sess.run(tf.global_variables_initializer())
        cur_best_pre_0 = 0.
        print('without pretraining.')

    """
    *********************************************************
    Get the performance w.r.t. different sparsity levels.
    """
    if args.report == 1:
        assert args.test_flag == 'full'
        users_to_test_list, split_state = data_generator.get_sparsity_split()
        users_to_test_list.append(list(data_generator.test_set.keys()))
        split_state.append('all')

        report_path = '%sreport/%s/%s.result' % (args.proj_path, args.dataset, model.model_type)
        ensureDir(report_path)
        f = open(report_path, 'w')
        f.write(
            'embed_size=%d, lr=%.4f, layer_size=%s, keep_prob=%s, regs=%s, loss_type=%s, adj_type=%s\n'
            % (args.embed_size, args.lr, args.layer_size, args.keep_prob, args.regs, args.loss_type, args.adj_type))

        for i, users_to_test in enumerate(users_to_test_list):
            ret = test(sess, model, users_to_test, drop_flag=True)

            final_perf = "recall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
                         ('\t'.join(['%.5f' % r for r in ret['recall']]),
                          '\t'.join(['%.5f' % r for r in ret['precision']]),
                          '\t'.join(['%.5f' % r for r in ret['hit_ratio']]),
                          '\t'.join(['%.5f' % r for r in ret['ndcg']]))
            print(final_perf)

            f.write('\t%s\n\t%s\n' % (split_state[i], final_perf))
        f.close()
        exit()

    """
    *********************************************************
    Train.
    """
    tensorboard_model_path = './tensorboard_1013'+ '/' +model.model_type
    if not os.path.exists(tensorboard_model_path):
        os.makedirs(tensorboard_model_path)
    run_time = 1
    while (True):
        if os.path.exists(tensorboard_model_path +model.log_dir+'/run_' + str(run_time)):
            run_time += 1
        else:
            break
    train_writer = tf.summary.FileWriter(tensorboard_model_path + model.log_dir + '/run_' + str(run_time), sess.graph)
    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    stopping_step = 0
    should_stop = False
    laplace_cos,laplace_pre,laplace_norm=sess.run([model.Laplace_cos,model.Laplace_pre,model.Laplace_norm])
    print('laplace_cos: %.5f , laplace_pre: %.5f , laplace_norm: %.5f'%(laplace_cos,laplace_pre,laplace_norm))
    print(sess.run(model.Laplace_pre_k2))
    print(sess.run([model.Laplace_cos_k2_u,model.Laplace_cos_k2_i]))
#     users_to_test = list(data_generator.test_set.keys())
#     ret = test(sess, model, users_to_test, drop_flag=True)
#     summary_test_acc = sess.run(model.merged_test_acc,
#                                 feed_dict={model.test_rec20: ret['recall'][0], model.test_rec100: ret['recall'][-1],
#                                            model.test_ndcg20: ret['ndcg'][0], model.test_ndcg100: ret['ndcg'][-1]})
#     train_writer.add_summary(summary_test_acc, 0)
#     print(111)

    for epoch in range(args.epoch):
        t1 = time()
        loss, mf_loss, emb_loss, reg_loss = 0., 0., 0., 0.
        n_batch = data_generator.n_train // args.batch_size + 1
        loss_test,mf_loss_test,emb_loss_test,reg_loss_test=0.,0.,0.,0.
        for idx in range(n_batch):
            users, pos_items, neg_items = data_generator.sample()
            _, batch_loss, batch_mf_loss, batch_emb_loss, batch_reg_loss = sess.run([model.opt, model.loss, model.mf_loss, model.emb_loss, model.reg_loss],
                               feed_dict={model.users: users, model.pos_items: pos_items,
                                          model.node_dropout: eval(args.node_dropout),
                                          model.mess_dropout: eval(args.mess_dropout),
                                          model.neg_items: neg_items})
            loss += batch_loss/n_batch
            mf_loss += batch_mf_loss/n_batch
            emb_loss += batch_emb_loss/n_batch
            reg_loss+=batch_reg_loss/n_batch
        summary_train_loss= sess.run(model.merged_train_loss,
                                      feed_dict={model.train_loss: loss, model.train_mf_loss: mf_loss,
                                                 model.train_emb_loss: emb_loss, model.train_reg_loss: reg_loss})
        train_writer.add_summary(summary_train_loss, epoch)
        if np.isnan(loss) == True:
            print('ERROR: loss is nan.')
            sys.exit()

        # print the test evaluation metrics each 10 epochs; pos:neg = 1:10.
        if (epoch + 1) % 20 != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f+%.5f+%.5f]' % (
                    epoch, time() - t1, loss, mf_loss, emb_loss,reg_loss)
                print(perf_str)
            continue
        users_to_test = list(data_generator.train_items.keys())
        ret = test(sess, model, users_to_test,drop_flag=True,train_set_flag=1)
        perf_str = 'Epoch %d: train==[%.5f=%.5f + %.5f + %.5f], recall=[%.5f, %.5f], ' \
                   'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                   (epoch, loss, mf_loss, emb_loss, reg_loss, ret['recall'][0], ret['recall'][-1],
                    ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
                    ret['ndcg'][0], ret['ndcg'][-1])
        print(perf_str)
        summary_train_acc = sess.run(model.merged_train_acc, feed_dict={model.train_rec20: ret['recall'][0],
                                                                        model.train_rec100: ret['recall'][-1],
                                                                        model.train_ndcg20: ret['ndcg'][0],
                                                                        model.train_ndcg100: ret['ndcg'][-1]})
        train_writer.add_summary(summary_train_acc, (epoch + 1) // 20)
        for idx in range(n_batch):
            users, pos_items, neg_items = data_generator.sample_test()
            batch_loss_test, batch_mf_loss_test, batch_emb_loss_test= sess.run(
                [model.loss, model.mf_loss, model.emb_loss],
                feed_dict={model.users: users, model.pos_items: pos_items,
                           model.neg_items: neg_items,
                          model.node_dropout: eval(args.node_dropout),
                                          model.mess_dropout: eval(args.mess_dropout)})
            loss_test += batch_loss_test / n_batch
            mf_loss_test += batch_mf_loss_test / n_batch
            emb_loss_test += batch_emb_loss_test / n_batch
            

        summary_test_loss = sess.run(model.merged_test_loss,
                                     feed_dict={model.test_loss: loss_test, model.test_mf_loss: mf_loss_test,
                                                model.test_emb_loss: emb_loss_test, model.test_reg_loss: reg_loss_test})
        train_writer.add_summary(summary_test_loss, (epoch + 1) // 20)

        t2 = time()
        users_to_test = list(data_generator.test_set.keys())
        ret = test(sess, model, users_to_test, drop_flag=True)
        summary_test_acc = sess.run(model.merged_test_acc,
                                    feed_dict={model.test_rec20: ret['recall'][0], model.test_rec100: ret['recall'][-1],
                                               model.test_ndcg20: ret['ndcg'][0], model.test_ndcg100: ret['ndcg'][-1]})
        train_writer.add_summary(summary_test_acc, (epoch + 1) // 20)
                                                                                                 
                                                                                                 
                                                                                                 
        t3 = time()

        loss_loger.append(loss)
        rec_loger.append(ret['recall'])
        pre_loger.append(ret['precision'])
        ndcg_loger.append(ret['ndcg'])
        hit_loger.append(ret['hit_ratio'])

        if args.verbose > 0:
            perf_str = 'Epoch %d [%.1fs + %.1fs]: test==[%.5f=%.5f + %.5f + %.5f], recall=[%.5f, %.5f], ' \
                       'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                       (epoch, t2 - t1, t3 - t2, loss_test, mf_loss_test, emb_loss_test, reg_loss_test, ret['recall'][0], ret['recall'][-1],
                        ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
                        ret['ndcg'][0], ret['ndcg'][-1])
            print(perf_str)

        cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                    stopping_step, expected_order='acc', flag_step=5)

        # *********************************************************
        # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
        if should_stop == True:
            break

        # *********************************************************
        # save the user & item embeddings for pretraining.
        if ret['recall'][0] == cur_best_pre_0 and args.save_flag == 1:
            save_saver.save(sess, weights_save_path + '/weights', global_step=epoch)
            print('save the weights in path: ', weights_save_path)

    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)
    hit = np.array(hit_loger)

    best_rec_0 = max(recs[:, 0])
    idx = list(recs[:, 0]).index(best_rec_0)

    final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
                 (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
                  '\t'.join(['%.5f' % r for r in pres[idx]]),
                  '\t'.join(['%.5f' % r for r in hit[idx]]),
                  '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
    print(final_perf)

    save_path = '%soutput/%s/%s.result' % (args.proj_path, args.dataset, model.model_type)
    ensureDir(save_path)
    f = open(save_path, 'a')

    f.write(
        'embed_size=%d, lr=%.4f, layer_size=%s, node_dropout=%s, mess_dropout=%s, regs=%s, adj_type=%s\n\t%s\n'
        % (args.embed_size, args.lr, args.layer_size, args.node_dropout, args.mess_dropout, args.regs,
           args.adj_type, final_perf))
    f.close()
