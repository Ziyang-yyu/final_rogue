# Used code from
# DQN implementation by Tycho van der Ouderaa found at
# https://github.com/tychovdo/PacmanDQN

# Used code from
# Dueling DQN implemention by Morvan Zhou found at
# https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow

# Dueling DQN implementation based on paper
# 'Dueling Network Architectures for Deep Reinforcement Learning' can be found at
# https://arxiv.org/abs/1511.06581

import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
class DQN:

    def __init__(self, params, name):
        self.params = params
        self.network_name = name
        self.sess = tf.compat.v1.Session()
        if self.params['history']:
            n = 16
        else:
            n = 4
        self.x = tf.compat.v1.placeholder('float', [None, params['width'],params['height'], n],name=self.network_name + '_x') # changed to 16 for 4 frames
        self.q_t = tf.compat.v1.placeholder('float', [None], name=self.network_name + '_q_t')

        self.actions = tf.compat.v1.placeholder("float", [None, 4], name=self.network_name + '_actions')
        self.rewards = tf.compat.v1.placeholder("float", [None], name=self.network_name + '_rewards')
        self.terminals = tf.compat.v1.placeholder("float", [None], name=self.network_name + '_terminals')
        print(self.x)

        if self.params['mlp']:
            print('MLP')
            #  layer 1 (fully connected)
            x_shape = self.x.get_shape().as_list()
            layer_name = 'fc1' ; hiddens = 256 ; dim = x_shape[1]*x_shape[2]*x_shape[3]
            self.x_flat = tf.reshape(self.x, [-1,dim],name=self.network_name + '_'+layer_name+'_input_flat')

            self.w1 = tf.Variable(tf.compat.v1.random_normal([dim,hiddens], stddev=0.01),name=self.network_name + '_'+layer_name+'_weights')
            self.b1 = tf.Variable(tf.constant(0.1, shape=[hiddens]),name=self.network_name + '_'+layer_name+'_biases')
            print(self.w1, self.b1, self.x_flat)
            self.ip1 = tf.add(tf.matmul(self.x_flat,self.w1),self.b1,name=self.network_name + '_'+layer_name+'_ips')
            self.o1 = tf.nn.relu(self.ip1,name=self.network_name + '_'+layer_name+'_activations')
            # Layer 2 (fully connected)
            o1_shape = self.o1.get_shape().as_list()
            layer_name = 'fc2' ; hiddens = 256 ; dim = o1_shape[1]
            self.w2 = tf.Variable(tf.compat.v1.random_normal([dim,hiddens], stddev=0.01),name=self.network_name + '_'+layer_name+'_weights')
            self.b2 = tf.Variable(tf.constant(0.1, shape=[hiddens]),name=self.network_name + '_'+layer_name+'_biases')
            self.ip2 = tf.add(tf.matmul(self.o1,self.w2),self.b2,name=self.network_name + '_'+layer_name+'_ips')
            print(self.w2, self.b2, self.o1)
            self.o2 = tf.nn.relu(self.ip2,name=self.network_name + '_'+layer_name+'_activations')
            dim = self.o2.shape[1] # 256
        else:
            # Layer 1 (Convolutional)
            layer_name = 'conv1' ; size = 3 ; channels = n ; filters = 16 ; stride = 1
            self.w1 = tf.Variable(tf.compat.v1.random_normal([size,size,channels,filters], stddev=0.01),name=self.network_name + '_'+layer_name+'_weights')
            self.b1 = tf.Variable(tf.constant(0.1, shape=[filters]),name=self.network_name + '_'+layer_name+'_biases')
            self.c1 = tf.nn.conv2d(self.x, self.w1, strides=[1, stride, stride, 1], padding='SAME',name=self.network_name + '_'+layer_name+'_convs')
            self.o1 = tf.nn.relu(tf.add(self.c1,self.b1),name=self.network_name + '_'+layer_name+'_activations')
            # Layer 2 (Convolutional)
            layer_name = 'conv2' ; size = 3 ; channels = 16 ; filters = 32 ; stride = 1
            self.w2 = tf.Variable(tf.compat.v1.random_normal([size,size,channels,filters], stddev=0.01),name=self.network_name + '_'+layer_name+'_weights')
            self.b2 = tf.Variable(tf.constant(0.1, shape=[filters]),name=self.network_name + '_'+layer_name+'_biases')
            self.c2 = tf.nn.conv2d(self.o1, self.w2, strides=[1, stride, stride, 1], padding='SAME',name=self.network_name + '_'+layer_name+'_convs')
            self.o2 = tf.nn.relu(tf.add(self.c2,self.b2),name=self.network_name + '_'+layer_name+'_activations')
            # o2:  [None, 7, 7, 32]
            o2_shape = self.o2.get_shape().as_list()
            dim = o2_shape[1]*o2_shape[2]*o2_shape[3]

        # Layer 3 (Fully connected)
        layer_name = 'fc3' ; hiddens = 256
        self.o2_flat = tf.reshape(self.o2, [-1,dim],name=self.network_name + '_'+layer_name+'_input_flat')
        self.w3 = tf.Variable(tf.compat.v1.random_normal([dim,hiddens], stddev=0.01),name=self.network_name + '_'+layer_name+'_weights')
        self.b3 = tf.Variable(tf.constant(0.1, shape=[hiddens]),name=self.network_name + '_'+layer_name+'_biases')
        self.ip3 = tf.add(tf.matmul(self.o2_flat,self.w3),self.b3,name=self.network_name + '_'+layer_name+'_ips')
        self.o3 = tf.nn.relu(self.ip3,name=self.network_name + '_'+layer_name+'_activations')
        print("output:",self.w3, self.b3, self.o2_flat, self.o3)

        # add dueling: split Layer 4 into 2
        if self.params['dueling']:
            print('Dueling DQN')
            layer_name = 'value' ; hiddens = 4 ; dim = 256
            self.w4 = tf.Variable(tf.compat.v1.random_normal([dim,hiddens], stddev=0.01),name=self.network_name + '_'+layer_name+'_weights')
            self.b4 = tf.Variable(tf.constant(0.1, shape=[hiddens]),name=self.network_name + '_'+layer_name+'_biases')
            self.V = tf.add(tf.matmul(self.o3,self.w4),self.b4,name=self.network_name + '_'+layer_name+'_outputs')

            layer_name = 'advantage' ; hiddens = 4 ; dim = 256
            self.w4 = tf.Variable(tf.compat.v1.random_normal([dim,hiddens], stddev=0.01),name=self.network_name + '_'+layer_name+'_weights')
            self.b4 = tf.Variable(tf.constant(0.1, shape=[hiddens]),name=self.network_name + '_'+layer_name+'_biases')
            self.A = tf.add(tf.matmul(self.o3,self.w4),self.b4,name=self.network_name + '_'+layer_name+'_outputs')

            layer_name = 'fc4'
            self.y = self.V + (self.A - tf.reduce_mean(self.A, axis=1, keepdims=True)) # Q = V(s) + A(s,a)
            print("output y:",self.y)
        else:
            # Layer 4
            layer_name = 'fc4' ; hiddens = 4 ; dim = 256
            self.w4 = tf.Variable(tf.compat.v1.random_normal([dim,hiddens], stddev=0.01),name=self.network_name + '_'+layer_name+'_weights')
            self.b4 = tf.Variable(tf.constant(0.1, shape=[hiddens]),name=self.network_name + '_'+layer_name+'_biases')
            self.y = tf.add(tf.matmul(self.o3,self.w4),self.b4,name=self.network_name + '_'+layer_name+'_outputs')

        #Q,Cost,Optimizer
        self.discount = tf.constant(self.params['discount'])
        self.yj = tf.add(self.rewards, tf.multiply(1.0-self.terminals, tf.multiply(self.discount, self.q_t))) # Bellman equation
        self.Q_pred = tf.compat.v1.reduce_sum(tf.multiply(self.y,self.actions), reduction_indices=1)
        self.cost = tf.compat.v1.reduce_sum(tf.pow(tf.subtract(self.yj, self.Q_pred), 2))

        if self.params['load_file'] is not None:
            self.global_step = tf.Variable(int(self.params['load_file'].split('_')[-1]),name='global_step', trainable=False)
        else:
            self.global_step = tf.Variable(0, name='global_step', trainable=False)

        # self.optim = tf.train.RMSPropOptimizer(self.params['lr'],self.params['rms_decay'],0.0,self.params['rms_eps']).minimize(self.cost,global_step=self.global_step)
        self.optim = tf.compat.v1.train.AdamOptimizer(self.params['lr']).minimize(self.cost, global_step=self.global_step)
        self.saver = tf.compat.v1.train.Saver(max_to_keep=0)

        self.sess.run(tf.compat.v1.global_variables_initializer())

        if self.params['load_file'] is not None:
            print('============================================Loading checkpoint=========================================')
            self.saver.restore(self.sess,self.params['load_file'])

    def train(self,bat_s,bat_a,bat_t,bat_n,bat_r):

        feed_dict={self.x: bat_n, self.q_t: np.zeros(bat_n.shape[0]), self.actions: bat_a, self.terminals:bat_t, self.rewards: bat_r}
        q_t = self.sess.run(self.y,feed_dict=feed_dict)
        q_t = np.amax(q_t, axis=1)
        feed_dict={self.x: bat_s, self.q_t: q_t, self.actions: bat_a, self.terminals:bat_t, self.rewards: bat_r}
        _,cnt,cost = self.sess.run([self.optim, self.global_step,self.cost],feed_dict=feed_dict)
        return cnt, cost

    def save_ckpt(self,filename):
        self.saver.save(self.sess, filename)
