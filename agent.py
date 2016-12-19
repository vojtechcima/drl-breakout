import numpy as np
import tensorflow as tf
import game
import random
class Agent:
    def __init__(self, load_model=None):
        self.g = game.Game(84, 84)
        self.session = tf.Session()
        self.learning_rate = 0.00025
        self. grad_momentum = 0.95
        self.shape = {"cv1": [8, 8, 1, 32],
                      "cv2": [4, 4, 32, 64],
                      "cv3": [3, 3, 64, 64],
                      "fc": [121 * 64, 512],
                      "out": [512, self.g.actions_len]}
        self.strides = {"cv1": [1, 4, 4, 1],
                        "cv2": [1, 2, 2, 1],
                        "cv3": [1, 1, 1, 1]}
        self.weights = self.init_weights()
        self.biases = self.init_biases()
        self.display_step = 1
        self.state_dims = self.g.get_state_dims()

        self.x = tf.placeholder(tf.float32, [None, self.state_dims[0] * self.state_dims[1]])
        self.y_ = tf.placeholder(tf.float32, [None, self.shape["out"][-1]])

        self.y = self.feed_forward()
        self.loss = tf.reduce_mean(tf.square(self.y - self.y_))

        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate, momentum=self.grad_momentum).minimize(self.loss)
        self.correct_prediction = tf.equal(
            tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(
            tf.cast(self.correct_prediction, tf.float32))

        if load_model:
            model_path = tf.train.latest_checkpoint("./model/model")
            saver = tf.train.Saver()
            saver.restore(self.session, model_path)
            print("Loading existing session %s" % (model_path))
        else:
            tf.initialize_all_variables().run(session=self.session)

    def train(self, verbose=True):
        save_model = False
        #epochs = 1000
        batch_size = 32
        replay_buf_len = 1000000
        replay_buf = []
        gamma = 0.99
        epsilon = 1
        epsilon_min = 0.1
        save_step = 10000
        frame = 0
        frame_max = 10000000
        saver = tf.train.Saver()
        t = 0
        epoch = 1
        while t < frame_max:
            step = 0
            while(True):
                state = self.g.get_state() # Get state S
                if (random.random() < epsilon): # choose random action
                    action_idx = np.random.randint(0, self.g.actions_len)
                else: # choose the most confident action
                    qval = self.y.eval(
                        {self.x: state.reshape(1, self.g.state_len)},
                        session=self.session)
                    action_idx = np.argmax(qval)
                action = self.g.actions[action_idx]
                reward = self.g.make_move(action) # Take action and get reward
                new_state = self.g.get_state() # Get new state S'
                game_over = self.g.game_over()
                t += 1
                
                # Take care memory buffer
                if (len(replay_buf) >= replay_buf_len):
                    replay_buf.pop(0)
                replay_buf.append((state, action_idx, reward, new_state, game_over))

                # Sample memory buffer
                if len(replay_buf) < batch_size:
                    sampled_replay_buf = random.sample(replay_buf, len(replay_buf))
                else:
                    sampled_replay_buf = random.sample(replay_buf, batch_size)

                X_train = []
                y_train = []
                for memory in sampled_replay_buf:
                    # Iterate through the memory buffer
                    old_state, action_idx, reward, new_state, terminal = memory
                    old_qval = self.y.eval(
                        {self.x: old_state.reshape(1, self.g.state_len)},
                        session=self.session)
                    new_qval = self.y.eval(
                        {self.x: new_state.reshape(1, self.g.state_len)},
                        session=self.session)

                    max_q = np.max(new_qval)
                    y = np.zeros((1, self.g.actions_len))
                    y[:] = old_qval[:]
                    if terminal:
                        update = reward
                    else:
                        update = reward + gamma * max_q 
                    y[0][action_idx] = update
                    X_train.append(old_state.reshape(self.g.state_len,))
                    y_train.append(y.reshape(self.g.actions_len,))
                X_train = np.array(X_train)
                y_train = np.array(y_train)
                #print self.y.eval({self.x: X_train, self.y_: y_train}, session=self.session)
                #print self.y_.eval({self.x: X_train, self.y_: y_train}, session=self.session)
                if verbose: 
                    print("Epoch: %s, Step: %s, Time: %s, Epsilon: %s, Loss: %s" %
                          (epoch, step, t, epsilon,
                           self.loss.eval({self.x: X_train, self.y_: y_train},
                                      session=self.session)))
                # Fit model
                self.optimizer.run({self.x: X_train, self.y_: y_train},
                                   session=self.session)
                
                step += 1
                if game_over:
                    epoch += 1
                    print("Game Over! Score: %s" % self.g.score)
                    self.g.reset_game()
                    break

            if epsilon > epsilon_min: # decrement epsilon
                epsilon -= ( 1. / 1000000.)

            if t % save_step == 0 and t > 0 and save_model: # save model
                saver.save(self.session, "/home/cim0009/rl/model/model-%s" % t)

    def conv2d(self, x, layer):
        return tf.nn.conv2d(x, self.weights[layer], strides=self.strides[layer], padding='SAME')

    def init_weight(self, shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

    def init_bias(self, shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

    def init_weights(self):
        weights = {}
        for k in self.shape.keys():
            weights[k] = self.init_weight(self.shape[k])
        return weights

    def init_biases(self):
        biases = {}
        for k in self.shape.keys():
            biases[k] = self.init_bias([self.shape[k][-1]])
        return biases

    def feed_forward(self):
        # Reshape image for convolution [-1, width, height, no_channels]
        x_image = tf.reshape(self.x, [-1, self.state_dims[0],
                                      self.state_dims[1], self.state_dims[2]])
        # First conv layer
        l_cv1 = tf.nn.relu(self.conv2d(
            x_image, "cv1") + self.biases["cv1"])

        # Second conv layer
        l_cv2 = tf.nn.relu(self.conv2d(
            l_cv1, "cv2") + self.biases["cv2"])

        # Third conv layer
        l_cv3 = tf.nn.relu(self.conv2d(
            l_cv2, "cv3") + self.biases["cv3"])

        # FC layer DO no pooling and reshaping
        h_pool2_flat = tf.reshape(l_cv3, [-1, self.shape["fc"][0]])
        h_fc1 = tf.nn.relu(
            tf.matmul(h_pool2_flat, self.weights["fc"]) + self.biases["fc"])

        # Readout layer
        y = tf.matmul(h_fc1, self.weights["out"]) + self.biases["out"]
        return y

    def test(self, verbose=False):
        if verbose:
            print("--------testing----------")
        i = 0
        while(not self.g.game_over()):
            i += 1
            state = self.g.get_state()
            qval = self.y.eval(
                {self.x: state.reshape(1, self.g.state_len)},
                session=self.session)
            action_idx = np.argmax(qval)
            action = self.g.actions[action_idx]
            if verbose:
                print('Move #: %s; Taking action: %s; Score: %s' %
                      (i, action, self.g.score))
            reward = self.g.make_move(action)
        return i, self.g.score
