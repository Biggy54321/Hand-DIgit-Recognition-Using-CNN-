# import the required libraries
import tensorflow as tf
import numpy as np
import os
from keras.datasets import mnist

# define the class for the Convolutional Neural Network
class CNN:
    def __init__(self):
        # define the placeholders
        self.x = tf.placeholder('float', [None, 28, 28, 1])
        self.y = tf.placeholder('float', [None, 10])
        
        # define the dictionaries for weights and biases
        self.weights = {
            'wc1': tf.get_variable('w1', shape=(3, 3, 1, 32)),
            'wc2': tf.get_variable('w2', shape=(3, 3, 32, 64)),
            'wc3': tf.get_variable('w3', shape=(3, 3, 64, 128)),
            'wc4': tf.get_variable('w4', shape=(3, 3, 128, 256)),
            'wd1': tf.get_variable('w5', shape=(7*7*256, 1024)),
            'wd2': tf.get_variable('w6', shape=(1024, 500)),
            'wd3': tf.get_variable('w7', shape=(500, 100)),
            'wout': tf.get_variable('w8', shape=(100, 10))
        }

        self.bias = {
            'bc1': tf.get_variable('b1', shape=(32)),
            'bc2': tf.get_variable('b2', shape=(64)),
            'bc3': tf.get_variable('b3', shape=(128)),
            'bc4': tf.get_variable('b4', shape=(256)),
            'bd1': tf.get_variable('b5', shape=(1024)),
            'bd2': tf.get_variable('b6', shape=(500)),
            'bd3': tf.get_variable('b7', shape=(100)),
            'bout': tf.get_variable('b8', shape=(10))
        }
        
        # establish a session for evaluating the variables further used in the program
        self.session = tf.Session()
                
        
    # the function will take input image as an array and return the respective 10 tuple output
    def __feed_forward(self, x, training_mode):
        # apply two convolutions(layer 1 and layer 2) and then one pooling
        conv1 = self.__conv2d(x, self.weights['wc1'], self.bias['bc1'])
        conv2 = self.__conv2d(conv1, self.weights['wc2'], self.bias['bc2'])
        conv2 = self.__maxpool2d(conv2)

        # apply two convolutions(layer 3 and layer 4) and then one pooling
        conv3 = self.__conv2d(conv2, self.weights['wc3'], self.bias['bc3'])
        conv4 = self.__conv2d(conv3, self.weights['wc4'], self.bias['bc4'])
        conv4 = self.__maxpool2d(conv4)
        
        # flatten the output from the fourth convolution layer
        fc1 = tf.reshape(conv4, [-1, 7*7*256])
        
        # perform dropout for regularization(only in train mode)
        if training_mode == True:
            fc1 = tf.nn.dropout(fc1, 0.15)
        
        # get the activations of dense layer 1
        fc1 = tf.add(tf.matmul(fc1, self.weights['wd1']), self.bias['bd1'])
        fc1 = tf.nn.relu(fc1)

        # get the activations of dense layer 2
        fc2 = tf.add(tf.matmul(fc1, self.weights['wd2']), self.bias['bd2'])
        fc2 = tf.nn.relu(fc2)
        
        # perform dropout for regularization(only in train mode)
        if training_mode == True:
            fc2 = tf.nn.dropout(fc2, 0.15)
        
        # get the activations of dense layer 3
        fc3 = tf.add(tf.matmul(fc2, self.weights['wd3']), self.bias['bd3'])
        fc3 = tf.nn.relu(fc3)

        # get the output activations
        out = tf.add(tf.matmul(fc3, self.weights['wout']), self.bias['bout'])
        return out
    
    # training method
    def train(self, learn_rate=0.001, batch_size=4000, epochs=30):
        # get the dataset
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        # reshape the features
        X_train = X_train.reshape([-1, 28, 28, 1])
        X_test = X_test.reshape([-1, 28, 28, 1])

        # one hot encode the labels
        y_train = list(y_train)
        for i in range(len(y_train)):
            temp = [0] * 10
            temp[y_train[i]] = 1
            y_train[i] = temp
        y_train = np.array(y_train)

        y_test = list(y_test)
        for i in range(len(y_test)):
            temp = [0] * 10
            temp[y_test[i]] = 1
            y_test[i] = temp
        y_test = np.array(y_test)
        
        # define the computational graph for training
        train_pred = self.__feed_forward(self.x, True)
        train_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=train_pred, labels=self.y))
        optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(train_cost)

        # define the computational graph for testing
        test_pred = self.__feed_forward(self.x, False)
        test_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=test_pred, labels=self.y))

        # initialize global variables
        self.session.run(tf.global_variables_initializer())

        # perform the optimization of the network
        for e in range(epochs):
            print('EPOCH', e)
            # update the weights and the biases for each batch
            for i in range(0, 60000, batch_size):
                _, c = self.session.run((optimizer, train_cost), feed_dict={self.x: X_train[i:i+batch_size], self.y: y_train[i:i+batch_size]})
                print('COST', c)

        # predict the results on the test set
        c = self.session.run(test_cost, feed_dict={self.x: X_test, self.y: y_test})
        print('TEST SET COST -', c)
        
    # define the predict funtion which takes a single image
    def predict(self, image):
        pred = self.__feed_forward(image, False)
        return np.argmax(self.session.run(pred), axis = 1)
    
    # define the required wrapper functions    
    def __conv2d(self, x, W, b):
        x_ret = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
        x_ret = tf.nn.bias_add(x_ret, b)
        return tf.nn.relu(x_ret)

    def __maxpool2d(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # save the current session of the model
    def save_model(self):
        saver = tf.train.Saver()
        save_location = os.getcwd()
        save_location = save_location + '/checkpoint_files'
        # create a separate folder for the checkpoint files
        access_rights = 0o777
        os.mkdir(save_location, access_rights)
        # save the file in the created directory
        save_location = save_location + '/model.ckpt'
        saver.save(self.session, save_location)
        print('Saved the model at', save_location)

    def restore_model(self):
        saver = tf.train.Saver()
        restore_location = os.getcwd()
        restore_location = restore_location + '/checkpoint_files/model.ckpt'
        saver.restore(self.session, restore_location)
        print('Restored the model from', restore_location)
