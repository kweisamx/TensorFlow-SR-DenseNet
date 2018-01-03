import tensorflow as tf
import numpy as np
import time
import os
import cv2
from utils import (
    input_setup,
    checkpoint_dir,
    read_data,
    checkimage,
    imsave,
    imread,
    load_data,
    preprocess,
    modcrop,
)
def Relu(input_):
    return tf.nn.relu(input_)

def Concatenation(layers):
    return tf.concat(layers, axis=3)
   
def SkipConnect(conv):
    skipconv = list()
    for i in conv:
        x = Concatenation(i)
        skipconv.append(x)
    return skipconv

class SRDense(object):

    def __init__(self,
                 sess,
                 image_size,
                 label_size,
                 is_train,
                 scale,
                 batch_size,
                 c_dim,
                 test_img,
                 des_block_H,
                 des_block_ALL,
                 growth_rate
                 ):

        self.sess = sess
        self.image_size = image_size
        self.is_train = is_train
        self.c_dim = c_dim
        self.scale = scale
        self.label_size = label_size
        self.batch_size = batch_size
        self.test_img = test_img
        self.des_block_H = des_block_H
        self.des_block_ALL = des_block_ALL
        self.growth_rate = growth_rate

        self.build_model()
    
    # Create DenseNet init weight and biases, init_block mean the growrate
    def DesWBH(self, desBlock_layer, filter_size, outlayer):
        weightsH = {}
        biasesH = {}
        fs = filter_size
        for i in range(1, outlayer+1):
            for j in range(1, desBlock_layer+1):
                if j is 1:
                    weightsH.update({'w_H_%d_%d' % (i, j): tf.Variable(tf.random_normal([fs, fs, self.growth_rate, self.growth_rate], stddev=np.sqrt(2.0/fs * fs/self.growth_rate)), name='w_H_%d_%d' % (i, j))}) 
                else:
                    weightsH.update({'w_H_%d_%d' % (i, j): tf.Variable(tf.random_normal([fs, fs, self.growth_rate * (j-1), self.growth_rate], stddev=np.sqrt(2.0/ fs * fs/ self.growth_rate * j )), name='w_H_%d_%d' % (i, j))}) 
                biasesH.update({'b_H_%d_%d' % (i, j): tf.Variable(tf.zeros([self.growth_rate], name='b_H_%d_%d' % (i, j)))})
        return weightsH, biasesH
    

    # Create one Dense Block Convolution Layer 
    def desBlock(self, desBlock_layer, outlayer, filter_size=3 ):
        nextlayer = self.low_conv
        conv = list()
        for i in range(1, outlayer+1):
            conv_in = list()
            for j in range(1, desBlock_layer+1):
                # The first conv need connect with low level layer
                if j is 1:
                    x = tf.nn.conv2d(nextlayer, self.weight_block['w_H_%d_%d' %(i, j)], strides=[1,1,1,1], padding='SAME') + self.biases_block['b_H_%d_%d' % (i, j)]
                    x = tf.nn.relu(x)
                    conv_in.append(x)
                else:
                    concat = Concatenation(conv_in)
                    x = tf.nn.conv2d(concat, self.weight_block['w_H_%d_%d' % (i, j)], strides=[1,1,1,1], padding='SAME')+ self.biases_block['b_H_%d_%d' % (i, j)]
                    x = tf.nn.relu(x)
                    conv_in.append(x)
            nextlayer = conv_in[-1]
            conv.append(conv_in)
        return conv

    def bot_layer(self, input_layer):
        x = tf.nn.conv2d(input_layer, self.bot_weight, strides=[1,1,1,1], padding='SAME') + self.bot_biases
        x = tf.nn.relu(x)
        return x 

    def deconv_layer(self, input_layer):
        deconv_output = [self.batch, self.label_size, self.label_size, 256]
        print(input_layer)
        x = tf.nn.conv2d_transpose(input_layer, self.deconv1_weight, output_shape=deconv_output, strides=[1,2,2,1], padding='SAME') + self.deconv1_biases
        #print(x)
        x = tf.nn.relu(x)
        
        return x 
    def reconv_layer(self, input_layer):
        x = tf.nn.conv2d(input_layer, self.reconv_weight, strides=[1,1,1,1], padding='SAME') + self.reconv_biases
        x = tf.nn.relu(x)
        return x 
        
    def model(self):
        
        x = self.desBlock(self.des_block_H, self.des_block_ALL, filter_size = 3)

        # NOTE: Cocate all dense block
        x = SkipConnect(x)
        print(x)
        x.append(self.low_conv)
        print(x)
        x = Concatenation(x)
        print(x)
        x = self.bot_layer(x)
        print(x)
        x = self.deconv_layer(x)
        print(x)
        x = self.reconv_layer(x)
        print(x)
        
        return x 


    def build_model(self):
        self.images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name='images')
        self.labels = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, self.c_dim], name='labels')

        # Low Level Layer
        self.low_weight = tf.Variable(tf.random_normal([3, 3, self.c_dim, 16], stddev= np.sqrt(2.0/9/16)), name='w_low')
        self.low_biases = tf.Variable(tf.zeros([16], name='b_low'))
        self.low_conv = tf.nn.relu(tf.nn.conv2d(self.images, self.low_weight, strides=[1,1,1,1], padding='SAME') + self.low_biases)
        
        # NOTE: Init each block weight
        """
            16 -> 128 -> 1024 
        """
        # DenseNet blocks 
        self.weight_block, self.biases_block = self.DesWBH(self.des_block_H, 3, self.des_block_ALL)
        #print(self.weight_block, self.biases_block)

        # Bottleneck layer
        allfeature = self.growth_rate * self.des_block_H * self.des_block_ALL + self.growth_rate
        print(allfeature)
        self.bot_weight = tf.Variable(tf.random_normal([1, 1, allfeature, 256], stddev = np.sqrt(2.0/1/256)), name='w_bot')
        self.bot_biases = tf.Variable(tf.zeros([256], name='b_bot'))

        # Deconvolution layer
        self.batch = tf.placeholder(tf.int32, shape=[], name='batch')
        self.deconv1_weight = tf.Variable(tf.random_normal([3, 3, 256, 256], stddev = np.sqrt(2.0/9/256)), name='w_deconv1')
        self.deconv1_biases = tf.Variable(tf.zeros([256], name='b_deconv1'))

        # Reconstruction layer

        self.reconv_weight = tf.Variable(tf.random_normal([3, 3, 256, self.c_dim], stddev = np.sqrt(2.0/9/self.c_dim)), name ='w_reconv')
        self.reconv_biases = tf.Variable(tf.zeros([self.c_dim], name='b_reconv'))

        
        self.pred = self.model()
        print(self.pred.get_shape())
        print(self.labels.get_shape())


        self.loss = tf.reduce_mean(tf.square(self.labels - self.pred))

        self.saver = tf.train.Saver() # To save checkpoint

    def train(self, config):
        
        # NOTE : if train, the nx, ny are ingnored
        input_setup(config)

        data_dir = checkpoint_dir(config)
        
        input_, label_ = read_data(data_dir)

        # Stochastic gradient descent with tself.des_block_ALLhe standard backpropagation
        self.train_op = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(self.loss)
        tf.initialize_all_variables().run()

        counter = 0
        time_ = time.time()

        self.load(config.checkpoint_dir)
        # Train
        if config.is_train:
            print("Now Start Training...")
            for ep in range(config.epoch):
                # Run by batch images
                batch_idxs = len(input_) // config.batch_size
                for idx in range(0, batch_idxs):
                    batch_images = input_[idx * config.batch_size : (idx + 1) * config.batch_size]
                    batch_labels = label_[idx * config.batch_size : (idx + 1) * config.batch_size]
                    counter += 1
                    _, err = self.sess.run([self.train_op, self.loss], feed_dict={self.images: batch_images, self.labels: batch_labels, self.batch: self.batch_size})
                    if counter % 10 == 0:
                        print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]" % ((ep+1), counter, time.time()-time_, err))
                        #print(label_[1] - self.pred.eval({self.images: input_})[1],'loss:]',err)
                    if counter % 500 == 0:
                        self.save(config.checkpoint_dir, counter)
        # Test
        else:
            print("Now Start Testing...")
            print(input_[0].shape)

            #checkimage(residul[0])
            result = self.pred.eval({self.images: input_[0].reshape(1, input_[0].shape[0], input_[0].shape[1], self.c_dim)})
            x = np.squeeze(result)
            #checkimage(x)
            x = residul[0] + x
            
            # back to interval [0 , 1]
            x = ( x + 1 ) / 2
            #checkimage(x)
            print(self.test_img)
            print(x.shape)
            if self.test_img is "":
                imsave(x, config.result_dir + '/result.png', config)
            else:
                string = self.test_img.split(".")
                print(string)
                imsave(x, config.result_dir + '/' + string[0] + '.png', config)

    def load(self, checkpoint_dir):
        """
            To load the checkpoint use to test or pretrain
        """
        print("\nReading Checkpoints.....\n\n")
        model_dir = "%s_%s_%s" % ("dnsr", self.image_size, self.scale)# give the model name by label_size
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        
        # Check the checkpoint is exist 
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_path = str(ckpt.model_checkpoint_path) # convert the unicode to string
            self.saver.restore(self.sess, os.path.join(os.getcwd(), ckpt_path))
            print("\n Checkpoint Loading Success! %s\n\n"% ckpt_path)
        else:
            print("\n! Checkpoint Loading Failed \n\n")
    def save(self, checkpoint_dir, step):
        """
            To save the checkpoint use to test or pretrain
        """
        model_name = "DenseNetSR.model"
        model_dir = "%s_%s_%s" % ("dnsr", self.image_size, self.scale)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
             os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)
