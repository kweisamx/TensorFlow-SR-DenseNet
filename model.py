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
    make_bicubic,
)
def conv_layer(input_, bias, weights, strides, padding='SAME'):
    conv = tf.nn.conv2d(input_, weights, strides=strides, padding=padding) + biases
    return conv

def Relu(input_):
    return tf.nn.relu(input_)

def Concatenation(layers):
    return tf.concat(layers, axis=3)

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
    
    # Create DenseNet init weight and biases
    def DesWBH(self, layers, filter_size):
        weightsH = {}
        biasesH = {}
        for i in range(1, layers+1):
            weightsH.update({'w_H_%d' % i: tf.Variable(tf.random_normal([filter_size, filter_size, self.growth_rate, self.growth_rate], stddev=np.sqrt(2.0/filter_size * filter_size/self.growth_rate)), name='w_H_%d' % i)})
            biasesH.update({'b_H_%d' % i:tf.Variable(tf.zeros([filter_size * filter_size ], name='b_H_%d' % i))})
        return weightsH, biasesH

    def build_model(self):
        self.images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name='images')
        self.labels = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, self.c_dim], name='labels')

        # NOTE: One block weight
        self.weight_block, self.biases_block = self.DesWBH(self.des_block_H, 3)
        print(self.weight_block)
        print(self.biases_block)


    #def model(self):

    def train(self, config):
        
        # NOTE : if train, the nx, ny are ingnored
        input_setup(config)

        data_dir = checkpoint_dir(config)
        
        input_, label_ = read_data(data_dir)

        residul = make_bicubic(input_, config.scale)


        '''
        opt = tf.train.AdamOptimizer(learning_rate=config.learning_rate)
        grad_and_value = opt.compute_gradients(self.loss)

        clip = tf.Variable(0.1, name='clip') 
        capped_gvs = [(tf.clip_by_value(grad, -(clip), clip), var) for grad, var in grad_and_value]

        self.train_op = opt.apply_gradients(capped_gvs)
        '''
        # Stochastic gradient descent with the standard backpropagation
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
                    batch_residul = residul[idx * config.batch_size : (idx + 1) * config.batch_size]
                    batch_labels = label_[idx * config.batch_size : (idx + 1) * config.batch_size]
                    counter += 1
                    _, err = self.sess.run([self.train_op, self.loss], feed_dict={self.images: batch_images, self.labels: batch_labels, self.residul: batch_residul })

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
                print( string)
                imsave(x, config.result_dir + '/' + string[0] + '.png', config)
    def load(self, checkpoint_dir):
        """
            To load the checkpoint use to test or pretrain
        """
        print("\nReading Checkpoints.....\n\n")
        model_dir = "%s_%s_%s" % ("rees", self.image_size, self.scale)# give the model name by label_size
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
        model_name = "REES.model"
        model_dir = "%s_%s_%s" % ("rees", self.image_size, self.scale)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
             os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)
