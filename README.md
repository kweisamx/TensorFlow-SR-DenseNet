
# TensorFlow-SRDenseNet


[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/kweisamx/VDSR/blob/master/LICENSE)
## Introduction

![](https://i.imgur.com/ZlNl6Zu.png)

We present a highly accurate single-image super-resolution (SR) method, Use the DenseNet, and use deconvulotion to scaling, the network model of densenet is:
```python
def desBlock(self, desBlock_layer, outlayer, filter_size=3 ):
        nextlayer = self.low_conv
        conv = list()
        for i in range(1, outlayer+1):
            conv_in = list()
            for j in range(1, desBlock_layer+1):
                # The first conv need connect with low level layer
                print(i,j)
                if j is 1:
                    x = tf.nn.conv2d(nextlayer, self.weight_block['w_H_%d_%d' %(i, j)], strides=[1,1,1,1], padding='SAME') + self.biases_block['b_H_%d_%d' % (i, j)]
                    x = tf.nn.relu(x)
                    conv_in.append(x)
                else:
                    x = Concatenation(conv_in)
                    x = tf.nn.conv2d(x, self.weight_block['w_H_%d_%d' % (i, j)], strides=[1,1,1,1], padding='SAME')+ self.biases_block['b_H_%d_%d' % (i, j)]
                    x = tf.nn.relu(x)
                    conv_in.append(x)

            nextlayer = conv_in[-1]
            print(conv_in[-1])
            conv.append(conv_in)
        print(conv)
        return conv
```

## Dependency

pip
* TensorFlow
* OpenCV
* h5py

## Environment

* Ubuntu 16.04
* Python 2.7

If you meet the problem with opencv when run the program

```
libSM.so.6: cannot open shared object file: No such file or directory
```
please install dependency package
```
sudo apt-get install libsm6
sudo apt-get install libxrender1
```

## All Parameter
```
usage: main.py [-h] [--epoch EPOCH] [--image_size IMAGE_SIZE]
               [--label_size LABEL_SIZE] [--c_dim C_DIM]
               [--is_train [IS_TRAIN]] [--nois_train] [--scale SCALE]
               [--stride STRIDE] [--checkpoint_dir CHECKPOINT_DIR]
               [--learning_rate LEARNING_RATE] [--batch_size BATCH_SIZE]
               [--des_block_H DES_BLOCK_H] [--des_block_ALL DES_BLOCK_ALL]
               [--result_dir RESULT_DIR] [--growth_rate GROWTH_RATE]
               [--test_img TEST_IMG]

optional arguments:
  -h, --help            show this help message and exit
  --epoch EPOCH         Number of epoch
  --image_size IMAGE_SIZE
                        The size of image input
  --label_size LABEL_SIZE
                        The size of label
  --c_dim C_DIM         The size of channel
  --is_train [IS_TRAIN]
                        if the train
  --nois_train
  --scale SCALE         the size of scale factor for preprocessing input image
  --stride STRIDE       the size of stride
  --checkpoint_dir CHECKPOINT_DIR
                        Name of checkpoint directory
  --learning_rate LEARNING_RATE
                        The learning rate
  --batch_size BATCH_SIZE
                        the size of batch
  --des_block_H DES_BLOCK_H
                        the size dense_block layer number
  --des_block_ALL DES_BLOCK_ALL
                        the size dense_block
  --result_dir RESULT_DIR
                        Name of result directory
  --growth_rate GROWTH_RATE
                        the size of growrate
  --test_img TEST_IMG   test_img
```
if you want to see the flag
```
python main.py -h
```

## How to train

```
python main.py
```

## How to test
```
python main.py --is_train False --stride 50
```

If you want to Test your own iamge

use test_img flag
```
python main.py --is_train False --stride 50 --test_img Train/t20.bmp
```
then result image also put in the result directory

## Result


* Origin

    ![Imgur](https://i.imgur.com/hhXBTfC.png)
        
            
* Bicbuic 

    ![Imgur](https://i.imgur.com/7UAzDf6.png)
    
* Result

    ![](https://i.imgur.com/oJzclAY.png)

Because the stride is 50, some part are cut.
