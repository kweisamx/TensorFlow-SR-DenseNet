import tensorflow as tf
from model import SRDense
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer("epoch", 15000, "Number of epoch")
flags.DEFINE_integer("image_size", 17, "The size of image input")
flags.DEFINE_integer("label_size", 17, "The size of label")
flags.DEFINE_integer("c_dim", 3, "The size of channel")
flags.DEFINE_boolean("is_train", True, "if the train")
flags.DEFINE_integer("scale", 2, "the size of scale factor for preprocessing input image")
flags.DEFINE_integer("stride", 14, "the size of stride")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Name of checkpoint directory")
flags.DEFINE_float("learning_rate", 1e-5 , "The learning rate")
flags.DEFINE_integer("batch_size", 32, "the size of batch")
flags.DEFINE_integer("growth_rate", 16, "the size of growrate")
flags.DEFINE_integer("des_block_H", 8, "the size dense_block layer number")
flags.DEFINE_integer("des_block_ALL", 3, "the size dense_block")
flags.DEFINE_string("result_dir", "result", "Name of result directory")
flags.DEFINE_string("test_img", "", "test_img")




def main(_): #?
    with tf.Session() as sess:
        espcn = SRDense(sess,
                      image_size = FLAGS.image_size,
                      label_size = FLAGS.label_size,
                      is_train = FLAGS.is_train,
                      scale = FLAGS.scale,
                      c_dim = FLAGS.c_dim,
                      batch_size = FLAGS.batch_size,
                      test_img = FLAGS.test_img,
                      des_block_H = FLAGS.des_block_H,
                      des_block_ALL = FLAGS.des_block_ALL,
                      growth_rate = FLAGS.growth_rate,
                      )

        espcn.train(FLAGS)

if __name__=='__main__':
    tf.app.run() # parse the command argument , the call the main function
