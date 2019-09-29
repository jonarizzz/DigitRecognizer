import data_converter as dc
import tensorflow as tf

TRAIN_FILENAME = "data/train.csv"
TEST_FILENAME = "data/test.csv"

data = dc.csv_data_reader(TRAIN_FILENAME)
dc.detach_labels(data)
train_data = dc.transform_to_array(data)
test_data = dc.transform_to_array(dc.csv_data_reader(TEST_FILENAME))

learning_rate = 0.01
training_iteration = 30
batch_size = 100
display_step = 2

# x = tf.placeholder("float", [None, 784])
print(tf.__version__)
