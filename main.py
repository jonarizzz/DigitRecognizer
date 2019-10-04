import zipfile
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

# load data from kaggle dataset
train_data = pd.read_csv("input/train.csv", header=0)
train_label = train_data['label']
train_img = train_data.iloc[:, 1:] / 255  # scale
train_img = train_img.to_numpy()
train_img.resize(42000, 28, 28, 1)

test_data = pd.read_csv("input/test.csv", header=0)
test_img = test_data.copy() / 255  # scale
test_img = test_img.to_numpy()
test_img.resize(28000, 28, 28, 1)

# check sample submission
submission = pd.read_csv("input/sample_submission.csv", header=0)
submission.head()


# to get 1.00 accuracy, train on entire MNIST dataset, added from kaggle datasets in .csv
# due to size of files they can't be pulled from git as is, so they need to be decompressed first
with zipfile.ZipFile("input/input.zip", 'r') as zip_ref:
    zip_ref.extractall("")
mnist_train = pd.read_csv("input/mnist_train.csv")
mnist_test = pd.read_csv("input/mnist_test.csv")
mnist = pd.concat([mnist_train, mnist_test], axis=0)

# Get all mnist as training
mnist_train_label = mnist['label']
mnist_train_img = mnist.drop('label', axis=1).to_numpy()
mnist_train_img = mnist_train_img / 255  # scale
mnist_train_img.resize(70000, 28, 28, 1)

# build image data generator with keras
datagen = ImageDataGenerator(
    # Parameters for data augmentation:
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

datagen.fit(mnist_train_img)

# create the NN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPool2D(2, 2),
    #    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(192, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D(192, (5, 5), activation='relu', padding='same'),
    tf.keras.layers.MaxPool2D(2, 2, padding='same'),
    #    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=256, activation='relu'),
    #    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(units=10, activation='softmax')
])

model.summary()

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

history = model.fit_generator(datagen.flow(mnist_train_img, mnist_train_label, batch_size=256),
                              epochs=50,
                              verbose=1,
                              validation_data=(train_img, train_label),
                              shuffle=True)

predictions = model.predict(test_img).argmax(axis=1)

results = pd.DataFrame({"ImageId": range(1, len(test_img) + 1), "Label": predictions})
results.to_csv("predictions_submission.csv", header=True, index=False)
