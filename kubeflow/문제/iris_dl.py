import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import *
from sklearn.metrics import *
print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))


# Dataset Load
train_dataset_url = (
    "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"
)
train_dataset_fp = tf.keras.utils.get_file(
    fname=os.path.basename(train_dataset_url), origin=train_dataset_url
)
print("Local copy of the dataset file: {}".format(train_dataset_fp))

# column order in CSV file
column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
feature_names = column_names[:-1]
label_name = column_names[-1]
print("Features: {}".format(feature_names))
print("Label: {}".format(label_name))

class_names = ["Iris setosa", "Iris versicolor", "Iris virginica"]
batch_size = 32

train_dataset = tf.data.experimental.make_csv_dataset(
    train_dataset_fp,
    batch_size,
    column_names=column_names,
    label_name=label_name,
    num_epochs=1,
)


def pack_features_vector(features, labels):
    """Pack the features into a single array."""
    features = tf.stack(list(features.values()), axis=1)
    return features, labels


train_dataset = train_dataset.map(pack_features_vector)
features, labels = next(iter(train_dataset))

# Build model, and train
model = Sequential()
model.add(Dense(10, activation=tf.nn.relu, input_shape=(4,)))
model.add(Dense(10, activation=tf.nn.relu))
model.add(Dense(3, activation="softmax"))
model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
model.fit(features, labels, epochs=200)

# Test Dataset Load
test_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv"
test_fp = tf.keras.utils.get_file(fname=os.path.basename(test_url), origin=test_url)

test_dataset = tf.data.experimental.make_csv_dataset(
    test_fp,
    batch_size,
    column_names=column_names,
    label_name="species",
    num_epochs=1,
    shuffle=False,
)

test_dataset = test_dataset.map(pack_features_vector)
test_features, test_labels = next(iter(train_dataset))

result = model.evaluate(test_features,  test_labels, verbose=2)
# katib의 metrics collector에서 메트릭을 수집하기 위해 stdout으로 출력합니다.
print("accuracy: {:.4}".format(result[1]))
