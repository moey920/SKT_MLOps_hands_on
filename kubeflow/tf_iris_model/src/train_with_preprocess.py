import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
from pytz import timezone, utc
from datetime import datetime as dt
import time

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import *

import common


KST = timezone("Asia/Seoul")
pipeline_start_date_str = utc.localize(dt.utcnow()).astimezone(KST).strftime("%Y%m%d")


def preprocessor():
    # column order in CSV file
    column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
    feature_names = column_names[:-1]
    label_name = column_names[-1]
    print("Features: {}".format(feature_names))
    print("Label: {}".format(label_name))

    batch_size, epochs, optimizer, save = common.get_hyperparams()
    batch_size = batch_size

    train_dataset = tf.data.experimental.make_csv_dataset(
        "/code/data/train_dataset.csv",
        batch_size,
        column_names=column_names,
        label_name=label_name,
        num_epochs=1,
    )
    test_dataset = tf.data.experimental.make_csv_dataset(
        "/code/data/test_dataset.csv",
        batch_size,
        column_names=column_names,
        label_name="species",
        num_epochs=1,
        shuffle=False,
    )


    def pack_features_vector(features, labels):
        """Pack the features into a single array."""
        features = tf.stack(list(features.values()), axis=1)
        return features, labels


    train_dataset = train_dataset.map(pack_features_vector)
    x_features, x_labels = next(iter(train_dataset))
    
    test_dataset = test_dataset.map(pack_features_vector)
    y_features, y_labels = next(iter(train_dataset))
    
    return x_features, x_labels, y_features, y_labels


def tf_iris_model():
    
    model_start = time.time()
    
    x_features, x_labels, y_features, y_labels = preprocessor()
    batch_size, epochs, optimizer, save = common.get_hyperparams()

    # Build model, and train
    model = Sequential()
    model.add(Dense(10, activation=tf.nn.relu, input_shape=(4,)))
    model.add(Dense(10, activation=tf.nn.relu))
    model.add(Dense(3, activation="softmax"))
    model.compile(optimizer=optimizer,
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
    model.fit(x_features, x_labels, epochs=epochs)
    
    result = model.evaluate(y_features,  y_labels, verbose=2)
    # katib??? metrics collector?????? ???????????? ???????????? ?????? stdout?????? ???????????????.
    print("accuracy={:.4}".format(result[1]))
    
    if save: # Katib??? ????????? TFJob????????? save??? arguement??? ?????? ????????? ???????????????.
        # version??? KFserving?????? TF????????? ???????????? ?????? ??????(str)?????????. ??????????????? ????????? ?????????.
        version = 1
        export_path = os.path.join(save, str(pipeline_start_date_str), str(version))
        model.save(f"{export_path}")
        print(f"{export_path}??? TF ????????? ?????????????????????.")

    print("\nModel Runtime: %0.2f Minutes" % ((time.time() - model_start) / 60))


if __name__ == "__main__":
    tf_iris_model()