import tensorflow as tf
import pandas as pd
import os
import requests
requests.packages.urllib3.disable_warnings()
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context


def data_loader():
    train_dataset_url = (
        "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"
    )
    train_dataset_fp = tf.keras.utils.get_file(
        fname=os.path.basename(train_dataset_url), origin=train_dataset_url
    )
    print("Local copy of the train dataset file: {}".format(train_dataset_fp))

    test_dataset_url = (
        "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv"
    )
    test_dataset_fp = tf.keras.utils.get_file(
        fname=os.path.basename(test_dataset_url), origin=test_dataset_url
    )

    # 서로 다른 컴포넌트간 데이터 공유를 위해 NFS에 데이터를 저장합니다.
    train_data = pd.read_csv(train_dataset_fp)
    train_data.to_csv("/code/data/train_dataset.csv", index=False)
    test_data = pd.read_csv(test_dataset_fp)
    test_data.to_csv("/code/data/test_dataset.csv", index=False)


if __name__ == "__main__":
    data_loader()