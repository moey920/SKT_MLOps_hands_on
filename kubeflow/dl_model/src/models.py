import sys, os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from Common import config
from Common import utils

# for ml models
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor, XGBRFRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from lineartree import LinearTreeRegressor
from sklearn.svm import LinearSVR
from sklearn import neighbors
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import *
import joblib
import warnings
import json
import pandas as pd

warnings.filterwarnings(action="ignore")

import time
import numpy as np
from pytz import timezone, utc
from datetime import datetime as dt

KST = timezone("Asia/Seoul")
pipeline_start_date_str = utc.localize(dt.utcnow()).astimezone(KST).strftime("%Y%m%d")

mae_performance = []
r2_performance = []
mape_performance = []
rmse_performance = []

(df_me, save, model_name,) = utils.get_ml_hyperparams()


TRAIN_SPLIT, TEST_SPLIT, df_all, dataset, df_dt = utils.prepare_tf_dataset(
    df_me, model_name
)

ml_models = config.ml_models
dl_models = config.dl_models


def prepare_rl_init_data():
    df_all.to_pickle(f"/code/data/rl_init_data_{pipeline_start_date_str}.pkl")


def prepare_rl_test_data():
    global df_dt
    df_dt = df_dt.set_index("timestamp_min")
    df_dt = df_dt.resample("10T").mean()
    pipeline_start_test_date_str = (
        utc.localize(dt.utcnow()).astimezone(KST).date().strftime("%Y-%m-%d")
    )
    date_index = pd.date_range(end=pipeline_start_test_date_str, periods=1)
    for date in date_index:
        date = date.strftime("%Y-%m-%d")
        df_dt[date].to_csv(f"/code/data/actual_data/{date}.csv")


def prepare_ml_data():

    x_train_multi, y_train_multi = utils.multivariate_multioutput_data(
        dataset[:, :],
        dataset[:, :],
        0,
        TRAIN_SPLIT,
        config.history_size,
        config.future_target,
        config.STEP,
        model_name,
        single_step=True,
    )
    x_val_multi, y_val_multi = utils.multivariate_multioutput_data(
        dataset[:, :],
        dataset[:, :],
        TRAIN_SPLIT,
        TEST_SPLIT,
        config.history_size,
        config.future_target,
        config.STEP,
        model_name,
        single_step=True,
    )
    x_test_multi, y_test_multi = utils.multivariate_multioutput_data(
        dataset[:, :],
        dataset[:, :],
        TEST_SPLIT,
        None,
        config.history_size,
        config.future_target,
        config.STEP,
        model_name,
        single_step=True,
    )

    ml_x_train_multi = x_train_multi.reshape(
        x_train_multi.shape[0] * x_train_multi.shape[1], x_train_multi.shape[2]
    )
    ml_y_train_multi = y_train_multi.reshape(
        y_train_multi.shape[0] * y_train_multi.shape[1], y_train_multi.shape[2]
    )
    ml_x_val_multi = x_val_multi.reshape(
        x_val_multi.shape[0] * x_val_multi.shape[1], x_val_multi.shape[2]
    )
    ml_y_val_multi = y_val_multi.reshape(
        y_val_multi.shape[0] * y_val_multi.shape[1], y_val_multi.shape[2]
    )
    ml_y_test_multi = x_test_multi.reshape(
        x_test_multi.shape[0] * x_test_multi.shape[1], x_test_multi.shape[2]
    )
    ml_y_test_multi = y_test_multi.reshape(
        y_test_multi.shape[0] * y_test_multi.shape[1], y_test_multi.shape[2]
    )

    return (
        ml_x_train_multi,
        ml_y_train_multi,
        ml_x_val_multi,
        ml_y_val_multi,
        ml_y_test_multi,
        ml_y_test_multi,
    )


def model_fit():
    print(f"{model_name} 훈련이 진행되고 있습니다. 잠시만 기다려주세요")
    model.fit(ml_x_train_multi[:-10], ml_x_train_multi[10:])


def model_evaluate():
    # evaluate
    y_pred = model.predict(ml_y_test_multi[:-10])

    y_test = ml_y_test_multi
    y_pred = np.squeeze(y_pred)

    for i in range(1, 4, 2):
        mae_performance.append(
            round(mean_absolute_error(y_pred[:, i], y_test[10:, i]), 4)
        )
        r2_performance.append(round(r2_score(y_pred[:, i], y_test[10:, i]), 4))
        mape_performance.append(
            round(mean_absolute_percentage_error(y_pred[:, i], y_test[10:, i]), 4)
        )
        rmse_performance.append(
            round(mean_squared_error(y_pred[:, i], y_test[10:, i], squared=False), 4)
        )

    mae_avg = sum(mae_performance) / len(mae_performance)
    r2_avg = sum(r2_performance) / len(r2_performance)
    mape_avg = sum(mape_performance) / len(mape_performance)
    rmse_avg = sum(rmse_performance) / len(rmse_performance)

    print(f"mean_absolute_error={mae_avg:0.4f}")
    print(f"r2_score={r2_avg:0.4f}")
    print(f"mean_absolute_percentage_error={mape_avg:0.4f}")
    print(f"root_mean_squared_error={rmse_avg:0.4f}")


def model_save():

    if save:
        prepare_rl_init_data()
        # prepare_rl_test_data()

        # 버전관리는 날짜로 합니다.
        version = 1
        export_path = os.path.join(save, str(pipeline_start_date_str), str(version))
        model_filepath = f"{export_path}/model.joblib"
        # assert os.path.isfile(model_filepath)

        if not os.path.exists(f"{export_path}"):
            os.makedirs(f"{export_path}", exist_ok=True)
        if not os.path.isfile(model_filepath):
            with open(model_filepath, "wb") as f:
                joblib.dump(model, f)
        print(f"{export_path}/model.joblib 에 ML 모델이 저장되었습니다.")

    print("\nModel Runtime: %0.2f Minutes" % ((time.time() - modelstart) / 60))


def xgb_regressor():
    model = MultiOutputRegressor(XGBRegressor(n_jobs=16))
    return model


def linear_regression():
    model = MultiOutputRegressor(LinearRegression())
    return model


def xgbrf_regressor():
    model = MultiOutputRegressor(XGBRFRegressor(n_jobs=16))
    return model


def kneighbors_regressor():
    model = MultiOutputRegressor(neighbors.KNeighborsRegressor())
    return model


def linear_svr():
    model = MultiOutputRegressor(LinearSVR(epsilon=0))
    return model


def linear_tree_regressor():
    model = MultiOutputRegressor(LinearTreeRegressor(base_estimator=LinearRegression()))
    return model


def extra_trees_regressor():
    model = MultiOutputRegressor(ExtraTreesRegressor())
    return model
