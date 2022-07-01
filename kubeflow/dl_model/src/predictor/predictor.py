from urllib import response
import paho.mqtt.client as mqtt
from pytz import timezone, utc
import time, json
import pandas as pd
from datetime import datetime as dt
import datetime
from apscheduler.schedulers.background import BackgroundScheduler
import numpy as np
import os.path
from joblib import load
from datetime import timedelta
from influxdb_client import (
    InfluxDBClient,
    WriteOptions,
    BucketsApi,
    Point,
    WritePrecision,
)
from influxdb_client.client.write_api import SYNCHRONOUS
from influxdb_client.client.exceptions import InfluxDBError

import warnings

warnings.filterwarnings(action="ignore")

import requests

# from kserve import KServeClient
# from kfserving import KFServingClient
# from kfserving import constants
# from kfserving import utils
# from kfserving import V1alpha2EndpointSpec
# from kfserving import V1alpha2PredictorSpec
# from kfserving import V1alpha2TensorflowSpec
# from kfserving import V1alpha2InferenceServiceSpec
# from kfserving import V1alpha2InferenceService

with open("/code/predictor/config.json") as config_file:
    config = json.load(config_file)

KST = timezone("Asia/Seoul")
pipeline_start_date_str = utc.localize(dt.utcnow()).astimezone(KST).strftime("%Y%m%d")

dehum = {}
temp = {}


def on_connect(client, userdata, flags, rc):
    print("Connected with result code", rc)


def on_message(client, userdata, msg):
    global dehum, temp
    msg_payload = msg.payload.decode("utf-8").replace("'", '"')
    msg_payload = msg_payload.replace('"{', "{")
    msg_payload = msg_payload.replace('}"', "}")
    json_msg = json.loads(msg_payload)
    sensor_dict = json_msg["payload"]
    if len(sensor_dict["Dehum"]) != 0:
        update_time = utc.localize(dt.utcnow()).astimezone(KST)
        for dehum_dict in sensor_dict["Dehum"]:
            dehum[dehum_dict["CURR_EQP_ID"]] = {
                "SNSR_ID": dehum_dict["SNSR_ID"],
                "MMDD": dehum_dict["MMDD"],
                "HHMI": dehum_dict["HHMI"],
                "EN_TEMP": dehum_dict["EN_TEMP"],
                "EX_TEMP": dehum_dict["EX_TEMP"],
                "EN_HUM": dehum_dict["EN_HUM"],
                "EX_HUM": dehum_dict["EX_HUM"],
                "CURTEMP": dehum_dict["CURTEMP"],
                "RUNSTOP": dehum_dict["RUNSTOP"],
                "SETMODE": dehum_dict["SETMODE"],
                "SETPLAYTEMP": dehum_dict["SETPLAYTEMP"],
                "SETSTATUS": dehum_dict["SETSTATUS"],
                "SETWARMTEMP": dehum_dict["SETWARMTEMP"],
                "RVOLTAGE": dehum_dict["RVOLTAGE"],
                "RCURRENT": dehum_dict["RCURRENT"],
                "KW": dehum_dict["KW"],
                "KWH": dehum_dict["KWH"],
                "UPDATETIME": update_time,
            }

    if len(sensor_dict["Temp"]) != 0:
        update_time = utc.localize(dt.utcnow()).astimezone(KST)
        for temp_dict in sensor_dict["Temp"]:
            temp[temp_dict["CURR_EQP_ID"]] = {
                "TEMP": temp_dict["TEMP"],
                "HUM": temp_dict["HUM"],
                "FL": temp_dict["FL"],
                "OUTTEMP": temp_dict["OUTTEMP"],
                "OUTHUM": temp_dict["OUTHUM"],
                "UPDATETIME": update_time,
            }

    dehum_list = list(dehum.keys())
    temp_list = list(temp.keys())
    if "2513_T4" not in [
        dehum_list[i].rsplit("_", 1)[0] for i in range(len(dehum_list))
    ]:
        print("dehum_sensor에 2513_T4 정보가 없습니다. 잠시만 기다려주세요.")
        on_message

    if "2513_T4" not in [temp_list[i].rsplit("_", 1)[0] for i in range(len(temp_list))]:
        print("temp_sensor에 2513_T4 정보가 없습니다. 잠시만 기다려주세요")
        on_message


def parsing_queryrange(s_date_str, e_date_str, timeformat):
    """
    Parsing start, stop time string for influx query
    """
    start, stop = "", ""

    dt_start = dt.strptime(s_date_str, timeformat) - timedelta(days=1)
    start = dt_start.strftime("%Y-%m-%d") + "T00:00:00Z"

    dt_end = dt.strptime(e_date_str, timeformat) + timedelta(days=1)
    stop = dt_end.strftime("%Y-%m-%d") + "T00:00:00Z"

    return start, stop


def save_to_influx(result_pd):

    """
    influxDB arguments
    """
    url = f'{config["influx_musma_server"]["host"]}:{config["influx_musma_server"]["port"]}'
    my_token = config["influx_musma_server"]["token"]
    org = config["influx_musma_server"]["org"]

    with InfluxDBClient(url=url, token=my_token, org=org) as _client:
        """
        If there not exists same bucket name, then create Bucket.
        """
        bucket_api = BucketsApi(_client)

        bucket = "realTimePredict"
        find_bucket = bucket_api.find_bucket_by_name(bucket)
        if find_bucket is None:
            print(f"Create '{bucket}' and load data.")
            bucket_api.create_bucket(bucket_name=bucket, org=org)
        else:
            print(f"'{bucket}' already exists. Data is accumulated.")

        p = (
            Point("dsmeHAVC")
            .tag("realtime", "Temp Hum Predictor")
            .field("HUM_03", result_pd["HUM_03"].values[0])
            .field("TEMP_03", result_pd["TEMP_03"].values[0])
            .field("HUM_10", result_pd["HUM_10"].values[0])
            .field("TEMP_10", result_pd["TEMP_10"].values[0])
            .field("EX_HUM", result_pd["EX_HUM"].values[0])
            .field("EN_TEMP", result_pd["EN_TEMP"].values[0])
            .field("TEMP", result_pd["TEMP"].values[0])
            .field("EX_TEMP", result_pd["EX_TEMP"].values[0])
            .field("EN_HUM", result_pd["EN_HUM"].values[0])
            .time(dt.utcnow(), WritePrecision.MS)
        )

        with _client.write_api(
            write_options=WriteOptions(
                batch_size=1000,
                flush_interval=100000,
                jitter_interval=2000,
                retry_interval=60000,
                max_retries=5,
                max_retry_delay=30000,
                exponential_base=2,
            ),
        ) as _write_client:
            """
            Write data into InfluxDB
            """
            _write_client.write(
                bucket=bucket, record=p,
            )
            print(f"'{bucket}'에 데이터 저장을 완료하였습니다. '{bucket}' 버킷의 write_client를 종료합니다.")
            _write_client.close()


def new_temp_dict():
    temp_list = list(temp.keys())
    new_temp_keys = []
    temp_fl_list = []

    for eqp_id in temp_list:
        new_temp_keys.append(eqp_id.rsplit("_", 1)[0])
        temp_fl_list.append(eqp_id.rsplit("_", 1)[1])
    new_temp_keys = list(set(new_temp_keys))
    temp_fl_list = list(set(temp_fl_list))

    new_temp = {}
    new_temp = dict.fromkeys(new_temp_keys)

    for new_key in new_temp_keys:
        new_temp[new_key] = {}

        for fl in temp_fl_list:
            if new_key + "_" + fl in temp_list:
                new_temp[new_key]["TEMP" + "_" + fl] = temp[new_key + "_" + fl]["TEMP"]
                new_temp[new_key]["HUM" + "_" + fl] = temp[new_key + "_" + fl]["HUM"]

        # OUTTEMP, OUTHUM = 0
        new_temp[new_key]["OUTTEMP"] = 0
        new_temp[new_key]["OUTHUM"] = 0

    return new_temp


def new_dehum_dict():
    dehum_list = list(dehum.keys())
    new_dehum_keys = []
    for eqp_id in dehum_list:
        new_dehum_keys.append(eqp_id.rsplit("_", 1)[0])

    new_dehum = {}
    new_dehum = dict.fromkeys(new_dehum_keys)
    SET_MODE_STATUS = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5"}
    dehum_col = ["EX_HUM", "EN_TEMP", "EX_TEMP", "EN_HUM", "CURTEMP"]

    for i, new_key in enumerate(new_dehum_keys):
        new_dehum[new_key] = {}
        for col in dehum_col:
            if col in dehum[dehum_list[i]].keys():
                if col == "CURTEMP":
                    new_dehum[new_key]["TEMP"] = dehum[dehum_list[i]][col]
                else:
                    new_dehum[new_key][col] = dehum[dehum_list[i]][col]

        for mode in SET_MODE_STATUS.keys():
            s = "SET_MODE_" + str(mode)
            new_dehum[new_key][s] = 0
            if dehum[dehum_list[i]]["SETMODE"] == mode:
                new_dehum[new_key][s] = 1

    return new_dehum


real_time_dataset = []


def real_time_data():
    new_temp = new_temp_dict()
    new_dehum = new_dehum_dict()

    mqtt_data = {}
    update_time = utc.localize(dt.utcnow()).astimezone(KST)
    # print(update_time.time())
    for eqp in list(new_temp.keys() & new_dehum.keys()):
        mqtt_data[eqp] = {}
        mqtt_data[eqp] = new_temp[eqp]
        mqtt_data[eqp].update(new_dehum[eqp])
        mqtt_data[eqp]["timestamp_min"] = update_time

    col_list = [
        "HUM_03",
        "TEMP_03",
        "HUM_10",
        "TEMP_10",
        "EX_HUM",
        "EN_TEMP",
        "TEMP",
        "EX_TEMP",
        "EN_HUM",
        "OUTHUM",
        "OUTTEMP",
        "SET_MODE_0",
        "SET_MODE_1",
        "SET_MODE_2",
        "SET_MODE_3",
        "SET_MODE_4",
        "SET_MODE_5",
    ]

    mqtt_data_dict = mqtt_data["2513_T4"]
    real_time = pd.DataFrame([mqtt_data_dict])
    real_time = real_time.set_index("timestamp_min")
    real_time = real_time[col_list]

    real_time_data = real_time.values
    real_time_dataset.append(real_time_data)


def load_data():
    df_me = pd.read_pickle(f"/code/data/df_me_{pipeline_start_date_str}.pkl")
    df_test = df_me[(df_me["PROJ_NO"] == "2513") & (df_me["Tank"] == "4")]
    df_test.sort_values(by=["timestamp_min"], axis=0, inplace=True, ignore_index=True)
    # df_test["diff"] = df_test["timestamp_min"].diff().dt.total_seconds()

    pre_col_list = [
        "HUM_03",
        "TEMP_03",
        "HUM_10",
        "TEMP_10",
        "EX_HUM",
        "EN_TEMP",
        "TEMP",
        "EX_TEMP",
        "EN_HUM",
    ]

    mode_col_list = [
        "OUTHUM",
        "OUTTEMP",
        "SET_MODE_0",
        "SET_MODE_1",
        "SET_MODE_2",
        "SET_MODE_3",
        "SET_MODE_4",
        "SET_MODE_5",
    ]

    df_pre = df_test[pre_col_list]
    df_pre_mode = df_test[mode_col_list]
    df_pre_mode = df_pre_mode.astype("float64")  # 형변화
    df_pre_mode = df_pre_mode

    return df_pre, df_pre_mode


if __name__ == "__main__":

    client = mqtt.Client(client_id=config["mqtt"]["client_id"])
    client.on_connect = on_connect
    client.on_message = on_message

    client.username_pw_set(
        username=config["mqtt"]["username"], password=config["mqtt"]["password"]
    )
    print("Connecting...")
    client.loop_start()
    client.connect(config["mqtt"]["host"], port=config["mqtt"]["port"])
    time.sleep(5)
    client.subscribe(config["mqtt"]["topic"])
    print("start")

    minute_scheduler = BackgroundScheduler(timezone="Asia/Seoul")
    minute_scheduler.add_job(real_time_data, "interval", seconds=10)
    minute_scheduler.start()

    try:
        while True:
            df_pre, df_pre_mode = load_data()
            df_all = pd.concat([df_pre, df_pre_mode], axis=1)
            dataset = df_all.values

            if len(real_time_dataset) != 0:
                for data in real_time_dataset:
                    real_data = np.concatenate(data).tolist()
                    dataset = np.vstack((dataset, real_data))

            # realtime_data_all shape (1,17)
            realtime_data_all = dataset[-1:, :]
            realtime_data_pre = dataset[-1:, :9]
            realtime_data_mode = dataset[-1:, 9:]

            df_pre_mean = df_pre.mean()
            df_pre_std = df_pre.std()

            for data, mean, std in zip(
                list(realtime_data_pre), list(df_pre_mean), list(df_pre_std)
            ):
                normal_pre = (data - mean) / std

            # normal_all shape (1,17)
            normal_realtime_data_all = np.append(
                normal_pre, realtime_data_mode
            ).reshape(1, 17)

            # Todo: model_path 추후에 DL/ML 조건 추가
            model_path = (
                f"/code/forecasting_model/{pipeline_start_date_str}/1/model.joblib"
            )

            # ML model
            if os.path.isfile(model_path):
                model = load(model_path)
                predict_model_data = realtime_data_all

            else:
                # DL - cnnlstm model
                try:
                    model_path = f"/code/forecasting_model/{pipeline_start_date_str}/1"
                    if os.path.isfile(model_path):
                        # model = load(model_name or model_path)
                        normal_realtime_data_all = normal_realtime_data_all.flatten()
                        predict_model_data = normal_realtime_data_all[
                            None, np.newaxis, :, np.newaxis, np.newaxis
                        ]

                # DL - else model
                except:
                    # model = load(model_name or model_path)
                    normal_realtime_data_all = normal_realtime_data_all.flatten()
                    predict_model_data = normal_realtime_data_all[None, :, np.newaxis]

            result = model.predict(predict_model_data)

            result_pd = pd.DataFrame(
                np.round(result, 1),
                columns=[
                    "HUM_03",
                    "TEMP_03",
                    "HUM_10",
                    "TEMP_10",
                    "EX_HUM",
                    "EN_TEMP",
                    "TEMP",
                    "EX_TEMP",
                    "EN_HUM",
                    "OUTHUM",
                    "OUTTEMP",
                    "SET_MODE_0",
                    "SET_MODE_1",
                    "SET_MODE_2",
                    "SET_MODE_3",
                    "SET_MODE_4",
                    "SET_MODE_5",
                ],
            )

            pipeline_start_date_min = (
                utc.localize(dt.utcnow()).astimezone(KST).strftime("%Y-%m-%d %H:%M")
            )
            pipeline_predict_data_min = (
                (utc.localize((dt.utcnow())) + datetime.timedelta(minutes=10))
                .astimezone(KST)
                .strftime("%Y-%m-%d %H:%M")
            )
            print(f"현재 시각은 {pipeline_start_date_min}입니다.")
            print(f"10분 뒤({pipeline_predict_data_min}) 예측 결과는 다음과 같습니다.")
            print(result_pd)

            save_to_influx(result_pd)
            # predict_model_data = predict_model_data.tolist()

            # predict_data_input = {"instances": predict_model_data}

            # temp_date = "20220506"
            # namespace = "kubeflow-user-example-com"
            # KFServing = KFServingClient()
            # isvc_resp = KFServing.get(f"forecasting-model-{temp_date}", namespace=namespace, watch=True, timeout_seconds=120)
            # isvc_url = isvc_resp["status"]["address"]["url"]
            # response = requests.post(isvc_url, json=predict_data_input)

            # KServe = KServeClient()
            # temp_date = "20220506"
            # isvc_resp = KServe.get(
            #     f"forecasting-model-{temp_date}",
            #     namespace="kubeflow-user-example-com",
            # )
            # isvc_url = isvc_resp["status"]["address"]["url"]

            # response = requests.post(isvc_url, json=predict_data_input)

            time.sleep(60)

    except KeyboardInterrupt:
        client.loop_stop()
        client.disconnect()
