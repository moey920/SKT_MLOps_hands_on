import requests
import json
import time

USERNAME = "user@example.com"
PASSWORD = "12341234"
# node_ip:http_port, 강의 자료의 추가 팁3을 참고해주세요. https://www.notion.so/moey920/KFServing-f0a0bab8197e46d9b7e159ba57d81703#2c9d7ccb85dd457cb59466d7e2963721
HOST = "http://192.168.0.14:32307"

with requests.Session() as session:
    # Auth
    response = session.get(HOST)
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
    }
    data = {"login": USERNAME, "password": PASSWORD}
    session.post(response.url, headers=headers, data=data)
    session_cookie = session.cookies.get_dict()["authservice_session"]

    # Predict
    headers = {"Host": "tf-iris-model.kubeflow-user-example-com.example.com"}
    cookies = {"authservice_session": session_cookie}

    data = {
        'instances': [
            [6.8,  2.8,  4.8,  1.4],
            [6.0,  3.4,  4.5,  1.6]
        ]
    }

    pre = time.time()
    res = session.post("http://192.168.0.14:32307/v1/models/tf-iris-model:predict", headers=headers, cookies=cookies, data=json.dumps(data))
    print(res.json())
    print(f"latency: {(time.time() - pre)*1000} ms")