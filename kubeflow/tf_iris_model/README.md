# 파이프라인 예시

## data_loader

`tf.keras.utils.get_file()` 으로 url에서 데이터를 다운로드합니다.
다른 컴포넌트에서 해당 데이터를 활용하기 위해 NFS에 데이터를 저장합니다.

## train_with_preprocess

NFS에 저장된 데이터를 읽어와 `tf.data.experimental.make_csv_dataset()`로 훈련/테스트 데이터셋을 만듭니다.
모델을 빌드하고 훈련 후 검증합니다.

# 디렉터리 구조

Kubeflow에서 권장하는 프로젝트 구조를 따라 디렉터리를 구성합니다.
```
- 컴포넌트
    ㄴ src
        ㄴ common.py # argparse 모듈
        ㄴ data_loader.py # 데이터 로드 모듈
        ㄴ train_with_preprocess.py # 훈련(main파일)
        ㄴ iris_ops.py # ContainerOp 생성을 위한 func(래퍼) 모음
        ㄴ iris_pipeline.py # Pipeline 생성을 위한 kfp 모듈
    ㄴ dockerfile
    ㄴ requirements.txt
    ㄴ build_image.sh
    ㄴ iris_pipeline.tar.gz # 컴파일된 파이프라인 파일
```

## Tip!
보통 데이터 수집 / 정제, 모델 훈련 / 서빙 등 컴포넌트를 관심사에 따라 세분화하지만,
본 문제는 데이터 수집부터 훈련의 전 과정에서 Tensorflow를 동일하게 활용하므로 하나의 컨테이너 이미지로 묶었습니다.