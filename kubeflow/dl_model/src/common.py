import argparse


def ml_parse_args():
    # 1. 모델네임 입력 2. 모델별 파리미터 입력 3. 훈련데이터 입력 4. 저장여부 입력
    parser = argparse.ArgumentParser(description="arguments needs to ml Model Training")
    parser.add_argument(
        "--model_name",
        "-m",
        choices=[
            "LinearRegression, XGBRegressor, XGBRFRegressor, ExtraTreesRegressor, LinearTreeRegressor, LinearSVR, neighbors"
        ],
        required=True,
        default="LinearRegression",
        type=str,
        help="specific ml model name",
    )
    parser.add_argument(
        "--train_data",
        "-d",
        required=False,
        default="train_data.pkl",
        type=str,
        help="preprocessed data saved pkl file",
    )
    parser.add_argument(
        "--save", "-s", required=False, type=str, help="model saved path",
    )
    parser.add_argument(
        "--mean_absolute_error",
        required=False,
        help="don't use it, but I added it because there is a slicing issue with ContainerOp.",
    )

    return parser.parse_args()


def get_ml_hyperparams():
    args = ml_parse_args()

    model_name = args.model_name
    train_data = args.train_data
    save = args.save

    return (
        train_data,
        save,
        model_name,
    )
