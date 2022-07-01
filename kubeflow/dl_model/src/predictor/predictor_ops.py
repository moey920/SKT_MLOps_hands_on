import kfp.dsl as dsl
from kfp import onprem


def exec_predictor():
    return (
        dsl.ContainerOp(
            name="forecasting_predictor",
            image="docker.io/moey920/train_forecasting_model:latest",
            command=["sh", "-c"],
            arguments=["python /code/predictor/predictor.py"],
        )
        .apply(
            onprem.mount_pvc(
                "rl-claim", volume_name="rl-model", volume_mount_path="/code/rl_model"
            )
        )
        .apply(
            onprem.mount_pvc(
                "forecasting-claim",
                volume_name="forecasting-model",
                volume_mount_path="/code/forecasting_model",
            )
        )
        .apply(
            onprem.mount_pvc(
                "data-claim", volume_name="data", volume_mount_path="/code/data"
            )
        )
        .set_display_name("Realtime Temp and Dehum Predict Using Forecasting Predictor")
    )
