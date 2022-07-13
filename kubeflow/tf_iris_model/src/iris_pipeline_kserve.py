import kfp.dsl as dsl
import kfp.compiler as compiler
from kfp import onprem
from pytz import timezone, utc
from datetime import datetime as dt
import iris_ops

KST = timezone("Asia/Seoul")
pipeline_start_date_str = utc.localize(dt.utcnow()).astimezone(KST).strftime("%Y%m%d")

op_name = "tf-iris-model"
namespace = "kubeflow-user-example-com"
serving_namespace = "kubeflow"


def make_dataset():
    return dsl.ContainerOp(
        name="load_and_save_iris_dataset",
        image="docker.io/moey920/tf_iris_model:latest",
        command=['python', 'src/data_loader.py']
    ).apply(
        onprem.mount_pvc(
            "data-claim", volume_name="data", volume_mount_path="/code/data"
        )
    )


def tf_iris_train():
    return dsl.ContainerOp(
        name="load_and_save_iris_dataset",
        image="docker.io/moey920/tf_iris_model:latest",
        command=['python', 'src/train_with_preprocess.py']
    ).apply(
        onprem.mount_pvc(
            "data-claim", volume_name="data", volume_mount_path="/code/data"
        )
    )


@dsl.pipeline(name="data_pipeline", description="kf_iris_model_example")
def KF_Iris_Pipeline():
    op_dict = {}
    # ContainerOp 생성
    op_dict["make_dataset"] = make_dataset().set_display_name("make iris dataset")
    # Run the hyperparameter tuning with Katib.
    op_dict["hyper_parameters_tuning"] = iris_ops.create_katib_experiment_task(op_name, namespace)
    op_dict["hyper_parameters_tuning"].set_display_name("hyper_parameters_tuning").after(op_dict["make_dataset"])
    # Run the distributive training with TFJob.
    op_dict["tfjob"] = iris_ops.create_tfjob_task(op_name, namespace, op_dict["hyper_parameters_tuning"])
    op_dict["tfjob"].set_display_name("tfjob")
    # Create the Kserve inference.
    op_dict["serving"] = iris_ops.create_kserve_task(op_name, serving_namespace, op_dict["tfjob"])
    op_dict["serving"].set_display_name("serving").after(op_dict["tfjob"])
    
    
if __name__ == "__main__":
    compiler.Compiler().compile(KF_Iris_Pipeline, __file__[:-3] + ".tar.gz")