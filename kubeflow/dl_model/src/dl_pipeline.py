import kfp
import kfp.dsl as dsl
from kubernetes import client as k8s_client
import kfp.compiler as compiler
import dl_ops

from pytz import timezone, utc
from datetime import datetime as dt

KST = timezone("Asia/Seoul")
pipeline_start_date_str = utc.localize(dt.utcnow()).astimezone(KST).strftime("%Y%m%d")

name = "forecasting-model-" + pipeline_start_date_str
dl_op_name = "forecasting-dl-model"
namespace = "kubeflow-user-example-com"
df_me = "df_me_14.pkl"


@dsl.pipeline(
    name="forecasting",
    description="forecasting model including hyperparameter tuning, train and inference",
)
def tfjob_pipeline(
    dl_op_name=dl_op_name, namespace=namespace, df_me=df_me,
):
    """
    Define ContainerOp in tfjob_ops and run this pipeline.

    Args:
        name(str): input pipeline name.
        namespace(str): it must be "kubeflow-user-example-com".
        df_me(str): Specifies the path of df_me.pkl where preprocessing has been completed.
    """

    # Run the hyperparameter tuning with Katib.
    dl_katib_op = dl_ops.create_katib_experiment_task(dl_op_name, namespace, df_me)

    # Run the distributive training with TFJob.
    tfjob_op = dl_ops.create_tfjob_task(name, namespace, dl_katib_op, df_me)

    # Create the KFServing inference.
    # 예측모델 서빙은 필요하지 않을 것 같아 일단 제외합니다(20220406~)
    # dl_ops.create_kfserving_task(name, namespace, tfjob_op)


if __name__ == "__main__":
    import kfp.compiler as compiler

    compiler.Compiler().compile(tfjob_pipeline, __file__[:-3] + ".tar.gz")
