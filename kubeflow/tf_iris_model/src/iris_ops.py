from kfp import components

from kubeflow.katib import ApiClient
from kubeflow.katib import V1beta1ExperimentSpec
from kubeflow.katib import V1beta1AlgorithmSpec
from kubeflow.katib import V1beta1ObjectiveSpec
from kubeflow.katib import V1beta1ParameterSpec
from kubeflow.katib import V1beta1FeasibleSpace
from kubeflow.katib import V1beta1TrialTemplate
from kubeflow.katib import V1beta1TrialParameterSpec

from pytz import timezone, utc
from datetime import datetime as dt

KST = timezone("Asia/Seoul")
pipeline_start_date_str = utc.localize(dt.utcnow()).astimezone(KST).strftime("%Y%m%d")


def create_katib_experiment_task(experiment_name, experiment_namespace):
    """
    Args:

    Retruns:
        op: return create katib experiment ContainerOp
    """

    # Trial count specification.
    max_trial_count = 12
    max_failed_trial_count = 10
    parallel_trial_count = 3

    # Objective specification.
    objective = V1beta1ObjectiveSpec(
        type="maximize", goal=1.0, objective_metric_name="accuracy"
    )


    # Algorithm specification.
    algorithm = V1beta1AlgorithmSpec(
        algorithm_name="random",
    )

    # Parameters specification.
    parameters = [
        V1beta1ParameterSpec(
            name="batch_size",
            parameter_type="int",
            feasible_space=V1beta1FeasibleSpace(min="16", max="64", step="16"),
        ),
        V1beta1ParameterSpec(
            name="epoch",
            parameter_type="int",
            feasible_space=V1beta1FeasibleSpace(min="100", max="500", step="100"),
        ),
        V1beta1ParameterSpec(
            name="optimizer",
            parameter_type="categorical",
            feasible_space=V1beta1FeasibleSpace(list=["rmsprop", "sgd", "adam"]),
        )
    ]
    
    data_pvc_name = "data-claim"
    # Experiment Trial template.
    trial_spec = {
        "apiVersion": "kubeflow.org/v1",
        "kind": "TFJob",
        "spec": {
            "tfReplicaSpecs": {
                "Worker": {
                    "replicas": 2,
                    "restartPolicy": "OnFailure",
                    "template": {
                        "metadata": {
                            "annotations": {"sidecar.istio.io/inject": "false"}
                        },
                        "spec": {
                            "containers": [
                                {
                                    "name": "tensorflow",
                                    "image": "docker.io/moey920/tf_iris_model:latest",
                                    "command": [
                                        "python",
                                        "/code/src/train_with_preprocess.py",
                                        "--batch_size=${trialParameters.batchSize}",
                                        "--epoch=${trialParameters.epoch}",
                                        "--optimizer=${trialParameters.optimizer}"
                                    ],
                                    "volumeMounts": [
                                        {
                                            "mountPath": "/code/data",
                                            "name": "data-volume",
                                        },
                                    ],
                                }
                            ],
                            "volumes": [
                                {
                                    "name": "data-volume",
                                    "persistentVolumeClaim": {
                                        "claimName": data_pvc_name
                                    },
                                },
                            ],
                        },
                    },
                },
            }
        },
    }

    # Configure parameters for the Trial template.
    trial_template = V1beta1TrialTemplate(
        primary_container_name="tensorflow",
        trial_parameters=[
            V1beta1TrialParameterSpec(
                name="batchSize",
                description="batch_size",
                reference="batch_size",
            ),
            V1beta1TrialParameterSpec(
                name="epoch",
                description="epoch",
                reference="epoch",
            ),
            V1beta1TrialParameterSpec(
                name="optimizer",
                description="optimizer",
                reference="optimizer",
            ),
        ],
        trial_spec=trial_spec,
    )

    # Create an Experiment from the above parameters.
    experiment_spec = V1beta1ExperimentSpec(
        max_trial_count=max_trial_count,
        max_failed_trial_count=max_failed_trial_count,
        parallel_trial_count=parallel_trial_count,
        objective=objective,
        algorithm=algorithm,
        parameters=parameters,
        trial_template=trial_template,
    )

    # Create the KFP task for the Katib Experiment.
    # Experiment Spec should be serialized to a valid Kubernetes object.
    katib_experiment_launcher_op = components.load_component_from_url(
        "https://raw.githubusercontent.com/kubeflow/pipelines/master/components/kubeflow/katib-launcher/component.yaml"
    )
    op = katib_experiment_launcher_op(
        experiment_name=experiment_name,
        experiment_namespace=experiment_namespace,
        experiment_spec=ApiClient().sanitize_for_serialization(experiment_spec),
        experiment_timeout_minutes=600,
        delete_finished_experiment=False,
    )

    return op


"""
This function converts the Katib Experiment HP result to args.

Args: 
    katib_results(json)

Retruns:
    best_hps = list("--input_width=6", "--label_width=4", "--shift=5", "--mae=0.0376")
    to convert " ".join(best_hps): return best hyper parameters like "--input_width=6 --label_width=4 --shift=5 --mae=0.0376"
"""
def convert_katib_results(katib_results) -> str:

    import json
    import pprint

    katib_results_json = json.loads(katib_results)
    print("Katib results:")
    pprint.pprint(katib_results_json)
    best_hps = []
    for pa in katib_results_json["currentOptimalTrial"]["parameterAssignments"]:
        if pa["name"] == "batch_size":
            best_hps.append("--batch_size=" + pa["value"])
        elif pa["name"] == "epoch":
            best_hps.append("--epoch=" + pa["value"])
        elif pa["name"] == "optimizer":
            best_hps.append("--optimizer=" + pa["value"])
    print("Best Hyperparameters: {}".format(best_hps))

    return " ".join(best_hps)


def create_tfjob_task(tfjob_name, tfjob_namespace, katib_op):
    """
    convert_katib_results returns the format ["--input_width=6", "--label_width=4", "--shift=5", "--model_name=baseline"]
    The results are stored in best_hp_op. Save the above to best_hps in str format.

    Args:
        tfjob_name (str) : Declare it as a global variable.
        tfjob_namespace (str) : Declare it as a global variable.
        katib_op : Take the ContainerOp to create earlier.
        model_volume_op : Take the ContainerOp to create earlier.

    Returns:
        op : Returns a TFjob ContainerOp.
    """
    import json

    convert_katib_results_op = components.func_to_container_op(convert_katib_results)
    best_hp_op = convert_katib_results_op(katib_op.output)
    best_hps = str(best_hp_op.output)
    print("best_hps:", best_hps)


    data_pvc_name = "data-claim"
    # Generate TFJob Chief and Worker specs with the best hyperparameters.
    tfjob_chief_spec = {
        "replicas": 1,
        "restartPolicy": "OnFailure",
        "template": {
            "metadata": {"annotations": {"sidecar.istio.io/inject": "false"}},
            "spec": {
                "containers": [
                    {
                        "name": "tensorflow",
                        "image": "docker.io/moey920/tf_iris_model:latest",
                        "command": ["sh", "-c"],
                        "args": [
                            f"python /code/src/train_with_preprocess.py --save=/code/data {best_hps}"
                        ],
                        "volumeMounts": [
                            {
                                "mountPath": "/code/data",
                                "name": "data-volume",
                            },
                        ],
                    }
                ],
                "volumes": [
                    {
                        "name": "data-volume",
                        "persistentVolumeClaim": {"claimName": data_pvc_name},
                    },
                ],
            },
        },
    }

    tfjob_worker_spec = {
        "replicas": 1,
        "restartPolicy": "OnFailure",
        "template": {
            "metadata": {"annotations": {"sidecar.istio.io/inject": "false"}},
            "spec": {
                "containers": [
                    {
                        "name": "tensorflow",
                        "image": "docker.io/moey920/tf_iris_model:latest",
                        "command": ["sh", "-c"],
                        "args": [
                            f"python /code/src/train_with_preprocess.py --save=/code/data {best_hps}"
                        ],
                        "volumeMounts": [
                            {
                                "mountPath": "/code/data",
                                "name": "data-volume",
                            },
                        ],
                    }
                ],
                "volumes": [
                    {
                        "name": "data-volume",
                        "persistentVolumeClaim": {"claimName": data_pvc_name},
                    },
                ],
            },
        },
    }

    # Create a KFP job for the TFJob.
    tfjob_launcher_op = components.load_component_from_url(
        "https://raw.githubusercontent.com/moey920/GitOps/main/tfjob_launcher.yaml"
    )
    op = tfjob_launcher_op(
        name=tfjob_name,
        namespace=tfjob_namespace,
        chief_spec=json.dumps(tfjob_chief_spec),
        worker_spec=json.dumps(tfjob_worker_spec),
        tfjob_timeout_minutes=300,
        delete_finished_tfjob=True,
    )

    return op.after(best_hp_op)


# In Arguments you must define the model name, namespace, TFJob, and the output of the model volume job.
def create_kfserving_task(name, namespace, tfjob_op):
    """
    Create a Kubeflow Pipelines job for KFServing inference.

    Args:
        name(str): KatibOp의 이름입니다.
        namespace(str): kubeflow-user-example-com만 사용합니다.
        tfjob_op: create_tfjob_task의 리턴으로 받아오는 ContainerOp입니다.
    """

    data_pvc_name = "data-claim"

    inference_service = """
apiVersion: "serving.kubeflow.org/v1beta1"
kind: "InferenceService"
metadata:
    name: {}
    namespace: {}
    annotations:
        "sidecar.istio.io/inject": "false"
spec:
    predictor:
        tensorflow:
            runtimeVersion: "2.8.0"
            storageUri: "pvc://{}/{}/"
""".format(
        name, namespace, data_pvc_name, pipeline_start_date_str
    )

    kfserving_launcher_op = components.load_component_from_url(
        "https://raw.githubusercontent.com/kubeflow/pipelines/master/components/kubeflow/kfserving/component.yaml"
    )
    op = kfserving_launcher_op(
        action="create",
        canary_traffic_percent="10",
        inferenceservice_yaml=inference_service,
    ).after(tfjob_op)

    return op
