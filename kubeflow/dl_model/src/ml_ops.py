from kfp import components

from kubeflow.katib import ApiClient
from kubeflow.katib import V1beta1ExperimentSpec
from kubeflow.katib import V1beta1AlgorithmSpec
from kubeflow.katib import V1beta1ObjectiveSpec
from kubeflow.katib import V1beta1ParameterSpec
from kubeflow.katib import V1beta1FeasibleSpace
from kubeflow.katib import V1beta1TrialTemplate
from kubeflow.katib import V1beta1TrialParameterSpec
from kubeflow.katib import KatibClient


def create_katib_experiment_task(experiment_name, experiment_namespace, df_me):
    """

    Args:

    Retruns:
        op: return create katib experiment ContainerOp
    """

    # Trial count specification.
    max_trial_count = 6
    max_failed_trial_count = 5
    parallel_trial_count = 3
    # max_trial_count = 2
    # max_failed_trial_count = 2
    # parallel_trial_count = 2

    # Objective specification.
    objective = V1beta1ObjectiveSpec(
        type="minimize", goal=0.01, objective_metric_name="mean_absolute_error"
    )

    # Algorithm specification.
    algorithm = V1beta1AlgorithmSpec(algorithm_name="grid",)

    # Parameters specification.
    parameters = [
        V1beta1ParameterSpec(
            name="model_name",
            parameter_type="categorical",
            feasible_space=V1beta1FeasibleSpace(
                list=[
                    "xgb_regressor",
                    "xgbrf_regressor",
                    "linear_regression",
                    "kneighbors_regressor",
                    "linear_svr",
                    "linear_tree_regressor",
                    "extra_trees_regressor",
                ]
            ),
        ),
    ]

    data_pvc_name = "data-claim"

    # Experiment Trial template.
    trial_spec = {
        "apiVersion": "kubeflow.org/v1",
        "kind": "TFJob",
        "spec": {
            "tfReplicaSpecs": {
                "Chief": {
                    "replicas": 1,
                    "restartPolicy": "OnFailure",
                    "template": {
                        "metadata": {
                            "annotations": {"sidecar.istio.io/inject": "false"}
                        },
                        "spec": {
                            "containers": [
                                {
                                    "name": "tensorflow",
                                    "image": "docker.io/moey920/train_forecasting_model:latest",
                                    "command": [
                                        "python",
                                        "/code/ml_models.py",
                                        f"--df_me={df_me}",
                                        "--model_name=${trialParameters.modelName}",
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
                "Worker": {
                    "replicas": 1,
                    "restartPolicy": "OnFailure",
                    "template": {
                        "metadata": {
                            "annotations": {"sidecar.istio.io/inject": "false"}
                        },
                        "spec": {
                            "containers": [
                                {
                                    "name": "tensorflow",
                                    "image": "docker.io/moey920/train_forecasting_model:latest",
                                    "command": [
                                        "python",
                                        "/code/ml_models.py",
                                        f"--df_me={df_me}",
                                        "--model_name=${trialParameters.modelName}",
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
                name="modelName", description="model_name", reference="model_name",
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
        if pa["name"] == "model_name":
            best_hps.append("--model_name=" + pa["value"])
    best_hps.append(
        "--mean_absolute_error="
        + katib_results_json["currentOptimalTrial"]["observation"]["metrics"][0]["min"]
    )
    print("Best Hyperparameters: {}".format(best_hps))

    return " ".join(best_hps)


def katib_logger(name, namespace):

    from kubeflow.katib import KatibClient

    kclient = KatibClient()
    trial_details_log = kclient.get_success_trial_details(
        name=name, namespace=namespace
    )
    optimal_trial_details_log = kclient.get_optimal_hyperparameters(
        name=name, namespace=namespace
    )
    print("trial_details_log :", trial_details_log)
    print("optimal_trial_details_log :", optimal_trial_details_log)
    delete_trial_log = kclient.delete_experiment(name=name, namespace=namespace)
    print("optimal_trial_details_log :", delete_trial_log)


def create_tfjob_task(tfjob_name, tfjob_namespace, ml_katib_op, df_me):
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

    ml_best_hp_op = convert_katib_results_op(ml_katib_op.output)

    best_hps_dict = {
        "ml": str(ml_best_hp_op.output),
    }

    config_op_packages = ["kubeflow-katib", "kfp"]

    katib_logger_op = components.func_to_container_op(
        katib_logger, packages_to_install=config_op_packages
    )

    katib_logger_result = katib_logger_op(
        name="forecasting-ml-model", namespace=tfjob_namespace
    ).after(ml_best_hp_op)

    print(katib_logger_result)

    # best_hps_1 = sorted(best_hps_dict.items(), key=lambda x: x[1][-5:])[0][1]
    best_hps = best_hps_dict["ml"]

    print("best_hps:", best_hps)

    # pvc_name = str(model_volume_op.outputs["name"])
    pvc_name = "forecasting-claim"
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
                        "image": "docker.io/moey920/train_forecasting_model:latest",
                        "command": ["sh", "-c"],
                        "args": [
                            "python /code/ml_models.py --save=/mnt/export --df_me={} {}".format(
                                df_me, best_hps
                            )
                        ],
                        "volumeMounts": [
                            {
                                "mountPath": "/mnt/export",
                                "name": "forecasting-model-volume",
                            },
                            {"mountPath": "/code/data", "name": "data-volume",},
                        ],
                    }
                ],
                "volumes": [
                    {
                        "name": "forecasting-model-volume",
                        "persistentVolumeClaim": {"claimName": pvc_name},
                    },
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
                        "image": "docker.io/moey920/train_forecasting_model:latest",
                        "command": ["sh", "-c"],
                        "args": [
                            "python /code/ml_models.py --save=/mnt/export --df_me={} {}".format(
                                df_me, best_hps
                            )
                        ],
                        "volumeMounts": [
                            {
                                "mountPath": "/mnt/export",
                                "name": "forecasting-model-volume",
                            },
                            {"mountPath": "/code/data", "name": "data-volume",},
                        ],
                    }
                ],
                "volumes": [
                    {
                        "name": "forecasting-model-volume",
                        "persistentVolumeClaim": {"claimName": pvc_name},
                    },
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

    return op.after(ml_best_hp_op)


# In Arguments you must define the model name, namespace, TFJob, and the output of the model volume job.
def create_kfserving_task(name, namespace, tfjob_op):
    """
    Create a Kubeflow Pipelines job for KFServing inference.

    Args:
        name(str): KatibOp의 이름입니다.
        namespace(str): kubeflow-user-example-com만 사용합니다.
        tfjob_op: create_tfjob_task의 리턴으로 받아오는 ContainerOp입니다.
        model_volume_op: dsl.VolumeOp로 생성한 ConatainerOp입니다.
    """

    pvc_name = "forecasting-claim"

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
            storageUri: "pvc://{}/"
""".format(
        name, namespace, pvc_name
    )

    kfserving_launcher_op = components.load_component_from_url(
        "https://raw.githubusercontent.com/kubeflow/pipelines/master/components/kubeflow/kfserving/component.yaml"
    )
    op = kfserving_launcher_op(
        action="create",
        framework="sklearn",
        canary_traffic_percent="10",
        inferenceservice_yaml=inference_service,
    ).after(tfjob_op)

    return op
