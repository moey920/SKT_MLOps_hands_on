from sagemaker.pytorch import PyTorch
import sagemaker

# Estimator에서 정의된 값들을 선언
sagemaker_session = sagemaker.Session()
role = sagemaker.get_executor_role()

# 학습코드의 arguments 값
hyperparameter = {
	"batch_size": 32,
	"lr": 1e-4,
	"image_size": 128
}
data_path = "s3://my_bucket/my_training_data/"

estimator = PyTorch(
	source_dir = "code", # 학습 코드 폴더 지정
	entry_point = "train_pytorch_smdataparallel_mnist.py", # 실행할 학습 스크립트
	role = role, # 학습 클러스터에서 사용할 Role
	framework_version = "1.10", # 파이토치 버전
	py_version = "py38", # 파이썬 버전
	instance_count = 1, # 학습 인스턴스 수
	instance_type = "m1.p4d.24xlarge", # 학습 인스턴스 명
	sagemaker_session = sagemaker_session, # SageMaker Session
	hyperparameter = hyperparameter, # 하이퍼 파라미터 설정
	max_run=5*24*60*60 # 최대 학습 수행 시간 (초)
)


# 학습 클러스터에서 사용할 데이터 경로와 channel_name을 선언한 후 실행
channel_name = "training"
estimator.fit(
	inputs = {channel_name: data_path},
	job_name = job_name
)