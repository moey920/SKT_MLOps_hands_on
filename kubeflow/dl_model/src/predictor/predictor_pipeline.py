import kfp.dsl as dsl
import sys, os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import predictor_ops


@dsl.pipeline(name="predictor_Pipeline", description="AutoML_Pipeline Kubeflow v3")
def Automl_Pipeline():

    op_dict = {}
    op_dict["forecasting_predictor_op"] = predictor_ops.exec_predictor()


if __name__ == "__main__":
    import kfp.compiler as compiler

    compiler.Compiler().compile(Automl_Pipeline, __file__[:-3] + ".tar.gz")
