from kedro.pipeline import Pipeline, node
from .nodes.modelisationSVM import modelisationSVM
from .nodes.modelisationSVM import predictionSVM
from .nodes.modelisationRF import predictionRF
from .nodes.modelisationRF import modelisationRF

# kedro run --pipeline= name


def dataScience_pipeline():
    return Pipeline(
        [
            # SVM
            node(
                func=modelisationSVM,
                inputs=["train_X", "train_Y", "test_X", "test_Y", "params:k", "params:svGamma",
                        "params:svcC", "params:pipelinePolynomialfeaturesDegree", "params:rangeMin", "params:rangeMax", "params:cv"],
                outputs="model_save_svm"
            ),
            node(
                func=predictionSVM,
                inputs=["model_save_svm", "test_X", "test_Y"],
                outputs=["Classification_report_svm", "test_Prediction_svm"]
            ),
            # RF
            node(
                func=modelisationRF,
                inputs=["train_X", "train_Y", "test_X", "test_Y"],
                outputs="model_save_RF"
            ),
            node(
                func=predictionRF,
                inputs=["model_save_RF", "test_X", "test_Y"],
                outputs="Classification_report_RF"
            ),

        ]
    )
