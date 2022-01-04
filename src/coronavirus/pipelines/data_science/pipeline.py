from kedro.pipeline import Pipeline, node
from .nodes.modelisation import modelisation
from .nodes.modelisation import prediction

#kedro run --pipeline= name


def dataScience_pipeline():
    return Pipeline(
        [
            node(
                func=modelisation,
                inputs=["train_X","train_Y","test_X","test_Y"],
                outputs="model_save"
            ),
            node(
                func=prediction,
                inputs=["model_save","test_X","test_Y"],
                outputs="prediction"
            ),
            
        ]
    )
