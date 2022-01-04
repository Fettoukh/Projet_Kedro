from kedro.pipeline import Pipeline, node
from .nodes.preprocessing import remove_Empty_Columns
from .nodes.preprocessing import encodage
from .nodes.preprocessing import feature_engineering
from .nodes.preprocessing import imputation
from .nodes.preprocessing import data_split

#kedro run --pipeline= name


def dataEng_pipeline():
    return Pipeline(
        [
            node(
                func=remove_Empty_Columns,
                inputs=["initial_dataset"],
                outputs="subset_dataset",
            ),
            node(
                func=encodage,
                inputs=["subset_dataset"],
                outputs="encoded_dataset",
            ),
            node(
                func=feature_engineering,
                inputs=["encoded_dataset", "initial_dataset"],
                outputs="feature_engineered_dataset",
            ),
            node(
                func=imputation,
                inputs=["feature_engineered_dataset"],
                outputs="imputation_dataset",
            ),

            node(
                func=data_split,
                inputs=["imputation_dataset" , "params:test_size"],
                outputs=["train_X","test_X","train_Y","test_Y"],
            ),
        ]
    )
