"""Project hooks."""
from typing import Any, Dict, Iterable, Optional
from kedro.pipeline import Pipeline
from kedro.config import ConfigLoader
from kedro.framework.hooks import hook_impl
from kedro.io import DataCatalog
from kedro.versioning import Journal

from .pipelines.data_engineering.pipeline import dataEng_pipeline
from .pipelines.data_science.pipeline import dataScience_pipeline


class ProjectHooks:
    @hook_impl
    def register_pipelines(self) -> Dict[str, Pipeline]:
        data_engineering_pipeline = dataEng_pipeline()
        data_science_pipeline = dataScience_pipeline()

        return {
            "de": data_engineering_pipeline,
            "ds" : data_science_pipeline,
            "__default__":  data_engineering_pipeline + data_science_pipeline
        }

    @hook_impl
    def register_config_loader(
        self, conf_paths: Iterable[str], env: str, extra_params: Dict[str, Any],
    ) -> ConfigLoader:
        return ConfigLoader(conf_paths)

    @hook_impl
    def register_catalog(
        self,
        catalog: Optional[Dict[str, Dict[str, Any]]],
        credentials: Dict[str, Dict[str, Any]],
        load_versions: Dict[str, str],
        save_version: str,
        journal: Journal,
    ) -> DataCatalog:
        return DataCatalog.from_config(
            catalog, credentials, load_versions, save_version, journal
        )
