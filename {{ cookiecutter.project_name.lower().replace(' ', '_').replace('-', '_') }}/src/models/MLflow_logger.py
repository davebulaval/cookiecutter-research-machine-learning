import os
import warnings
from typing import Dict, Union
from mlflow import pytorch, set_tracking_uri, start_run, log_params, log_metric, log_param, end_run, active_run, \
    create_experiment, get_experiment_by_name
from mlflow.exceptions import MlflowException
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from poutyne.framework import Logger

warnings.filterwarnings('ignore')
# https://github.com/ymym3412/Hydra-MLflow-experiment-management/blob/c0bef5b5117970dd2a465b97504a5478b3eff704/mlflow_writer.py
class MlFlowWriter(Logger):
    def __init__(self,
                 experiment_name: str,
                 tracking_path: str,
                 batch_granularity: bool = False,
                 same_run_logging: bool = True) -> None:
        super().__init__(batch_granularity=batch_granularity)
        set_tracking_uri(tracking_path)

        self.same_run_logging = same_run_logging
        self._handle_experiment_id(experiment_name)
        self.run_id = start_run(experiment_id=self.experiment_id).info.run_id

    def log_config_params(self, params: Union[Dict, DictConfig, ListConfig]) -> None:
        if isinstance(params, Dict):
            log_params(params)
        else:
            for param_name, element in params.items():
                self._log_config_write(param_name, element)

    def _log_config_write(self, parent_name: str, element: Union[Dict, ListConfig]) -> None:
        if isinstance(element, DictConfig):
            for key, value in element.items():
                if isinstance(value, DictConfig) or isinstance(value, ListConfig):
                    self._log_config_write('{}.{}'.format(parent_name, key), value)
                else:
                    log_param('{}.{}'.format(parent_name, key), value)
        elif isinstance(element, ListConfig):
            for idx, value in enumerate(element):
                log_param('{}.{}'.format(parent_name, idx), value)

    def _on_train_batch_end_write(self, batch_number: int, logs: Dict) -> None:
        if self.batch_granularity:
            for key, value in logs.items():
                log_metric(key, value, step=batch_number)

    def _on_epoch_end_write(self, epoch_number: int, logs: Dict) -> None:
        logs.pop('epoch')
        for key, value in logs.items():
            log_metric(key, value, step=epoch_number)

    def on_train_end(self, logs: Dict):
        self._on_train_end_write(logs)
        end_run()

    def _on_train_end_write(self, logs) -> None:
        last_epoch = self.params['epochs']
        log_metric('last-epoch', last_epoch)

    def on_test_begin(self, logs: Dict) -> None:
        if self.same_run_logging:
            start_run(run_id=self.run_id)  # take the previous know run
        else:
            start_run(experiment_id=self.experiment_id)  # start a new run

    def on_test_end(self, logs: Dict):
        self._on_test_end_write(logs)
        end_run()

    def _on_test_end_write(self, logs: Dict) -> None:
        for key, value in logs.items():
            log_metric('test-{}'.format(key), value)

    def log_model(self):  # todo management of device cpu and gpu go back
        # device = self.model.device
        with active_run():
            pytorch.log_model(self.model.network, 'trained-model')
        # self.model.to(device)

    def _handle_experiment_id(self, experiment_name):
        try:
            self.experiment_id = create_experiment(experiment_name)
        except MlflowException:
            self.experiment_id = get_experiment_by_name(experiment_name).experiment_id


class LocalMlFlowWriter(MlFlowWriter):
    def __init__(self,
                 root_path: str,
                 local_path: str,
                 experiment_name: str,
                 batch_granularity: bool = False,
                 same_run_logging: bool = True) -> None:

        local_path = os.path.join(root_path, local_path) if root_path is not None else local_path
        full_tracking_path = 'file:{}'.format(local_path)

        super().__init__(experiment_name=experiment_name,
                         tracking_path=full_tracking_path,
                         batch_granularity=batch_granularity,
                         same_run_logging=same_run_logging)

class ServerMlFlowWriter(MlFlowWriter):
    def __init__(self,
                 experiment_name: str,
                 tracking_uri: str,
                 batch_granularity: bool = False,
                 same_run_logging: bool = True) -> None:

        super().__init__(experiment_name=experiment_name,
                         tracking_path=tracking_uri,
                         batch_granularity = batch_granularity,
                         same_run_logging = same_run_logging)
