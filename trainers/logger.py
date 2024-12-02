# If used in your project please cite this work as described in the README file.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


from typing import Callable
import tianshou.utils.logger.base as base


class CustomLogger(base.BaseLogger):
    """Logger that reports only the most important results during training and validation,
    compatible with the mode of reward reporting in the BeamManagement6G environment."""

    def __init__(self, train_interval, test_interval) -> None:
        super().__init__(train_interval=train_interval, test_interval=test_interval)
        self.train_returns = []
        self.test_returns = []

    def prepare_dict_for_logging(self, data: dict[str, base.VALID_LOG_VALS_TYPE], **kwargs) -> dict[str, base.VALID_LOG_VALS_TYPE]:
        return data.get('returns')

    def write(self, step_type: str, step: int, data: dict[str, base.VALID_LOG_VALS_TYPE]) -> None:
        if 'train/env_step' == step_type:
            if data is None:
                raise RuntimeError('No train rewards could be logged.')
            self.train_returns.append(data)
        elif 'test/env_step' == step_type:
            if data is None:
                raise RuntimeError('No test rewards could be logged.')
            self.test_returns.append(data)
        else:
            pass

    def get_train_returns(self):
        return self.train_returns

    def get_test_returns(self):
        return self.test_returns

    def save_data(
        self,
        epoch: int,
        env_step: int,
        gradient_step: int,
        save_checkpoint_fn: Callable[[int, int, int], str] | None = None,
    ) -> None:
        pass

    def restore_data(self) -> tuple[int, int, int]:
        return 0, 0, 0

    def restore_logged_data(self, log_path: str) -> dict:
        return {}
