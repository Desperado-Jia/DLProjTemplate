# Copyright (C) 2020 Tong Jia cecilio.jia@gmail.com. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
import copy
import json
import logging
from typing import Any, Dict


logger = logging.getLogger(name=__name__)


class BaseConfig(object):
    r"""Base class for all configuration classes, i.e., :class:`XXXModelingConfig`.
    :class:`~BaseConfig` takes care of storing the configuration of the modeling modules and handles methods for
    loading/saving object.

    Example usage::
        ```python
        class XXXConfig(BaseConfig):
            def __init__(vocab_size, hidden_size, ..., **kwargs):
                super(XXXConfig, self).__init__(**kwargs)
                self.vocab_size = vocab_size
                self.hidden_size = hidden_size
                ...
        ```
    """
    def __init__(self, **kwargs):
        for (key, value) in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as error:
                logger.error(msg="Can't set attribute {} with value {} for class {}".format(key, value, self))
                raise error

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "BaseConfig":
        """Instantiates a class:`BaseConfig` from a Python dictionary of hyperparameters."""
        config = cls(**config_dict)
        return config

    @classmethod
    def from_json(cls, json_dir: str) -> "BaseConfig":
        """Instantiates a class:`BaseConfig` from the path to a JSON file of hyperparameters."""
        with open(file=json_dir, mode="r", encoding="utf-8") as f:
            config_dict = json.load(fp=f)
        return cls.from_dict(config_dict=config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes this instance to a Python dictionary."""
        out_dict = copy.deepcopy(x=self.__dict__)
        return out_dict

    def to_json(self, json_dir: str):
        """Save this instance to a JSON file."""
        with open(file=json_dir, mode="w") as f:
            json.dump(obj=self.__dict__, fp=f, indent=4, sort_keys=False)
