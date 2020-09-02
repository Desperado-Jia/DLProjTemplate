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
import numpy as np
from collections import deque, defaultdict
from typing import Deque, DefaultDict


class RunningAverageValue(object):
    def __init__(self, window_size: int = 20):
        self.deque: Deque[float] = deque(maxlen=window_size)
        self.total: float = 0.0
        self.steps: int = 0

    def update(self, value: float):
        self.deque.append(value)
        self.total += value
        self.steps += 1

    @property
    def local_median(self):
        return np.median(a=self.deque, axis=0, keepdims=False)

    @property
    def local_average(self):
        return np.mean(a=self.deque, axis=0, dtype=np.float, keepdims=False)

    @property
    def global_average(self):
        return self.total / self.steps

    @property
    def local_maximum(self):
        return np.max(a=self.deque, axis=0, keepdims=False)

    @property
    def value(self):
        return self.deque[-1]


class MetricSummarizer(object):
    def __init__(self, delimiter: str = "\t"):
        self.metrics: DefaultDict[str, RunningAverageValue] = defaultdict(RunningAverageValue)
        self.delimiter = delimiter

    def register_meter(self, name: str, meter: RunningAverageValue):
        try:
            self.metrics.setdefault(name, meter)
        except:
            raise ValueError


if __name__ == '__main__':
    # deque = deque(maxlen=3)
    # deque.append(1.0)
    # deque.append(2.0)
    # deque.append(3.0)
    # deque.append(4.0)
    # a = np.median(a=deque, axis=0, keepdims=False)

    loss = RunningAverageValue(window_size=3)
    loss.update(value=1.0)
    loss.update(value=2.0)
    loss.update(value=3.0)
    loss.update(value=4.0)

    a = loss.local_average
    b = loss.global_average