# Copyright 2020 XAMES3. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ======================================================================
"""Utility for simplifing streaming using OpenCV."""

from math import cos, radians, sin
from typing import Any, Optional, Tuple, Union

# TODO(xames3): Remove suppressed pylint warnings.
# pyright: reportMissingImports=false
import numpy as np
from cv2 import INTER_AREA, resize


def rescale(feed: np.ndarray, width: Optional[int] = 500) -> np.ndarray:
  """Rescale display feed as per requirement."""
  ratio = width / feed.shape[1]
  dimensions = (width, int(feed.shape[0] * ratio))
  return resize(feed, dimensions, interpolation=INTER_AREA)
