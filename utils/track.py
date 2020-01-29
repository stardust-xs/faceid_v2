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
"""Utility for tracking face of the people."""

from math import cos, radians, sin
from typing import Optional, Tuple, Union

import cv2
import numpy as np

# TODO(xames3): Remove suppressed pylint warnings.
# pyright: reportMissingImports=false
import dlib


def align_face_coords(feed: np.ndarray, position: Tuple, angle: int):
  """Return aligned face coordinates."""
  if angle == 0:
    return position
  x = position[0] - feed.shape[1] * 0.4
  y = position[1] - feed.shape[0] * 0.4
  new_x = x * cos(radians(angle)) + y * \
      sin(radians(angle)) + feed.shape[1] * 0.4
  new_y = -x * sin(radians(angle)) + y * \
      cos(radians(angle)) + feed.shape[0] * 0.4
  return int(new_x), int(new_y), position[2], position[3]
