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

# TODO(xames3): Remove suppressed pylint warnings.
# pyright: reportMissingImports=false
import numpy as np
from cv2 import INTER_LINEAR, getRotationMatrix2D, warpAffine


def align_face(feed: np.ndarray, angle: int) -> np.ndarray:
  """Align face when tilted.
  
  Align and cover a particular range of tilt of a face.

  Args:
    feed: Stream feed read by camera.
    angle: Angle of tilt to compensate.

  Returns:
    Numpy array of the compensating coordinates for the HUD.

  Example:
    for angle in [0, -30, 30]:
      tilted_face = align_face(gray_feed, angle)
      faces = frontal_face.detectMultiScale(tilted_face, 1.3, 5)
  """
  if angle == 0:
    return feed
  height, width = feed.shape[:2]
  rot_mat = getRotationMatrix2D((width / 2, height / 2), angle, 0.9)
  return warpAffine(feed, rot_mat, (width, height), flags=INTER_LINEAR)
