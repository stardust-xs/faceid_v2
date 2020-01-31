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
"""Core utility for overlaying HUDs on top of the detected faces."""

from typing import List, Optional, Tuple, Union

# TODO(xames3): Remove suppressed pylint warnings.
# pyright: reportMissingImports=false
import numpy as np
from cv2 import rectangle

from faceid_v2.config import colors


def detect_face(frame: np.ndarray,
                start_xy: Tuple,
                end_xy: Tuple,
                color: Optional[Union[List, int, str]] = None,
                thickness: Optional[int] = 1) -> None:
  """Draw bounding box around the detected object.

  The size of the bounding box adjusts automatically as per the size of
  the object in reference.

  Args:
    frame: Numpy array from the captured frame.
    start_xy: Tuple of top left coordinates.
    end_xy: Tuple of bottom right coordinates.
    color: Frame color (default: None -> yellow)
    thickness: Thickness of the bounding box.
  """
  color = colors.yellow if color is None else color
  return rectangle(frame, start_xy, end_xy, color, thickness)


def detect_motion(frame: np.ndarray,
                  top_x: Union[int],
                  top_y: Union[int],
                  btm_x: Union[int],
                  btm_y: Union[int],
                  color: Optional[Union[List, int, str]] = None,
                  thickness: Optional[int] = 1) -> None:
  """Draw bounding box around the detected object.

  The size of the bounding box adjusts automatically as per the size of
  the object in reference.

  Args:
    frame: Numpy array from the captured frame.
    start_xy: Tuple of top left coordinates.
    end_xy: Tuple of bottom right coordinates.
    color: Frame color (default: None -> yellow)
    thickness: Thickness of the bounding box.
  """
  color = colors.green if color is None else color
  return rectangle(frame,
                   (top_x, top_y),
                   (top_x + btm_x, top_y + btm_y),
                   color,
                   thickness)
