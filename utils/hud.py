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

from typing import List, Optional, Union, Tuple

import numpy
# TODO(xames3): Remove suppressed pylint warnings.
# pyright: reportMissingImports=false
from cv2 import line, ellipse

from . config import colors


def detected_face(frame: numpy.ndarray,
                  start_xy: Tuple,
                  end_xy: Tuple,
                  thickness: Optional[int] = 1,
                  color: Optional[Union[List, int, str]] = None) -> None:
  """Draw rounded square HUD on the face.

  Draw rounded square around the detected face. The size of the square
  adjusts automatically as per the proximity of the face in reference.

  Args:
    frame: Numpy array from the captured frame.
    x: X-position/coordinate of the face.
    y: Y-position/coordinate of the face.
    w: Width of the face in pixels.
    h: Height of the face in pixels.
    color: Frame color (default: None -> yellow)
  """
  start_x, start_y = start_xy
  end_x, end_y = end_xy
  radius = trimmed = int(end_x / 30)
  color = colors.yellow if color is None else color
  line(frame,
       (start_x + radius, start_y),
       (start_x + radius + trimmed, start_y), color, thickness)
  line(frame,
       (start_x, start_y + radius),
       (start_x, start_y + radius + trimmed), color, thickness)
  ellipse(frame,
          (start_x + radius, start_y + radius),
          (radius, radius), 180, 0, 90, color, thickness)
  line(frame,
       (end_x - radius, start_y),
       (end_x - radius - trimmed, start_y), color, thickness)
  line(frame,
       (end_x, start_y + radius),
       (end_x, start_y + radius + trimmed), color, thickness)
  ellipse(frame,
          (end_x - radius, start_y + radius),
          (radius, radius), 270, 0, 90, color, thickness)
  line(frame,
       (start_x + radius, end_y), 
       (start_x + radius + trimmed, end_y), color, thickness)
  line(frame,
       (start_x, end_y - radius),
       (start_x, end_y - radius - trimmed), color, thickness)
  ellipse(frame,
          (start_x + radius, end_y - radius),
          (radius, radius), 90, 0, 90, color, thickness)
  line(frame,
       (end_x - radius, end_y),
       (end_x - radius - trimmed, end_y), color, thickness)
  line(frame,
       (end_x, end_y - radius),
       (end_x, end_y - radius - trimmed), color, thickness)
  ellipse(frame,
          (end_x - radius, end_y - radius),
          (radius, radius), 0, 0, 90, color, thickness)

  # r = int(w / 20)
  # h = int(h + 0.15 * h)
  # line(frame, (x + r, y), (x + w - r, y), color)
  # line(frame, (x + r, y + h), (x + w - r, y + h), color)
  # line(frame, (x, y + radius), (x, y + h - radius), color)
  # line(frame, (x + w, y + radius), (x + w, y + h - radius), color)
  # ellipse(frame, (x + r, y + radius),(radius, r), 180, 0, 90, color)
  # ellipse(frame, (x + w - r, y + radius),(radius, r), 270, 0, 90, color)
  # ellipse(frame, (x + r, y + h - radius),(radius, r), 90, 0, 90, color)
  # ellipse(frame, (x + w - r, y + h - radius),(radius, r), 0, 0, 90, color)
