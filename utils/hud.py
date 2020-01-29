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

from typing import List, Optional, Union

import numpy
# TODO(xames3): Remove suppressed pylint warnings.
# pyright: reportMissingImports=false
from cv2 import line, ellipse


def detected_face(feed: numpy.ndarray,
                  x: int,
                  y: int,
                  w: int,
                  h: int,
                  color: Optional[Union[List, str]] = None) -> None:
  """Draw rounded square HUD on the face.

  Draw rounded square around the detected face. The size of the square
  adjusts automatically as per the proximity of the face in reference.

  Args:
    feed: Numpy array from the captured feed.
    x: X-position/coordinate of the face.
    y: Y-position/coordinate of the face.
    w: Width of the face in pixels.
    h: Height of the face in pixels.
    color: Frame color (default: None -> yellow)
  """
  r = int(w / 20)
  h = int(h + 0.15 * h)
  color = [0, 204, 255] if color is None else color
  line(feed, (x + r, y), (x + w - r, y), color)
  line(feed, (x + r, y + h), (x + w - r, y + h), color)
  line(feed, (x, y + r), (x, y + h - r), color)
  line(feed, (x + w, y + r), (x + w, y + h - r), color)
  ellipse(feed, (x + r, y + r), (r, r), 180, 0, 90, color)
  ellipse(feed, (x + w - r, y + r), (r, r), 270, 0, 90, color)
  ellipse(feed, (x + r, y + h - r), (r, r), 90, 0, 90, color)
  ellipse(feed, (x + w - r, y + h - r), (r, r), 0, 0, 90, color)
