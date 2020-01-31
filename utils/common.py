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
"""Utility for handling common functions."""

import socket
from typing import Optional, Union

import numpy as np
from cv2 import INTER_AREA, VideoCapture, destroyAllWindows, resize

# TODO(xames3): Remove suppressed pylint warnings.
# pyright: reportMissingImports=false
from faceid_v2.config import dev




def check_internet(timeout: Optional[Union[float, int]] = 10.0) -> bool:
    """Check the internet connectivity."""
    # You can find the reference code here:
    # https://github.com/xames3/mle/blob/9ad24ffa97fe361f5c506e7279d598c47fc57534/mle/utils/common.py#L65
    try:
        socket.create_connection(
            (dev.PING_URL, dev.PING_PORT), timeout=timeout)
        return True
    except OSError:
        pass
    return False


def rescale(feed: np.ndarray, width: Optional[int] = 500) -> np.ndarray:
  """Rescale display feed as per requirement."""
  ratio = width / feed.shape[1]
  dimensions = (width, int(feed.shape[0] * ratio))
  return resize(feed, dimensions, interpolation=INTER_AREA)


def disconnect(stream: VideoCapture, leave: Optional[bool] = False) -> None:
  """Disconnect the stream and close all instances."""
  destroyAllWindows()
  stream.stop()
  print('Stream terminated.')
  if leave:
    exit(1)


def stream_status(stream: VideoCapture) -> Union[bool, None]:
  """Check if stream is live."""
  # TODO(xames3): Fix "XXXX is not a known member of module" warning.
  try:
    if check_internet():
      return stream is not None and stream.isOpened()
  except:
    return False
