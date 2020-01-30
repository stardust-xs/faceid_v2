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
"""The `faceid_v2.face` module."""

# TODO(xames3): Remove suppressed pylint warnings.
# pyright: reportMissingImports=false
import numpy as np
import time
import cv2
import imutils
from imutils.video import VideoStream

from utils.hud import detected_face_ml
from utils.stream import rescale
from utils.config.dev import PROTOTEXT, CAFFEMODEL

model = cv2.dnn.readNetFromCaffe(PROTOTEXT, CAFFEMODEL)

blur = False
stream = VideoStream(src=0).start()
time.sleep(2.0)

while True:
  frame = stream.read()
  frame = imutils.resize(frame, width=400)
  h, w = frame.shape[:2]
  face_blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                    (300, 300), (104.0, 177.0, 123.0))
  model.setInput(face_blob)
  detections = model.forward()
  for idx in range(detections.shape[2]):
    confidence = detections[0, 0, idx, 2]
    if confidence < 0.7:
      continue
    bounding_box = detections[0, 0, idx, 3:7] * np.array([w, h, w, h])
    start_x, start_y, end_x, end_y = bounding_box.astype('int')
    
    if blur:
      frame[start_y:end_y, start_x:end_x] = cv2.blur(
          frame[start_y:end_y, start_x:end_x], (25, 25))

    detected_face_ml(frame, (start_x, start_y), (end_x, end_y))
  cv2.imshow('Face tracking', rescale(frame))

  if cv2.waitKey(5) & 0xFF == ord('b'):
    blur = not blur

  if cv2.waitKey(5) & 0xFF == int(27):
    exit(1)
    break

cv2.destroyAllWindows()
stream.stop()
