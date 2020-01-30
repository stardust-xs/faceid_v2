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
import random

from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
from cv2 import destroyAllWindows

from utils.config.cascades import eyes, frontal_face
from utils.config.colors import color
from utils.hud import detected_face
from utils.stream import rescale
from utils.track import align_face

model = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt',
                                 'res10_300x300_ssd_iter_140000.caffemodel')

stream = VideoStream(src=0).start()
time.sleep(2.0)

while True:
  frame = stream.read()
  frame = imutils.resize(frame, width=400)
  h, w = frame.shape[:2]
  blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                               (300, 300), (104.0, 177.0, 123.0))
  model.setInput(blob)
  detections = model.forward()
  for idx in range(detections.shape[2]):
    confidence = detections[0, 0, idx, 2]
    if confidence < 0.7:
      continue
    bounding_box = detections[0, 0, idx, 3:7] * np.array([w, h, w, h])
    start_x, start_y, end_x, end_y = bounding_box.astype('int')
    detected_face(frame, (start_x, start_y), (end_x, end_y))
  cv2.imshow('Face tracking', rescale(frame))
  if cv2.waitKey(5) & 0xFF == int(27):
    exit(1)
    break

destroyAllWindows()
stream.stop()


# stream = cv2.VideoCapture(0)

# while True:
#   state, color_feed = stream.read()
#   if state:
#     gray_feed = cv2.cvtColor(color_feed, cv2.COLOR_BGR2GRAY)
#     faces = frontal_face.detectMultiScale(gray_feed, 1.3, 5)
#     for x, y, w, h in faces:
#       if len(faces) == 1:
#         detected_face(color_feed, x, y, w, h)
#       elif len(faces) > 1:
#         detected_face(color_feed, x, y, w, h, random.sample(color, 3))
#     cv2.imshow('Face tracking', rescale(color_feed))
#     if cv2.waitKey(5) & 0xFF == int(27):
#       break
#   else:
#     print('No stream available.')

# stream.release()
# cv2.destroyAllWindows()

# stream = cv2.VideoCapture(0)

# while True:
#   state, color_feed = stream.read()
#   color_feed = rescale(color_feed)
#   if state:
#     gray_feed = cv2.cvtColor(color_feed, cv2.COLOR_BGR2GRAY)
#     faces = frontal_face.detectMultiScale(gray_feed, 1.3, 5)
#     for x, y, w, h in faces:
#       if len(faces) == 1:
#         detected_face(color_feed, x, y, w, h)
#       elif len(faces) > 1:
#         detected_face(color_feed, x, y, w, h, random.sample(color, 3))
#     cv2.imshow('Face tracking', color_feed)
#     if cv2.waitKey(5) & 0xFF == int(27):
#       break
#   else:
#     print('No stream available.')

# stream.release()
# cv2.destroyAllWindows()
