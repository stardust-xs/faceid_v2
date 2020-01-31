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
"""The `faceid_v2.run` module."""

import os
import time

import cv2
import imutils
import numpy as np
from imutils.video import VideoStream

# TODO(xames3): Remove suppressed pylint warnings.
# pyright: reportMissingImports=false
from faceid_v2.config import dev
from faceid_v2.utils.hud import detect_face, detect_motion

# Loading necessary models.
models_path = os.path.join(dev.PROJECT_PATH, 'models')
face_prototext = os.path.join(models_path, dev.FACE_PROTOTEXT)
face_caffemodel = os.path.join(models_path, dev.FACE_CAFFEMODEL)
face_net = cv2.dnn.readNetFromCaffe(face_prototext, face_caffemodel)
# Setting constants for the service.
blur = False
base_frame = None
# Streaming engine
stream = VideoStream(src=0).start()
time.sleep(2.0)

try:
  while True:
    frame = stream.read()
    frame = imutils.resize(frame, width=400)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)
    frame_height, frame_width = frame.shape[:2]

    if base_frame is None:
      base_frame = gray_frame
      continue

    frame_delta = cv2.absdiff(base_frame, gray_frame)
    motion_threshold = cv2.threshold(frame_delta, 30, 255, cv2.THRESH_BINARY)[1]
    motion_threshold = cv2.dilate(motion_threshold, None, iterations=2)
    motion_contours = cv2.findContours(motion_threshold.copy(),
                                      cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)
    detected_motion = imutils.grab_contours(motion_contours)
    for idx in detected_motion:
      if cv2.contourArea(idx) < 1000:
        continue
      (motion_top_x, motion_top_y,
      motion_btm_x, motion_btm_y) = cv2.boundingRect(idx)
      detect_motion(frame, motion_top_x, motion_top_y,
                    motion_btm_x, motion_btm_y)

    face_blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                      (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(face_blob)
    detected_face = face_net.forward()
    for idx in range(detected_face.shape[2]):
      confidence = detected_face[0, 0, idx, 2]
      if confidence < dev.FACE_DETECTING_CONFIDENCE:
        continue
      face_coords = detected_face[0, 0, idx, 3:7] * np.array([frame_width,
                                                              frame_height,
                                                              frame_width,
                                                              frame_height])
      face_top_x, face_top_y, face_btm_x, face_btm_y = face_coords.astype('int')
      if blur:
        frame[face_top_y:face_btm_y, face_top_x:face_btm_x] = cv2.blur(
            frame[face_top_y:face_btm_y, face_top_x:face_btm_x], (25, 25))
      detect_face(frame, (face_top_x, face_top_y), (face_btm_x, face_btm_y))

    # cv2.imshow('Motion threshold', motion_threshold)
    # cv2.imshow('Frame delta', frame_delta)
    cv2.imshow('Live feed', frame)

    if cv2.waitKey(5) & 0xFF == ord('b'):
      blur = not blur

    if cv2.waitKey(5) & 0xFF == int(27):
      print('Terminating...')
      break

except cv2.error as error:
  print('Something failed with CV2.')
  raise error
finally:
  cv2.destroyAllWindows()
  stream.stop()
  print('Stream terminated.')
