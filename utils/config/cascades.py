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
"""Config utility for initializing cascades."""

import os

# TODO(xames3): Remove suppressed pylint warnings.
# pyright: reportMissingImports=false
import cv2

from . import dev

cascades = os.path.join(dev.PROJECT_PATH, dev.CASCADES)
frontal_face_xml = os.path.join(cascades, 'haarcascade_frontalface_default.xml')
profile_face_xml = os.path.join(cascades, 'haarcascade_profileface.xml')
eyes_xml = os.path.join(cascades, 'haarcascade_eye.xml')

frontal_face = cv2.CascadeClassifier(frontal_face_xml)
profile_face = cv2.CascadeClassifier(profile_face_xml)
eyes = cv2.CascadeClassifier(eyes_xml)
