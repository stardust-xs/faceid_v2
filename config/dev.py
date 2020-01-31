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
"""Config utility to define variables for development."""

import os

# Settings for the project
PROJECT_NAME = 'faceid_v2'
PROJECT_LINK = 'https://github.com/xames3/faceid_v2/'
PROJECT_LICENSE = 'Apache 2.0'
PROJECT_PATH = os.path.dirname(os.path.dirname(__file__))

# This project adheres to Semantic Versioning Specification (SemVer)
# starting with version 0.0.1.
# You can read about it here: https://semver.org/spec/v2.0.0.html
PROJECT_VERSION = '1.0'

# Author details.
AUTHOR = 'XAMES3'
AUTHOR_EMAIL = 'xames3.developer@gmail.com'

# Local time zone details
# You can find all the choices here:
# https://en.wikipedia.org/wiki/List_of_tz_zones_by_name
TIME_ZONE = 'Asia/Kolkata'

# Language used by the project
# You can find all the choices here:
# http://www.i18nguy.com/unicode/language-identifiers.html
LANG_CODE = 'en-gb'

# Default encoding used for all read-write objects.
DEF_CHARSET = 'utf-8'

# OpenCV models
FACE_PROTOTEXT = 'deploy.prototxt.txt'
FACE_CAFFEMODEL = 'res10_300x300_ssd_iter_140000.caffemodel'

# Default urls
# This url is used for checking if the internet connection exists.
PING_URL = 'www.google.com'
PING_PORT = 80

# Confidence scores
FACE_DETECTING_CONFIDENCE = 0.7
