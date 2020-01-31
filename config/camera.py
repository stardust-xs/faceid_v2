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
"""Config utility for defining the IP Camera parameters."""

DEF_PROTOCOL = 'rtsp'
DEF_USER = 'xxx'
DEF_PASSWORD = 'xxx'
DEF_STREAM_ADDR = 'xxx'
DEF_PORT = 554
DEF_STREAM_ID = 1

LIVE = (f'{DEF_PROTOCOL}://{DEF_USER}:{DEF_PASSWORD}@{DEF_STREAM_ADDR}:'
        f'{DEF_PORT}//Streaming/Channels/{DEF_STREAM_ID}')
