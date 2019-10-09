"""
 Copyright (c) 2019 Intel Corporation
 Modified by: Alejandro Pereira (2019)

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import cv2

from security_barrier.pipelining.queue import Signal
from security_barrier.pipelining.steps.base import PipelineStep

#TODO: Add desired framerate to read
class DataStep(PipelineStep):
    """Step that processes incoming video data"""

    def __init__(self, videoFile):
        super().__init__()
        self.videoFile = videoFile
        self.cap = None


    def setup(self):
        self._openVideo()


    def process(self, item):
        # If video is not available stop
        if not self.cap.isOpened() and not self._openVideo():
            return Signal.STOP

        status, frame = self.cap.read()

        # If video ended or failed to get next frame, stop
        if not status:
            return Signal.STOP

        return frame


    def end(self):
        self.cap.release()


    def _openVideo(self):
        self.cap = cv2.VideoCapture(self.videoFile)
        return self.cap.isOpened()