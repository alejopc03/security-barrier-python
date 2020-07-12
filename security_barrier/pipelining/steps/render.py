"""
 Alejandro Pereira (2019)

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
import time

import cv2

from security_barrier.pipelining.meters import MovingAverageMeter
from security_barrier.pipelining.queue import Signal
from security_barrier.pipelining.steps.base import PipelineStep


class RenderStep(PipelineStep):
    """Passes inference result to render function"""

    def __init__(self, render_fn, fps):
        super().__init__()
        cv2.startWindowThread()
        self.render = render_fn
        self.fps = fps
        self._frames_processed = 0
        self._t0 = None
        self._render_time = MovingAverageMeter(0.9)


    def process(self, item):
        if item is None:
            return
        self._sync_time()

        #frame, detection_results, attributes_results, timers = item
        render_start = time.time()
        status = self.render(*item, self._frames_processed)
        self._render_time.update(time.time() - render_start)

        self._frames_processed += 1
        if status is not None and status < 0:
            return Signal.STOP_IMMEDIATELY

        return status


    def end(self):
        cv2.destroyAllWindows()


    def _sync_time(self):
        now = time.time()
        if self._t0 is None:
            self._t0 = now
        expected_time = self._t0 + (self._frames_processed + 1) / self.fps
        if self._render_time.avg:
            expected_time -= self._render_time.avg
        if expected_time > now:
            time.sleep(expected_time - now)

