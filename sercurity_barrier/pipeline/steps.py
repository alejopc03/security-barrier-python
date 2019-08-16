"""
 Copyright (c) 2019 Intel Corporation

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
from collections import deque
from itertools import cycle

import cv2
import numpy as np

from .meters import MovingAverageMeter
from .models import AsyncWrapper, preprocess_frame
from .pipeline import AsyncPipeline, PipelineStep
from .queue import Signal


def run_pipeline(video, encoder, decoder, render_fn, fps=30):
    pipeline = AsyncPipeline()
    pipeline.add_step("Data", DataStep(video), parallel=False)
    pipeline.add_step("Encoder", EncoderStep(encoder), parallel=False)
    pipeline.add_step("Decoder", DecoderStep(decoder), parallel=False)
    pipeline.add_step("Render", RenderStep(render_fn, fps=fps), parallel=True)

    pipeline.run()
    pipeline.close()
    pipeline.print_statistics()


class DataStep(PipelineStep):

    def __init__(self, video_file, loop=True):
        super().__init__()
        self.video_file = video_file
        self.cap = None


    def setup(self):
        self._open_video()

    def process(self, item):
        if not self.cap.isOpened() and not self._open_video():
            return Signal.STOP
        status, frame = self.cap.read()
        if not status:
            return Signal.STOP

        return frame

    def end(self):
        self.cap.release()

    def _open_video(self):
        result = False
        self.cap = cv2.VideoCapture(self.video_file)
        
        if self.cap.isOpened():
            result = True

        return result


class RenderStep(PipelineStep):
    """Passes inference result to render function"""

    def __init__(self, render_fn, fps):
        super().__init__()
        self.render = render_fn
        self.fps = fps
        self._frames_processed = 0
        self._t0 = None
        self._render_time = MovingAverageMeter(0.9)

    def process(self, item):
        if item is None:
            return
        self._sync_time()
        
        frame, detection_results, attributes_results, timers = item
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


class VehicleDetectionStep(PipelineStep):

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.async_model = AsyncWrapper(self.model, self.model.num_requests)

    def process(self, frame):
        preprocessed = self._preprocess_input(frame)
        result, frame = self.async_model.infer(preprocessed, frame)

        if result is None:
            return None
        output = result[self.model.output_names[0]]
        detection_results = self._postprocess_output(output, frame)
        timers = {'VehicleDetectionStep': self.own_time.last}
        return frame, detection_results, timers


    def _preprocess_input(self, frame):
        """Resizes frame to match network input size"""
        in_frame = cv2.resize(frame, (self.model.input_w, self.model.input_h)) #TODO: add model inputs (w,h) to IEModel class
        in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        in_frame = in_frame.reshape((self.model.input_n, self.model.input_c, self.model.input_h, self.model.input_w))
        return in_frame


    def _postprocess_output(self, output, frame):
        """Filters only objects that are above the threshold
            Returns a list of objects per label"""
        filtered_output = {}
        vehicle_label_map = self.model.labels[self.model.output_names[0]]
        for label in vehicle_label_map:
            filtered_output[label] = []

        initial_h, initial_w, _ = frame.shape

        for obj in output[0][0]: #TODO: can we optimize this loop?
            # Draw only objects when probability more than specified threshold
            if obj[2] > self.model.prob_threshold:
                xmin = int(obj[3] * initial_w)
                ymin = int(obj[4] * initial_h)
                xmax = int(obj[5] * initial_w)
                ymax = int(obj[6] * initial_h)
                prob = obj[2]
                class_id = int(obj[1])
                det_label = vehicle_label_map[class_id]
                bbox = [xmin, ymin, xmax, ymax, prob]
                filtered_output[det_label].append(bbox)

        return filtered_output
        

class VehicleAttributesStep(PipelineStep):

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.async_model = AsyncWrapper(self.model, self.model.num_requests)

    def process(self, item):
        vehicle_label = 'vehicle'
        frame, detection_results, timers = item
        result = None
        attributes_results = []
        if vehicle_label in detection_results.keys():
            for vehicle_obj in detection_results[vehicle_label]:
                preprocessed = self._preprocess_input(frame, vehicle_obj)
                result, frame = self.async_model.infer(preprocessed, frame)
                attributes_results.append(self._postprocess_output(result))

        
        #TODO: Frame post processing?
        timers['VehicleAttributesStep'] = self.own_time.last
        return frame, detection_results, attributes_results, timers


    def _preprocess_input(self, frame, object):
        xmin = object[0]
        ymin = object[1]
        xmax = object[2]
        ymax = object[3]
        vehicle_frame = frame[ymin:ymax,xmin:xmax,:]
        vehicle_frame = cv2.resize(vehicle_frame, (self.model.input_w, self.model.input_h))
        vehicle_frame = vehicle_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        vehicle_frame = vehicle_frame.reshape((self.model.input_n, self.model.input_c, self.model.input_h, self.model.input_w))
        return vehicle_frame


    def _postprocess_output(self, output):
        """Decodes attributes detected for the object"""
        result = {}
        # Get the highest probabilty label
        color_id = np.argmax(output['color'][0,:,0,0])
        type_id = np.argmax(output['type'][0,:,0,0])

        result['color'] = self.model.labels['color'][color_id]
        result['type'] = self.model.labels['type'][type_id]

        return result