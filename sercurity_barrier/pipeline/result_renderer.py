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

from collections import Counter, defaultdict, deque
from functools import partial
from itertools import islice

from .meters import WindowAverageMeter

import cv2
import numpy as np

FONT_COLOR = (255, 255, 255)
FONT_STYLE = cv2.FONT_HERSHEY_DUPLEX
FONT_SIZE = 1
TEXT_VERTICAL_INTERVAL = 45
TEXT_LEFT_MARGIN = 15


class ResultRenderer(object):
    def __init__(self, display_fps=False, display_confidence=False, output_height=720):
        #self.display_confidence = display_confidence
        self.display_fps = display_fps
        self.output_height = output_height
        self.meters = defaultdict(partial(WindowAverageMeter, 16))
        print("To close the application, press 'CTRL+C' here or switch to the output window and press any key")


    def update_timers(self, timers):
        self.meters['VehicleDetectionStep'].update(timers['VehicleDetectionStep'])
        self.meters['VehicleAttributesStep'].update(timers['VehicleAttributesStep'])
        return self.meters['VehicleDetectionStep'].avg + self.meters['VehicleAttributesStep'].avg


    def render_frame(self, frame, detection_results, attributes_results, timers, frame_ind):
        inference_time = self.update_timers(timers)
        # Render detected vehicles
        color = (0, 255, 0) # Green
        if detection_results is not None and 'vehicle' in detection_results.keys():
            for i in range(len(detection_results['vehicle'])):
                vehicle = detection_results['vehicle'][i]
                attributes = attributes_results[i]
                self.render_vehicle(vehicle, attributes, color, frame)

        # Render detected plates
        if detection_results is not None and 'license_plate' in detection_results.keys():
            for plate in detection_results['license_plate']:
                self.render_plate(plate, color, frame)

        # Fill text area
        #fill_area(frame, (0, 70), (700, 0), alpha=0.6, color=(0, 0, 0))

        if self.display_fps:
            fps = 1000 / (inference_time + 1e-6)
            text_loc = (TEXT_LEFT_MARGIN, TEXT_VERTICAL_INTERVAL)
            cv2.putText(frame, "Inference time: {:.2f}ms ({:.2f} FPS)".format(inference_time, fps), text_loc, FONT_STYLE, FONT_SIZE, FONT_COLOR)

        cv2.imshow("Vehicle Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            return -1


    def render_vehicle(self, vehicle, attributes, color, frame):
        xmin, ymin, xmax, ymax, prob = vehicle[0], vehicle[1], vehicle[2], vehicle[3], vehicle[4]
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(frame, 'Vehicle ' + str(round(prob * 100, 1)) + ' %', (xmin, ymin - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)
        cv2.putText(frame, 'Type: ' + attributes['type'] + ' ', (xmin, ymin - 40), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)
        cv2.putText(frame, 'Color: ' + attributes['color'], (xmin, ymin - 70), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)


    def render_plate(self, plate, color, frame):
        xmin, ymin, xmax, ymax, prob = plate[0], plate[1], plate[2], plate[3], plate[4]
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(frame, 'Plate ' + str(round(prob * 100, 1)) + ' %', (xmin, ymin - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)


def fill_area(image, bottom_left, top_right, color=(0, 0, 0), alpha=1.):
    """Fills area with the specified color"""
    xmin, ymax = bottom_left
    xmax, ymin = top_right

    image[ymin:ymax, xmin:xmax, :] = image[ymin:ymax, xmin:xmax, :] * (1 - alpha) + np.asarray(color) * alpha
    return image
