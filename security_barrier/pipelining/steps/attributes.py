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

import cv2
import numpy as np
from openvino.inference_engine import IECore

from security_barrier.pipelining.models import AsyncWrapper, IEModel
from security_barrier.pipelining.steps.base import PipelineStep
from security_barrier.pipelining.queue import AsyncQueue, Signal


class VehicleAttributesStep(PipelineStep):

    def __init__(self, inferenceStep):
        super().__init__()
        self.modelPath = r"C:\Users\alpc_\Projects\security-barrier-python\security_barrier\models\object_attributes\vehicle\FP16\vehicle-attributes-recognition-barrier-0039"
        self.labels = {
            "color": r"C:\Users\alpc_\Projects\security-barrier-python\security_barrier\labels\color.labels",
            "type": r"C:\Users\alpc_\Projects\security-barrier-python\security_barrier\labels\type.labels"
        }
        self.outputs = ['color', 'type']
        self.device = "MYRIAD"

        self.sendInferenceQueue, self.recvInferencePipe = inferenceStep.addConsumerStep(self.__class__.__name__, self.modelPath, self.device)
        print("VehicleAtt sendInferenceQueue: {}".format(str(self.sendInferenceQueue)))
        self.inputBatch = 1 #TODO: When is this filled correctly?
        self.inputWidth = 72
        self.inputHeight = 72
        self.inputChannels = 3

        self.labelmap = self._readLabels(self.labels)


    def process(self, item):
        vehicle_label = 'vehicle'
        frame, detection_results, timers = item
        result = None
        attributes_results = []
        if vehicle_label in detection_results.keys():
            for vehicle_obj in detection_results[vehicle_label]:
                preprocessed = self._preprocessInput(frame, vehicle_obj)
                result, frame = self._sendInferenceRequest(preprocessed, frame)
                attributes_results.append(self._postprocessOutput(result))

        #TODO: Frame post processing?
        timers[self.__class__.__name__] = self.own_time.last
        return frame, detection_results, attributes_results, timers


    def end(self):
        print("sending stop to inference step")
        self.sendInferenceQueue.put(Signal.STOP)


    def _preprocessInput(self, frame, detectedObject):
        xmin = detectedObject[0]
        ymin = detectedObject[1]
        xmax = detectedObject[2]
        ymax = detectedObject[3]
        vehicle_frame = frame[ymin:ymax, xmin:xmax, :]
        vehicle_frame = cv2.resize(vehicle_frame, (self.inputWidth, self.inputHeight))
        vehicle_frame = vehicle_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        vehicle_frame = vehicle_frame.reshape((self.inputBatch, self.inputChannels, self.inputHeight, self.inputWidth))
        return vehicle_frame


    def _postprocessOutput(self, output):
        """Decodes attributes detected for the object"""
        result = {}
        # Get the highest probabilty label
        color_id = np.argmax(output['color'][0, :, 0, 0])
        type_id = np.argmax(output['type'][0, :, 0, 0])

        result['color'] = self.labelmap['color'][color_id]
        result['type'] = self.labelmap['type'][type_id]

        return result


    def _sendInferenceRequest(self, preprocessedFrame, frame):
        payload = {
            "stepName": self.__class__.__name__,
            "preprocessedFrame": preprocessedFrame,
            "frame": frame
        }
        self.sendInferenceQueue.put(payload)

        # Wait for inference response
        result, frame = None, None
        while True:
            response = self.recvInferencePipe.get()
            if response:
                result = response['result']
                frame = response['frame']
                break


        return result, frame


    def _readLabels(self, outputs):
        labelMap = {}

        for output, labelsPath in outputs.items():
            with open(labelsPath, 'r') as f:
                labelMap[output] = [x.strip() for x in f]

        return labelMap
