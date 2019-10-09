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

from security_barrier.pipelining.steps.base import PipelineStep


class VehicleDetectionStep(PipelineStep):

    def __init__(self, inferenceStep):
        super().__init__()
        self.modelPath = "C:\\Users\\alpc_\\Projects\\security-barrier-python\\security_barrier\\models\\object_detection\\barrier\\FP16\\vehicle-license-plate-detection-barrier-0106"
        self.device = "MYRIAD"
        self.outputs = ['DetectionOutput_']
        self.labels = {
            "DetectionOutput_": r"C:\Users\alpc_\Projects\security-barrier-python\security_barrier\labels\detection.labels"
        }

        self.sendInferenceQueue, self.recvInferencePipe = inferenceStep.addConsumerStep(self.__class__.__name__, self.modelPath, self.device)
        print("VehicleAtt sendInferenceQueue: {}".format(str(self.sendInferenceQueue)))
        self.inputBatch = 1 #TODO: When is this filled correctly?
        self.inputWidth = 300
        self.inputHeight = 300
        self.inputChannels = 3
        self.probThreshold = 0.5

        self.labelmap = self._readLabels(self.labels)


    def process(self, frame):
        preprocessed = self._preprocessInput(frame)
        result, frame = self._sendInferenceRequest(preprocessed, frame)

        output = result["DetectionOutput_"]
        detectionResults = self._postprocessOutput(output, frame)
        timers = {self.__class__.__name__: self.own_time.last}
        return frame, detectionResults, timers


    def _preprocessInput(self, frame):
        """Resizes frame to match network input size"""
        in_frame = cv2.resize(frame, (self.inputWidth, self.inputHeight))
        in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        in_frame = in_frame.reshape((self.inputBatch, self.inputChannels, self.inputHeight, self.inputWidth))
        return in_frame


    def _postprocessOutput(self, output, frame):
        """Filters only objects that are above the threshold
            Returns a list of objects per label"""
        filtered_output = {}
        vehicle_label_map = self.labelmap[self.outputs[0]]
        for label in vehicle_label_map:
            filtered_output[label] = []

        initial_h, initial_w, _ = frame.shape

        for obj in output[0][0]: #TODO: can we optimize this loop?
            # Draw only objects when probability more than specified threshold
            if obj[2] > self.probThreshold:
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


    def _sendInferenceRequest(self, preprocessedFrame, frame):
        payload = {
            "stepName": self.__class__.__name__,
            "preprocessedFrame": preprocessedFrame,
            "frame": frame
        }
        print("{} sendInferenceQueue".format(self.__class__.__name__))
        self.sendInferenceQueue.put(payload)

        # Wait for inference response
        result, frame = None, None
        while True:
            response = self.recvInferencePipe.get()
            result = response['result']
            frame = response['frame']
            if response:
                break

        return result, frame


    def _readLabels(self, outputs):
        labelMap = {}

        for output, labelsPath in outputs.items():
            with open(labelsPath, 'r') as f:
                labelMap[output] = [x.strip() for x in f]

        return labelMap
