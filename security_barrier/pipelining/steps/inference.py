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
from openvino.inference_engine import IECore

from security_barrier.pipelining.models import IEModel, AsyncWrapper
from security_barrier.pipelining.queue import AsyncQueue
from security_barrier.pipelining.steps.base import PipelineStep


class InferenceStep(PipelineStep):
    """Step that handles inference operations.
    This is needed because Movidius NCS device only enables one process to handle it."""

    def __init__(self, device):
        super().__init__()
        self.ie = None
        self.device = device
        self.numRequests = 1
        self.inputQueue = AsyncQueue(maxsize=100)
        self.consumerSteps = {}


    def setup(self):
        # Initialize core inference engine
        print("Initializing IECore")
        self.ie = IECore()

        if self.device == 'MYRIAD':
            myriad_config = {"VPU_HW_STAGES_OPTIMIZATION": "YES"}
            self.ie.set_config(myriad_config, "MYRIAD")

        # Initialize models per consumer step
        for stepName, step in self.consumerSteps.items():
            print("Initializing model for: {}".format(stepName))
            self._initializeModel(stepName, step)


    def process(self, item):
        step = self.consumerSteps[item['stepName']]
        print("Consumer Step: {}".format(item['stepName']))
        result, frame, stepName = step['asyncModel'].infer(item['preprocessedFrame'], item['frame'])

        if result is not None and frame is not None and stepName is not None:
            stepToReturn = self.consumerSteps[stepName]
            payload = {'result': result, 'frame': frame}
            stepToReturn["recvInferencePipe"].put(payload)


    def addConsumerStep(self, stepName, modelPath, device):
        """Adds a pipeline step as a consumer for inference
            Returns: inputQueue, sendQueue"""
        self.consumerSteps[stepName] = {}
        self.consumerSteps[stepName]["modelPath"] = modelPath
        self.consumerSteps[stepName]["device"] = device
        self.consumerSteps[stepName]["recvInferencePipe"] = AsyncQueue(maxsize=10) #TODO: This will change to pipe class
        print("InferenceStep inputQueue: {}".format(str(self.inputQueue)))
        return self.inputQueue, self.consumerSteps[stepName]["recvInferencePipe"]


    def _initializeModel(self, stepName, step):
        xmlPath = step['modelPath'] + '.xml'
        binPath = step['modelPath'] + '.bin'

        model = IEModel(xmlPath, binPath, self.ie, self.device, self.numRequests)
        asyncModel = AsyncWrapper(model, self.numRequests, stepName)

        step['asyncModel'] = asyncModel
