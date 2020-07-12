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

from collections import OrderedDict
from itertools import chain, cycle

from .queue import AsyncQueue, Signal, StubQueue, VoidQueue, isStopSignal


class Pipeline(object):
    """Represents an Asynchronous pipeline"""

    def __init__(self):
        self.asyncSteps = OrderedDict()
        self.syncSteps = OrderedDict()
        self.extraAsyncSteps = OrderedDict()

        self._ownQueue = AsyncQueue()#VoidQueue()
        self._lastStep = None
        self._isLastStepAsync = False


    def addStep(self, name, step, maxQueueSize=100, isAsync=True):
        """Adds a step to the pipeline"""
        step.outputQueue = self._ownQueue

        # Check if last step was async
        if self._lastStep:
            if isAsync or self._isLastStepAsync:
                queue = AsyncQueue(maxsize=maxQueueSize)
                self._lastStep.isLastStep = False
            else:
                queue = StubQueue()

            self._lastStep.outputQueue = queue
            step.inputQueue = queue
        else:
            step.inputQueue = VoidQueue()#self._ownQueue

        if isAsync:
            self.asyncSteps[name] = step
        else:
            self.syncSteps[name] = step

        self._lastStep = step
        self._lastStep.isLastAsyncStep = isAsync
        self._isLastStepAsync = isAsync


    def run(self):
        """Starts all async and sync steps in the pipeline"""
        self._runAsyncSteps()
        self._runExtraAsyncSteps()
        self._runSyncSteps()


    def close(self):
        """Ends pipeline processing"""
        for step in self.asyncSteps.values():
            step.inputQueue.put(Signal.STOP_IMMEDIATELY)

        for step in self.extraAsyncSteps.values():
            step.inputQueue.put(Signal.STOP_IMMEDIATELY)

        for step in self.asyncSteps.values():
            step.join()

        for step in self.extraAsyncSteps.values():
            step.join()


    def printStatistics(self, timers):
        for name, timer in timers.items():
            print("{}: {}".format(name, timer))


    def addExtraAsyncStep(self, name, step):
        """Adds extra helper steps not intended to run in the same pipe"""
        self.extraAsyncSteps[name] = step


    def _runSyncSteps(self):
        """Run steps in main thread"""
        # Sleep main thread if there are no sync steps
        #TODO: Is this good practice?
        if not self.syncSteps:
            while True:
                signal = self._ownQueue.get()
                print("Pipeline get {}".format(signal))
                if signal is Signal.STOP:
                    break
            return

        for step in self.syncSteps.values():
            step.working = True
            step.setup()

        for step in cycle(self.syncSteps.values()):
            step.total_time.tick()
            item = step.inputQueue.get()

            if isStopSignal(item):
                step.inputQueue.close()
                step.outputQueue.put(item)
                break

            step.own_time.tick()
            output = step.process(item)
            step.own_time.tock()

            if isStopSignal(output):
                step.inputQueue.close()
                step.outputQueue.put(output)
                break

            step.total_time.tock()
            step.outputQueue.put(output)

        for step in self.syncSteps.values():
            step.working = False
            step.end()


    def _runAsyncSteps(self):
        for step in self.asyncSteps.values():
            if not step.working:
                print(step.__class__.__name__)
                step.start()


    def _runExtraAsyncSteps(self):
        for step in self.extraAsyncSteps.values():
            if not step.working:
                print(step.__class__.__name__)
                step.start()