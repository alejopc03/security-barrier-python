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
from multiprocessing import Process

from security_barrier.pipelining.queue import VoidQueue, Signal, isStopSignal
from security_barrier.pipelining.timer import TimerGroup, IncrementalTimer

class PipelineStep(object):
    """Class that specifies a step in a pipeline"""

    def __init__(self):
        self._process = None
        self.inputQueue = None
        self.outputQueue = VoidQueue()
        self.working = False
        self.isLastAsyncStep = False

        self.timers = TimerGroup()
        self.total_time = IncrementalTimer()
        self.own_time = IncrementalTimer()
        self._start_t = None


    def start(self):
        """Starts step work"""
        # TODO: Improve error handling
        if self.inputQueue is None or self.outputQueue is None:
            raise Exception("No input or output queue")

        if self._process is not None:
            raise Exception("Thread is already running")

        self._process = Process(target=self._run)
        self._process.start()
        self.working = True


    def process(self, item):
        raise NotImplementedError


    def setup(self):
        pass


    def end(self):
        pass

    def join(self):
        print("Finishing {}".format(self))

        self.inputQueue.put(Signal.STOP)
        self._process.join()
        self._process = None
        self.working = False


    def _run(self):
        """Step main function"""
        print("Starting {}".format(self.__class__.__name__))
        self._start_t = time.time()
        self.setup()

        self.total_time = IncrementalTimer()
        self.own_time = IncrementalTimer()

        # Step main loop
        while True:
            self.total_time.tick()
            item = self.inputQueue.get()
            print("{} get: {}".format(self.__class__.__name__, type(item)))

            if self._check_output(item):
                break

            self.own_time.tick()
            output = self.process(item)
            self.own_time.tock()

            if self._check_output(output):
                break

            self.total_time.tock()
            self.inputQueue.task_done()
            self.outputQueue.put(output)
            print("{} put: {}".format(self.__class__.__name__, type(output)))

        # End step work
        self.inputQueue.close()
        print("{} ended".format(self.__class__.__name__))
        self.working = False
        self.end()


    def _check_output(self, item):
        if isStopSignal(item):
            self.outputQueue.put(item)
            return True
        return False

