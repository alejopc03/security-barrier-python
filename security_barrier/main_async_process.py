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
#!/usr/bin/env python
# Simplified port of OpenVINO security-barrier demo

import sys
import os
import logging as log
from argparse import ArgumentParser

from security_barrier.pipelining.steps.data import DataStep
from security_barrier.pipelining.steps.detection import VehicleDetectionStep
from security_barrier.pipelining.steps.attributes import VehicleAttributesStep
from security_barrier.pipelining.steps.render import RenderStep
from security_barrier.pipelining.steps.inference import InferenceStep
from security_barrier.pipelining.result_renderer import ResultRenderer
from security_barrier.pipelining.pipeline import Pipeline


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument("-i", "--input", help="Required. Path to video file or image. 'cam' for capturing video stream from camera", type=str, default='C:\\Users\\alpc_\\Projects\\security-barrier-python\\security_barrier\\camera_video.mp4')
    args.add_argument("-d", "--device", help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL, MYRIAD or HETERO: is "
                           "acceptable. The sample will look for a suitable plugin for device specified. Default "
                           "value is MYRIAD", default="MYRIAD", type=str)

    return parser


def main():
    log.basicConfig(format="[%(levelname)s] %(message)s", level=log.INFO, stream=sys.stdout)

    # -------------------- Parse args --------------------- #
    args = build_argparser().parse_args()
    device = args.device
       
    if args.input == 'cam':
        inputStream = 0
    else:
        inputStream = args.input
        assert os.path.isfile(args.input), "Specified input file doesn't exist"

    # -------------------- Create Pipeline --------------------- #
    log.info("Creating inference pipeline")
    inferencePipeline = Pipeline()

    inferenceStep = InferenceStep(device)
    dataStep = DataStep(inputStream)
    detectionStep = VehicleDetectionStep(inferenceStep)
    attributesStep = VehicleAttributesStep(inferenceStep)
    
    inferencePipeline.addStep("Data", dataStep, isAsync=True)
    inferencePipeline.addStep("Detection", detectionStep, isAsync=True)
    inferencePipeline.addStep("Attributes", attributesStep, isAsync=True)
    renderer = ResultRenderer(display_fps=True)
    inferencePipeline.addStep("Render", RenderStep(renderer.render_frame, fps=30), isAsync=True)
    inferencePipeline.addExtraAsyncStep("Inference", inferenceStep)

    # -------------------- Run Pipeline --------------------- #
    log.info("Running inference pipeline")
    inferencePipeline.run()
    inferencePipeline.close()
    # Statistics wont work until timers are multi-processed
    #inferencePipeline.printStatistics(renderer.timers)


if __name__ == '__main__':
    sys.exit(main() or 0)
