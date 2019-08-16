#!/usr/bin/env python

# Simplified port of OpenVINO security-barrier demo

from __future__ import print_function
import sys
import os
from argparse import ArgumentParser, SUPPRESS
import cv2
import numpy as np
import logging as log
import time

from openvino.inference_engine import IECore
from pipeline.models import IEModel
from pipeline.result_renderer import ResultRenderer
from pipeline.queue import Signal
from pipeline.pipeline import AsyncPipeline

import pipeline.steps


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument("-m1", "--model1", help="Required. Path to object detection .xml trained model.", type=str, default='{0}/models/object_detection/barrier/{1}/vehicle-license-plate-detection-barrier-0106.xml')
    args.add_argument("-m2", "--model2", help="Required. Path to vehicle attributes .xml trained model.", type=str, default='{0}/models/object_attributes/vehicle/{1}/vehicle-attributes-recognition-barrier-0039.xml')
    args.add_argument("-m3", "--model3", help="Required. Path to OCR .xml trained model.", type=str, default='{0}/models/optical_character_recognition/license_plate/{1}/license-plate-recognition-barrier-0001.xml')
    args.add_argument("-i", "--input", help="Required. Path to video file or image. 'cam' for capturing video stream from camera", type=str, default='/home/pereiraa/projects/security-barrier-python/sercurity_barrier/test_video.mp4')
    args.add_argument("-d", "--device", help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL, MYRIAD or HETERO: is "
                           "acceptable. The sample will look for a suitable plugin for device specified. Default "
                           "value is CPU", default="MYRIAD", type=str)
    args.add_argument("-pt", "--prob_threshold", help="Optional. Probability threshold for detections filtering", default=0.5, type=float)

    return parser


def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)

    # -------------------- Parse args --------------------- #
    args = build_argparser().parse_args()
    root_dir = os.path.dirname(os.path.realpath(__file__))
    device = args.device
    precision = 'FP16' if device == 'MYRIAD' else 'FP32'

    # Get models absolute paths
    detection_model_xml = args.model1.format(root_dir, precision)
    detection_model_bin = os.path.splitext(detection_model_xml)[0] + ".bin"
    detection_labels = os.path.join(root_dir, 'labels', 'detection.labels')
    det_labels_dict = {}
        
    attributes_model_xml = args.model2.format(root_dir, precision)
    attributes_model_bin = os.path.splitext(attributes_model_xml)[0] + ".bin"
    att_color_labels = os.path.join(root_dir, 'labels', 'color.labels')
    att_types_labels = os.path.join(root_dir, 'labels', 'type.labels')
    att_labels_dict = {}

    with open(detection_labels, 'r') as f:
        det_labels_dict['DetectionOutput_'] = [x.strip() for x in f]

    with open(att_color_labels, 'r') as f:
        att_labels_dict['color'] = [x.strip() for x in f]

    with open(att_types_labels, 'r') as f:
        att_labels_dict['type'] = [x.strip() for x in f]

    # -------------------- Load Inference Engine --------------------- #
    log.info("Loading Inference Engine...")
    ie = IECore()

    if device == 'MYRIAD':
        myriad_config = {"VPU_HW_STAGES_OPTIMIZATION": "YES"}
        ie.set_config(myriad_config, "MYRIAD")

    cpu_extension = None
    if cpu_extension and device == 'CPU':
        ie.add_extension(args.cpu_extension, "CPU")
    
    # Load IR Models
    log.info("Loading detection model:\n\t{}\n\t{}".format(detection_model_xml, detection_model_bin))
    detection_model = IEModel(detection_model_xml, detection_model_bin, ie, device, num_requests=2, labels=det_labels_dict)

    log.info("Loading attributes model:\n\t{}\n\t{}".format(attributes_model_xml, attributes_model_bin))
    attributes_model = IEModel(attributes_model_xml, attributes_model_bin, ie, device, num_requests=2, labels=att_labels_dict)

       
    if args.input == 'cam':
        input_stream = 0
    else:
        input_stream = args.input
        assert os.path.isfile(args.input), "Specified input file doesn't exist"

    # -------------------- Create Pipeline --------------------- #
    log.info("Creating inference pipeline...")
    inferencePipeline = AsyncPipeline()
    inferencePipeline.add_step("Data", pipeline.steps.DataStep(input_stream), parallel=False)
    inferencePipeline.add_step("Detection", pipeline.steps.VehicleDetectionStep(detection_model), parallel=False)
    inferencePipeline.add_step("Attributes", pipeline.steps.VehicleAttributesStep(attributes_model), parallel=False)
    renderer = ResultRenderer()
    inferencePipeline.add_step("Render", pipeline.steps.RenderStep(renderer.render_frame, fps=30), parallel=False)

    # -------------------- Run Pipeline --------------------- #
    log.info("Running inference pipeline...")
    inferencePipeline.run()
    inferencePipeline.close()
    inferencePipeline.print_statistics()

    #region Test pipeline single-threaded
    # -------------------- Create Pipeline --------------------- #
    # log.info("Creating inference pipeline...")
    # dataStep = pipeline.steps.DataStep(input_stream)
    # detectionStep = pipeline.steps.VehicleDetectionStep(detection_model)
    # attributesStep = pipeline.steps.VehicleAttributesStep(attributes_model)
    # renderer = ResultRenderer()
    # renderStep = pipeline.steps.RenderStep(renderer.render_frame, fps=30)

    # # -------------------- Initialize Pipeline --------------------- #
    # dataStep.setup()
    # detectionStep.setup()
    # attributesStep.setup()
    # renderStep.setup()

    # log.info("Starting inference pipeline...")

    #complete = False
    # while not complete:
    #     output = dataStep.process(None)
    #     if output is Signal.STOP:
    #         complete = True
    #     else:
    #         output = detectionStep.process(output)
    #         output = attributesStep.process(output)
    #         status = renderStep.process(output)

    # cv2.destroyAllWindows()
    #endregion

if __name__ == '__main__':
    sys.exit(main() or 0)
