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
from openvino.inference_engine import IENetwork, IECore, IEPlugin


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

    attributes_model_xml = args.model2.format(root_dir, precision)
    attributes_model_bin = os.path.splitext(attributes_model_xml)[0] + ".bin"
    att_color_labels = os.path.join(root_dir, 'labels', 'color.labels')
    att_types_labels = os.path.join(root_dir, 'labels', 'type.labels')

    ocr_model_xml = args.model3.format(root_dir, precision)
    ocr_model_bin = os.path.splitext(ocr_model_xml)[0] + ".bin"

    # -------------------- Load Inference Engine --------------------- #
    log.info("Loading Inference Engine...")
    plugin = IEPlugin(device=device)
    ie = IECore()
    
    # Load IR Models
    log.info("Loading detection model:\n\t{}\n\t{}".format(detection_model_xml, detection_model_bin))
    detection_net = IENetwork(model=detection_model_xml, weights=detection_model_bin)

    log.info("Loading attributes model:\n\t{}\n\t{}".format(attributes_model_xml, attributes_model_bin))
    attributes_net = IENetwork(model=attributes_model_xml, weights=attributes_model_bin)

    log.info("Loading OCR model:\n\t{}\n\t{}".format(ocr_model_xml, ocr_model_bin))
    ocr_net = IENetwork(model=ocr_model_xml, weights=ocr_model_bin)

    
    log.info("Preparing detection input/output blobs")
    # Detection input/output blobs
    det_input_blob = None
    det_img_info_input_blob = None
    for blob_name in detection_net.inputs:
        if len(detection_net.inputs[blob_name].shape) == 4:
            det_input_blob = blob_name
        elif len(detection_net.inputs[blob_name].shape) == 2:
            det_img_info_input_blob = blob_name
        else:
            raise RuntimeError("Unsupported {}D input layer '{}'. Only 2D and 4D input layers are supported".format(len(detection_net.inputs[blob_name].shape), blob_name))

    detection_out_blob = next(iter(detection_net.outputs))

    log.info("Preparing attributes input/output blobs")
    # Attributes input/output blobs
    att_input_blob = None
    att_img_info_input_blob = None
    for blob_name in attributes_net.inputs:
        if len(attributes_net.inputs[blob_name].shape) == 4:
            att_input_blob = blob_name
        elif len(attributes_net.inputs[blob_name].shape) == 2:
            att_img_info_input_blob = blob_name
        else:
            raise RuntimeError("Unsupported {}D input layer '{}'. Only 2D and 4D input layers are supported".format(len(attributes_net.inputs[blob_name].shape), blob_name))
    
    attributes_out_blobs = []
    for blob_name in attributes_net.outputs:
        attributes_out_blobs.append(blob_name)

    #TODO: Add OCR input/output blobs
    #ocr_out_blob = next(iter(ocr_net.outputs))
    
    # Read and pre-process input
    log.info("Pre-processing inputs")
    det_feed_dict = {}
    att_feed_dict = {}
    det_n, det_c, det_h, det_w = detection_net.inputs[det_input_blob].shape # Detection network input shape
    att_n, att_c, att_h, att_w = attributes_net.inputs[att_input_blob].shape # Attributes network input shape

    if det_img_info_input_blob:
        det_feed_dict[det_img_info_input_blob] = [det_h, det_w, 1]

    if att_img_info_input_blob:
        att_feed_dict[att_img_info_input_blob] = [att_h, att_w, 1]
    
    if args.input == 'cam':
        input_stream = 0
    else:
        input_stream = args.input
        assert os.path.isfile(args.input), "Specified input file doesn't exist"
    
    with open(detection_labels, 'r') as f:
        det_labels_map = [x.strip() for x in f]

    with open(att_color_labels, 'r') as f:
        color_labels_map = [x.strip() for x in f]

    with open(att_types_labels, 'r') as f:
        types_labels_map = [x.strip() for x in f]
    
    # Loading model to the plugin
    log.info("Loading models to the plugin")
    detection_exec_net = plugin.load(network=detection_net, num_requests=2)
    attributes_exec_net = plugin.load(network=attributes_net, num_requests=2)
    #ocr_exec_net = plugin.load(network=ocr_net)

    # -------------------- Start Inference --------------------- #
    log.info("Starting inference in async mode...")
    # Using OpenCV for video stream processing
    cap = cv2.VideoCapture(input_stream)
    cur_request_id = 0
    next_request_id = 1
    is_async_mode = False
    render_time = 0
    if is_async_mode:
        ret, frame = cap.read()

    print("To close the application, press 'CTRL+C' here or switch to the output window and press ESC key")
    print("To switch between sync/async modes, press TAB key in the output window")
   
    while cap.isOpened():
            if is_async_mode:
                ret, next_frame = cap.read()
            else:
                ret, frame = cap.read()
            if not ret:
                break
            initial_w = cap.get(3)
            initial_h = cap.get(4)
            # Main sync point:
            # in the truly Async mode we start the NEXT infer request, while waiting for the CURRENT to complete
            # in the regular mode we start the CURRENT request and immediately wait for it's completion
            inf_start = time.time()
            if is_async_mode:
                in_frame = cv2.resize(next_frame, (det_w, det_h))
                in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
                in_frame = in_frame.reshape((det_n, det_c, det_h, det_w))
                det_feed_dict[det_input_blob] = in_frame
                detection_exec_net.start_async(request_id=next_request_id, inputs=det_feed_dict)
            else:
                in_frame = cv2.resize(frame, (det_w, det_h))
                in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
                in_frame = in_frame.reshape((det_n, det_c, det_h, det_w))
                det_feed_dict[det_input_blob] = in_frame
                detection_exec_net.start_async(request_id=cur_request_id, inputs=det_feed_dict)

            if detection_exec_net.requests[cur_request_id].wait(-1) == 0:
                inf_end = time.time()
                det_time = inf_end - inf_start

                # Parse detection results of the current request
                res = detection_exec_net.requests[cur_request_id].outputs[detection_out_blob]
                for obj in res[0][0]:
                    # Draw only objects when probability more than specified threshold
                    if obj[2] > args.prob_threshold:
                        xmin = int(obj[3] * initial_w)
                        ymin = int(obj[4] * initial_h)
                        xmax = int(obj[5] * initial_w)
                        ymax = int(obj[6] * initial_h)
                        class_id = int(obj[1])
                        # Draw box and label\class_id
                        color = (min(class_id * 12.5, 255), min(class_id * 7, 255), min(class_id * 5, 255))
                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                        det_label = det_labels_map[class_id] if det_labels_map else str(class_id)
                        cv2.putText(frame, det_label + ' ' + str(round(obj[2] * 100, 1)) + ' %', (xmin, ymin - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)

                        # ----------- Infer vehicle attributes ---------------
                        if det_label == 'vehicle':
                            # Get vehicle frame and transform it to OpenVINO IR format
                            vehicle_frame = frame[ymin:ymax,xmin:xmax,:]
                            vehicle_frame = cv2.resize(vehicle_frame, (att_w, att_h))
                            vehicle_frame = vehicle_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
                            vehicle_frame = vehicle_frame.reshape((att_n, att_c, att_h, att_w))
                            att_feed_dict[att_input_blob] = vehicle_frame
                            # Infer attributes
                            attributes_exec_net.start_async(request_id=cur_request_id, inputs=att_feed_dict)
                            # Get labels from detected color and type
                            if attributes_exec_net.requests[cur_request_id].wait(-1) == 0:
                                color_res = attributes_exec_net.requests[cur_request_id].outputs['color']
                                type_res = attributes_exec_net.requests[cur_request_id].outputs['type']

                                color_class_id = np.argmax(color_res[0,:,0,0])
                                type_class_id = np.argmax(type_res[0,:,0,0])

                                color_label = color_labels_map[color_class_id] if color_labels_map else str(color_class_id)
                                type_label = types_labels_map[type_class_id] if types_labels_map else str(type_class_id)
                                cv2.putText(frame, 'Type: ' + type_label + ' ', (xmin, ymin - 40), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)
                                cv2.putText(frame, 'Color: ' + color_label + ' ', (xmin, ymin - 70), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)


                # Draw performance stats
                inf_time_message = "Inference time: N\A for async mode" if is_async_mode else \
                    "Inference time: {:.3f} ms".format(det_time * 1000)
                render_time_message = "OpenCV rendering time: {:.3f} ms".format(render_time * 1000)
                async_mode_message = "Async mode is on. Processing request {}".format(cur_request_id) if is_async_mode else \
                    "Async mode is off. Processing request {}".format(cur_request_id)

                cv2.putText(frame, inf_time_message, (15, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
                cv2.putText(frame, render_time_message, (15, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (10, 10, 200), 1)
                cv2.putText(frame, async_mode_message, (10, int(initial_h - 20)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (10, 10, 200), 1)

            render_start = time.time()
            cv2.imshow("Detection Results", frame)
            render_end = time.time()
            render_time = render_end - render_start

            if is_async_mode:
                cur_request_id, next_request_id = next_request_id, cur_request_id
                frame = next_frame

            key = cv2.waitKey(1)
            if key == 27:
                break
            if (9 == key):
                is_async_mode = not is_async_mode
                log.info("Switched to {} mode".format("async" if is_async_mode else "sync"))

    cv2.destroyAllWindows()


if __name__ == '__main__':
    sys.exit(main() or 0)
