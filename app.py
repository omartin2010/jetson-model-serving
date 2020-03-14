from flask import Flask
from flask import request
import base64
import logging
from PIL import Image
import io
import os
import traceback
import numpy as np
import json
import time
import sys
import signal
import tensorflow as tf
from utils import label_map_util
from logger import RoboLogger
from constant import LOGGER_SCORING_STARTUP, LOGGER_SCORING_SIGTERM, \
    LOGGER_SCORING_GRACEFUL_SHUTDOWN, LOGGER_SCORING_LOAD_MODELS, \
    LOGGER_SCORING_MODEL_RUN
import argparse
app = Flask(__name__)

log = RoboLogger.getLogger()
log.warning(LOGGER_SCORING_STARTUP,
            msg="Initial imports are completed.")
category_index = None
categories = None
detection_graph = None
tf_sess = None


@app.route('/score/model/<model_name>', methods=['POST'])
def score_model_object_detection(model_name):
    global detection_graph, tf_sess
    try:
        if model_name == 'object_detector':
            if request.method == 'POST':
                with detection_graph.as_default():
                    image_tensor, boxes, scores, classes, num_detections = \
                        get_tensors()
                    # Read frame from camera
                    base64_encoded_image_dict = request.get_json()
                    base64_decoded_image = base64.b64decode(
                        base64_encoded_image_dict['image'])
                    # deserialize that from the post
                    rgb_image_color_np = np.array(Image.open(
                        io.BytesIO(base64_decoded_image)).convert("RGB"))
                    # Expand dimensions to [1, None, None, 3] for TF model
                    image_np_expanded = \
                        np.expand_dims(rgb_image_color_np, axis=0)
                    # Actual detection
                    start_time = time.time()
                    log.debug(LOGGER_SCORING_MODEL_RUN,
                              msg=f'Starting inferencing...')
                    (boxes, scores, classes, num_detections) = tf_sess.run(
                        [boxes, scores, classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded})
                    end_time = time.time()
                    inference_time = end_time - start_time
                    log.debug(LOGGER_SCORING_MODEL_RUN,
                              msg=f'Inference time : {inference_time:.2f}s')
                    predictions = {}
                    predictions['boxes'] = boxes.tolist()
                    predictions['scores'] = scores.tolist()
                    predictions['classes'] = classes.tolist()
                    predictions['num_detections'] = num_detections.tolist()
                    predictions['inference_time'] = inference_time
                    return json.dumps(predictions)
        else:
            return (f'Model not implemented.')
    except Exception:
        log.error(
            LOGGER_SCORING_MODEL_RUN,
            f'Error : {traceback.print_exc()}')


def get_tensors():
    """
    Helper function to getting tensors for object detection API
    """
    global detection_graph
    # Extract image tensor
    image_tensor = detection_graph.get_tensor_by_name(
        'image_tensor:0')
    # Extract detection boxes
    boxes = detection_graph.get_tensor_by_name(
        'detection_boxes:0')
    # Extract detection scores
    scores = detection_graph.get_tensor_by_name(
        'detection_scores:0')
    # Extract detection classes
    classes = detection_graph.get_tensor_by_name(
        'detection_classes:0')
    # Extract number of detections
    num_detections = detection_graph.get_tensor_by_name(
        'num_detections:0')
    return image_tensor, boxes, scores, classes, num_detections


def sigterm_handler(sig, frame):
    log.info(LOGGER_SCORING_SIGTERM,
             'SIGTERM caught. Docker Container being terminated.')
    graceful_shutdown()
    log.info(LOGGER_SCORING_SIGTERM, 'SIGTERM signal processing done.')


def graceful_shutdown():
    try:
        log.info(LOGGER_SCORING_GRACEFUL_SHUTDOWN,
                 'Initiating Graceful Shutdown')
        sys.exit(0)
    except SystemExit:
        log.info(LOGGER_SCORING_GRACEFUL_SHUTDOWN, 'Exiting process.')
    except Exception:
        log.critical(LOGGER_SCORING_GRACEFUL_SHUTDOWN,
                     'trace: {}'.format(traceback.print_exc()))


def load_models(conf: dict):
    global category_index, categories, detection_graph, tf_sess
    detection_graph = tf.Graph()
    frozen_graph_path = conf['models']['object_detector']['frozen_graph_path']
    label_map_path = conf['models']['object_detector']['label_map_path']
    num_classes = conf['models']['object_detector']['num_classes']
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(frozen_graph_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            log.warning(LOGGER_SCORING_LOAD_MODELS,
                        msg=f'Loading model in memory...')
            tf.import_graph_def(od_graph_def, name='')
    log.warning(LOGGER_SCORING_LOAD_MODELS,
                msg='Model loaded in memory.')
    # Loading label map
    # Label maps map indices to category names, so that when our convolution
    # network predicts `5`, we know that this corresponds to `airplane`.
    # Here we use internal utility functions, but anything that returns a
    # dictionary mapping integers to appropriate string labels would be fine
    label_map = label_map_util.load_labelmap(label_map_path)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=num_classes, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    tf_sess = tf.Session(graph=detection_graph)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Scoring Service')
    parser.add_argument(
        "--config_file", help=f'json configuration file containint params'
                              f'to initialize the model inference service',
                              type=str)
    args = parser.parse_args()
    try:
        if not os.path.exists(args.config_file):
            raise ValueError(
                f'Cannot find configuration file "{args.config_file}"')

        with open(args.config_file, 'r') as f:
            robotJetsonConfiguration = json.load(f)

        log.warning(LOGGER_SCORING_STARTUP,
                    msg='Loading models now...')
        signal.signal(signal.SIGTERM, sigterm_handler)
        load_models(robotJetsonConfiguration)
        log.warning(LOGGER_SCORING_STARTUP,
                    msg='Launching web app...')
        ip = robotJetsonConfiguration['app']['ip']
        port=robotJetsonConfiguration['app']['port']
        app.run(host=ip, port=port)

    except SystemExit:
        log.info(LOGGER_SCORING_STARTUP, 'Caught SystemExit...')
    except Exception:
        log.critical(LOGGER_SCORING_STARTUP,
                     'Crash in startup : {}'.format(traceback.print_exc()))
    finally:
        graceful_shutdown()
        logging.shutdown()
