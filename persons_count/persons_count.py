from pathlib import Path
import depthai as dai
import cv2
import numpy as np

# Helper function
def frame_to_planar(frame, shape=(300, 300)):
    return cv2.resize(frame, shape).transpose(2,0,1).flatten()

# Start defining a pipeline
pipeline = dai.Pipeline()

# Define a source - color camera
cam_rgb = pipeline.createXLinkIn()
cam_rgb.setStreamName('in')
# cam_rgb.setMetadataSize(4)  # Set the metadata size

# Create output links
xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("nn")

# NeuralNetwork
nn = pipeline.createNeuralNetwork()
nn.setBlobPath(((Path(__file__).parent / Path('../depthai-python/examples/models/person-detection-retail-0013_openvino_2021.4_7shave.blob')).resolve().absolute()))
cam_rgb.out.link(nn.input)
nn.out.link(xout_nn.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    q_in = device.getInputQueue('in')
    q_nn = device.getOutputQueue('nn')

    cap = cv2.VideoCapture(str((Path(__file__).parent / Path('../depthai-python/examples/models/airport_crowd.mp4')).resolve().absolute()))
    person_count = 0

    while cap.isOpened():
        read_correctly, frame = cap.read()

        if not read_correctly:
            break

        # Resize the frame to 300x300 before processing
        small_frame = cv2.resize(frame, (300, 300))

        # Prepare the frame to be fed into the neural network
        nn_data = dai.NNData()
        nn_data.setLayer("input", frame_to_planar(small_frame))
        q_in.send(nn_data)

        # Get the neural network output
        in_nn = q_nn.get()
        detections = in_nn.getFirstLayerFp16()

        # Count the number of people detected in the frame
        frame_person_count = sum(1 for detection in detections if detection.label == 1)
        person_count += frame_person_count

    print(f'Total number of people detected: {person_count}')
