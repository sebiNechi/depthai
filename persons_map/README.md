"# depthai" 
#Person Tracking and Counting with DepthAI
This Python script uses the DepthAI library, an interface to Intel's OpenVINO toolkit, to detect and track persons in a video.

##Overview
The script sets up a processing pipeline where each video frame is resized and passed to a pre-trained MobileNet detection model. The MobileNet model, which has been fine-tuned for person detection, then identifies any persons in the frame. The detections are tracked across frames using the ObjectTracker node, allowing for continuous monitoring of each individual.

##Features
The script contains several unique features designed to enhance person tracking:

Unique Identification: Each detected person is assigned a unique ID by the ObjectTracker, and their detection status is stored in the tracked_ids dictionary.

Person Counting: The total number of unique persons detected in the video is maintained, and the current count is displayed on the video frame.

Live Feed: The current video frame, complete with bounding boxes around detected persons and their unique IDs, is displayed in real-time.

FPS Calculation: The script calculates and displays the Frames Per Second (FPS) of the video processing pipeline, useful for performance benchmarking.

##Setup
To use this script, you'll need to have the DepthAI library installed. You can do this using pip:

pip install depthai

You'll also need to have OpenCV installed:

pip install opencv-python

Next, you'll need to download the person-detection-retail-0013 model and place it in the models folder in the root directory of this project. You can download this model from the OpenVINO Model Zoo https://docs.openvino.ai/2022.3/omz_models_model_person_detection_retail_0013.html.

Finally, place the video file you want to process in the models directory and set its path as the --videoPath argument when running the script.

##Running the Script
To run the script, navigate to the root directory of the project and use the following command:

python3 main.py --nnPath models/person-detection-retail-0013.blob --videoPath models/your_video.mp4
