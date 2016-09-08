# HandTracking
This is the final project for master degree of The University of Manchester
2016.05 - 2016.08
Lei Liu

------------------------------
Project Description:
The aims of this prototype were to detect an arbitrary number of objects which specifically represent the areas of an image that contain human skin, and then to track the movement of detected objects in a sequence of image frames in real-time performance.

------------------------------
Project Objectives:
1. System Initialization. In this step, a camera device or a valid video file is loaded for capturing images and then tracking hand objects within the sequence of video.

2. Pre-process of an image. The colour information that employed in this prototype is the UV channels which are extracted from YUV colour space. Every original image is converted to YUV colour space before the processes of detection and tracking.

3. Object detection. For each captured image, a skin colour classifier is applied to detect those pixels that contain skin colour information.

4. Object generation. The pixels that detected in the above step are able to generalise valid objects which are the area of skin colour in an image (e.g. a face region, a hand, or any skin region). A connected-component analysis technique is used for gathering the skin colour pixels and labelling distinctive objects of skin colour areas.

5. Object tracking. With detected regions in an image, their positions are associated with the history of previous frames in order to show the trajectory of moving objects in a video.

------------------------------
Development Environment:
Linux(Ubuntu 14.04)
CMake v2.8 + C++11 + OpenCV v3.0

------------------------------
flow-chart.png:
The flowchart explained the working process of this system. 

There are mainly three components in this system:
1) object detection, implemented by "SkinClassifier.cpp"
2) object generation, implemented by "Blobs.cpp" 
3) object tracking, implemented by "TrackingProcess.cpp"

Folder: data/skin_colour contains the training images with corresponding .ground truth binary image.