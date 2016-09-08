/**
* FaceDetector.cpp
* Class for detecting the face from an RGB resized image
*/

#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#include <iostream>
#include <string>
#include <vector>

using namespace std;

class FaceDetector
{

private:
	//classifier for Face Detection
	cv::CascadeClassifier detector;	

public:
	//load classifier with specificed cascade file
	FaceDetector()
	{	
		cout << "Face Detector" << endl;
		this->load();
	}
	
	bool load( string filePath="data/haarcascades/haarcascade_frontalface_alt.xml" )
	{
		if( (this->detector).load(filePath) )
		{
			cout << "Detect face with: " << filePath << endl;
			return true;
		}
		else
		{
			cout << "Cannot load classifier" << filePath << endl;
			return false;
		}
	}

	//detect faces and draw rectangle around each face from an RGB image
	void detectFace(cv::Mat &original, vector< cv::Rect > &faces)
	{
		//convert original to grayscale
		cv::Mat gray;
		cv::cvtColor(original, gray, CV_BGR2GRAY);
		cv::equalizeHist(gray, gray);
		//detect the faces from an gray image
		(this->detector).detectMultiScale( gray, faces, 1.1, 3, 0, cv::Size(30,30) );
	}

	void showFaces(cv::Mat &original, vector<cv::Rect> &faces)
	{
		// cout << "Num of faces: " << faces.size() << endl;
		for(int i=0; i<faces.size(); i++)
		{	
			cv::rectangle(original, faces[i], cv::Scalar( 255, 0, 255 ), 4, 8, 0 );
		}			
	}

};
