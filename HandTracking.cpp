/**
	Hand Tracking System
	main function
	@author Lei Liu
*/

#include <iostream>
#include <vector>
#include <string>
#include <dirent.h>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "FaceDetector.cpp"
#include "SkinClassifier.cpp"
#include "TrackingProcess.cpp"

using namespace std;

/*
* Main Object that holds all detectors and data
*/
class HandTracking
{
private:
	//create face detector
	FaceDetector faceDetector;
	//create skin classifier
	SkinClassifier skinClassifier;
	//create tracking process 
	TrackingProcess tracking; 

	float sizeScale;

public:
	HandTracking()
	{	}

	/**
	*
	*/
	void offLineTrain(string folder = "data/skin_colour/" )
	{
		skinClassifier.clear_statistic(); //clear statisic result
		skinClassifier.train(folder); //train with specificed folder
		skinClassifier.writeProbab(); //save trained probability into file
	}

	/**
	 Test the detection functions
	*/
	void detectionTest()
	{
		// temp/Megan-Fox-Pretty-Face-1-1024x768.jpg
		cv::Mat mat = cv::imread("temp/v3002.png", CV_LOAD_IMAGE_COLOR);
		if( !mat.data ) cout << "not such image" << endl;
		cv::resize( mat, mat, cv::Size(960, 540), CV_INTER_AREA );

		SkinClassifier classifier;
		classifier.train(); //train classifier
		cv::Mat bwimg;
		classifier.detect(mat, bwimg); //detect an image and draw ellipses.

		cv::Mat truth = cv::imread("temp/Megan-Fox-Pretty-Face-1-1024x768.png", CV_LOAD_IMAGE_GRAYSCALE);
		cv::imshow("Truth", truth);
	}

	//test ellipse fitting for each region of interest
	void ellipseTest()
	{
		//test ellipse functions
		uchar data[3][10] = { {0, 255, 255, 255, 255, 0, 0, 255, 255, 0},
							 {255, 255, 255, 0, 0, 0, 0, 0, 0, 255},
							 {0, 0, 0, 0, 0, 255, 255, 0, 0, 0} };
		// cv::Mat mat = cv::Mat(3, 10, CV_8U, data);
		// cv::Mat mat = cv::imread("temp/06Apr03Face.png", CV_LOAD_IMAGE_GRAYSCALE);
		uchar d[200][200] = {0};
		cv::Mat mat = cv::Mat(200, 200, CV_8UC1, d);
		for(int i=50; i<=150; i++)
		{
			for(int j=80; j<=100; j++)
			{
				mat.at<uchar>(i, j) = (uchar)255;
			}
		}

		mat = cv::imread("data/skin_colour/sarah.png", CV_LOAD_IMAGE_GRAYSCALE);
		cout << mat.size() << endl;	
		// cv::namedWindow("original", CV_WINDOW_AUTOSIZE);
		// cv::imshow("original", mat );

		Blobs blobs(mat);
		map< int, vector< pair<int, int> > > blob = blobs.getBlobs();	
		vector< pair<int, int> > b = blob.at(2); //points in a blos, pair: rowIdx and colIdx

		//show a blob only
		// for(int i=0; i<b.size(); i++)
		// {	
		// 	int x = b[i].first;
		// 	int y = b[i].second;
		// 	mat.at<uchar>(x, y) = (uchar) 100;
		// }

		// cout << "Method:                        " << "size                " << "center        " << "angle     " << endl;
		// Ellipse ell;
		// vector<cv::RotatedRect> ellipses = ell.getEllipses(mat);
		// cv::ellipse(mat, ellipses[0], cv::Scalar( 128, 128, 128 ), 2, 8);
		// cout << "Opencv find Ellipse " << ellipses[0].size << "  " << ellipses[0].center << "  " << ellipses[0].angle << endl;

		TrackingProcess pro;
		pro.newHypothesis(b);
		pro.showEllipses(mat);
		cv::RotatedRect tempEll = (pro.getHistory()).at(1);
		// vector< pair<int, int> > contour = pro.maxContour(b);
		// cv::RotatedRect contour_ellipse = pro.contourEllipse(contour);
		cv::ellipse(mat, tempEll, cv::Scalar( 100, 100, 100 ), 3, 8);
		// cout << "Opencv with blob contour       " << contour_ellipse.size << "  " << contour_ellipse.center << contour_ellipse.angle << endl;

		// cv::RotatedRect con_ = pro.newHypothesis(contour);
		// cv::ellipse(mat, con_, cv::Scalar( 255, 255, 255 ), 1, 8);
		// cout << "covariance matrix with contour " << con_.size << "  " << con_.center << "  " << con_.angle << endl;

		// pro.firstFrame(mat);
		// map<int, cv::RotatedRect> history = pro.getHistory();
		// for(map<int, cv::RotatedRect>::iterator hist=history.begin(); hist!=history.end(); ++hist)
		// {
		// 	cout << hist->first << endl;
		// 	cout << (hist->second).center << (hist->second).size << (hist->second).angle << endl; 
		// }
		// pro.showEllipses(mat);

		// show a blob only
		for(int i=0; i<b.size(); i++)
		{	
			int x = b[i].first;
			int y = b[i].second;
			mat.at<uchar>(x, y) = (uchar) 255;
		}
		//distance
		for(int i=0; i<mat.rows; i++)
		{
			for(int j=0; j<mat.cols; j++)
			{
				cv::Point p(j,i);
				double dis = pro.distance(tempEll, p);
				if( dis <= 1.0)
					mat.at<uchar>(i, j) = (uchar)200;

				// if( pro.inEllipse(tempEll, p) )
				// 	mat.at<uchar>(i, j) = (uchar)200;			
			}
		}
		cv::namedWindow("el", CV_WINDOW_AUTOSIZE);
		cv::imshow("el", mat );
	}

	//test tracking process
	void trackingTest()
	{
		TrackingProcess pro;

		//initialize first frame
		cv::Mat frame = cv::Mat::zeros(200, 200, CV_8UC1);
		for(int r=40; r<=60; r++)
		{
			for(int c=10; c<=90; c++)
			{
				frame.at<uchar>(r, c) = (uchar)255;
			}
		}
		pro.firstFrame(frame, 0);
		pro.showEllipses(frame);
		cv::namedWindow("firstFrame", CV_WINDOW_AUTOSIZE);
		cv::imshow("firstFrame", frame);

		//next frame, has intersection blob with previous one, and new blob
		cv::Mat next = cv::Mat::zeros(200, 200, CV_8UC1);
		for(int r=50; r<=80; r++)
		{
			for(int c=50; c<=150; c++)
			{
				next.at<uchar>(r, c) = (uchar)255;
			}
		}
		//new blob
		for(int r=100; r<=140; r++)
		{
			for(int c=70; c<=150; c++)
			{
				next.at<uchar>(r, c) = (uchar)255;
			}
		}
		pro.track(next, 0);
		pro.showEllipses(next);
		cv::namedWindow("nextFrame", CV_WINDOW_AUTOSIZE);
		cv::imshow("nextFrame", next);

		//third frame, has intersection blob with previous one, and new blob
		cv::Mat third = cv::Mat::zeros(200, 200, CV_8UC1);
		for(int r=50; r<=100; r++)
		{
			for(int c=50; c<=150; c++)
			{
				third.at<uchar>(r, c) = (uchar)255;
			}
		}
		//new blob
		for(int r=100; r<=140; r++)
		{
			for(int c=70; c<=150; c++)
			{
				third.at<uchar>(r, c) = (uchar)255;
			}
		}
		pro.track(third, 0);
		pro.showEllipses(third);
		cv::namedWindow("thirdFrame", CV_WINDOW_AUTOSIZE);
		cv::imshow("thirdFrame", third);

		//fourth frame, has intersection blob with previous one, and new blob
		cv::Mat fourth = cv::Mat::zeros(200, 200, CV_8UC1);
		for(int r=50; r<=80; r++)
		{
			for(int c=50; c<=90; c++)
			{
				fourth.at<uchar>(r, c) = (uchar)255;
			}
		}
		for(int r=50; r<=80; r++)
		{
			for(int c=100; c<=150; c++)
			{
				fourth.at<uchar>(r, c) = (uchar)255;
			}
		}
		for(int r=100; r<=140; r++)
		{
			for(int c=70; c<=150; c++)
			{
				fourth.at<uchar>(r, c) = (uchar)255;
			}
		}
		pro.track(fourth, 0);
		pro.showEllipses(fourth);
		cv::namedWindow("fourthFrame", CV_WINDOW_AUTOSIZE);
		cv::imshow("fourthFrame", fourth);

		//fifth frame, has intersection blob with previous one, and new blob
		cv::Mat fifth = cv::Mat::zeros(200, 200, CV_8UC1);
		pro.track(fifth, 0);
		pro.showEllipses(fifth);
		cv::namedWindow("fifthFrame", CV_WINDOW_AUTOSIZE);
		cv::imshow("fifthFrame", fifth);
	}	

	//test face detection in real-time camera
	void faceTest(string videoName, float rotateAngle)
	{		
		//get a handle to the video device, load video as fault
		cv::VideoCapture videoCap;
		int deviceId = atoi( videoName.c_str() );
		if( (videoName.size() == 1) && deviceId == 0 ) //load camera
			videoCap = cv::VideoCapture( deviceId );
		else //load video
			videoCap = cv::VideoCapture( videoName );
		//check if the video or camera is loaded successful.
		if( !videoCap.isOpened() )
		{
			std::cout << "Video or Camera cannot be opened." << endl;
			return;
		}
		FaceDetector faceDetector; //create face detector
		faceDetector.load();
		int winWidth = 480;
		int winHeight = 320;
		string winName = "Hand Tracking";
		//create window for displaying images
		cv::namedWindow(winName, CV_WINDOW_AUTOSIZE);
		cv::resizeWindow(winName, winWidth, winHeight);
		cv::Mat frame; //Holds the current fram from the Video device
		bool isFirst = true;
		for(;;)
		{
			videoCap >> frame;
			if( !frame.data )
				break;
			//clone the current frame and resize to default size
			cv::Mat original = frame.clone();
			if( rotateAngle != 0) //rotate original frame
			{
				cv::Mat rotateMat = cv::getRotationMatrix2D( cv::Point(original.cols/2, original.rows/2), rotateAngle, 1 );
				cv::warpAffine( frame, original, rotateMat, cv::Size(frame.cols, frame.rows) );
			}
			cv::resize( original, original, cv::Size(winWidth, winHeight), CV_INTER_AREA );
			cv::Mat bwimg; //current bwimg after detection

			//Detect face for each frame
			vector<cv::Rect> faces;
			faceDetector.detectFace( original, faces );
			faceDetector.showFaces(original, faces);

			//show the result
			cv::imshow(winName, original);
				
			//control flow
			char key = (char) cv::waitKey(20);
			//pause loop by pressing "space" key
			if( key==32 )
				cv::waitKey(0);
			if( key==32 )
				continue;
			if(key == 27)//Exit the loop bt pressing "Esc"
				break;
		}
	}

	void cameraTest(string videoName, float rotateAngle, int winWidth, int winHeight, string winName)
	{		
		//get a handle to the video device, load video as fault
		cv::VideoCapture videoCap;
		int deviceId = atoi( videoName.c_str() );
		if( (videoName.size() == 1) && deviceId == 0 ) //load camera
			videoCap = cv::VideoCapture( deviceId );
		else //load video
			videoCap = cv::VideoCapture( videoName );
		//check if the video or camera is loaded successful.
		if( !videoCap.isOpened() )
		{
			std::cout << "Video or Camera cannot be opened." << endl;
			return;
		}
		
		cv::Mat frame; //Holds the current fram from the Video device
		bool isFirst = true;
		int count =0;
		for(;;)
		{
			videoCap >> frame;
			if( !frame.data )
				break;

			count++;
			//clone the current frame and resize to default size
			cv::Mat original = frame.clone();
			if( rotateAngle != 0) //rotate original frame
			{
				cv::Mat rotateMat = cv::getRotationMatrix2D( cv::Point(original.cols/2, original.rows/2), rotateAngle, 1 );
				cv::warpAffine( frame, original, rotateMat, cv::Size(frame.cols, frame.rows) );
			}
			cv::resize( original, original, cv::Size(winWidth, winHeight), CV_INTER_AREA );
			const string nm = "testing/tooltill_1/tooltill_" + to_string(count) + ".jpg";
			cout << nm << " ";
			cout << cv::imwrite(nm, original) << endl;

			cv::Mat bwimg; //current bwimg after detection

			//Detect face for each frame
			vector<cv::Rect> faces;
			faceDetector.detectFace( original, faces );
			float ave_face_size = 0;
			//update skin color probability with detected face areas
			for(int i=0; i<faces.size(); i++)
			{
				cv::Mat face(original, faces[i]);
				skinClassifier.adaptive(face);
				ave_face_size += ( face.rows * face.cols );
			}
			ave_face_size = ave_face_size / (faces.size()) * sizeScale;
		
			//Detect skin area for current frame
			skinClassifier.detect( original, bwimg );	
			//tracking
			if(isFirst)
			{
				tracking.firstFrame(bwimg, ave_face_size);
				isFirst = false;
				cout << "First frame" << endl;
			}
			else //tracking
			{
				tracking.track(bwimg, ave_face_size);
				cout << " next frame " << endl;
			}
			//draw ellipse and rectangle on original
			tracking.showEllipses(original);
			faceDetector.showFaces(original, faces);
			//show the result
			cv::imshow(winName, original);

			//control flow
			char key = (char) cv::waitKey(20);
			//pause loop by pressing "space" key
			if( key==32 )
				cv::waitKey(0);
			if( key==32 )
				continue;
			if(key == 27)//Exit the loop bt pressing "Esc"
				break;
		}// end of video tracking------------------------------
	}

	void setThresholds(float upper_T, float lower_T)
	{
		skinClassifier.setThresholds(upper_T, lower_T);
	}

	void setSizeScale(float scale)
	{
		this->sizeScale = scale;
	}

	void offlineTesting()
	{
		cout << "5-fold cross-validation for off-line skin detector." << endl;
		skinClassifier.testSkinDetector();
	}

	void getFilenames(string folder, vector<string> &filepaths, bool inSequence=false)
	{
		//load a pair of training image and turth image from the folder
		DIR *dir;
		struct dirent *entry;
		//load training images fold, get list of images name
		dir = opendir( folder.c_str() );
		if(dir)
		{
			while( entry = readdir(dir) )
			{
				if( strcmp( entry->d_name, "." ) != 0 && strcmp( entry->d_name, ".." ) != 0 )
				{
					std::string temp(entry->d_name);
					if( !(temp.at(temp.size()-4) == '.') ) continue;
					//concact train image and truth image file path
					std::string path = folder + temp;
					ifstream fin;
					fin.open(path);
					if(fin)
					{
						filepaths.push_back(temp);
						fin.close();
					}
				}
			}
			closedir(dir);
		}
		else
		{
			std::cout << "open dataset folder failed, path: " << folder << std::endl;
		}

		//sort the filename in sequences of frame number
		if(inSequence)
		{
			for(int i=0; i<filepaths.size()-1; i++)
			{
				for(int j=0; j< filepaths.size()-i-1; j++)
				{
					string p1 = filepaths[j];
					int num1 = std::stoi( p1.substr( (p1.find_last_of("_"))+1, (p1.size()-4) ) );
					string p2 = filepaths[j+1];
					int num2 = std::stoi( p2.substr( (p2.find_last_of("_"))+1, (p2.size()-4) ) );
					if(num1 > num2)
					{
						filepaths[j] = p2;
						filepaths[j+1] = p1; 
					}
				}
			}
		}
	}

	void onlineTrain(string folder, vector<string> filenames)
	{
		//train with adaptive
 	 	cout << "Retrain classifier with detected faces" << endl;
 	 	for(int i=0; i<filenames.size(); i++)
 	 	{
 	 		cv::Mat img = cv::imread( (folder+filenames[i]), CV_LOAD_IMAGE_COLOR);
 	 		if( !img.data ) continue;

 	 		//adaptive with face
 			vector< cv::Rect > faces;
			faceDetector.detectFace(img, faces);
			for(int f=0; f<faces.size(); f++)
			{
				cv::Mat face(img, faces[f]);
				skinClassifier.adaptive(face);
			}
 	 	}
	}

	void onlineTesting(string folder)
	{
		cout << "n-flod cross-validation for on-line skin detector with detected Face.\n";

		//get all testing image paths
		vector<string> filepaths;
		//folder: testing/LF31/
		getFilenames(folder, filepaths);
		//randomizing the vector of file paths
 	 	std::random_shuffle ( filepaths.begin(), filepaths.end() );

		vector<string>::iterator split = filepaths.begin() + filepaths.size()*0.1;
		vector<string> testFiles(filepaths.begin(), split);
 	 	vector<string> trainFiles(split, filepaths.end());
 	 	cout << trainFiles.size() << endl;
 	 	cout << testFiles.size() << endl;

 	 	//testing testFiles on offline probability
 	 	cout << "Test files with Offline probability" << endl;
 	 	for(int i=0; i<testFiles.size(); i++)
 	 	{
 	 		cv::Mat img = cv::imread( (folder+testFiles[i]), CV_LOAD_IMAGE_COLOR);
 	 		if( !img.data ) continue;
 	 		cv::Mat bwimg(img.size(), CV_8UC1);
 	 		skinClassifier.detectSkin(img, bwimg);
 	 		skinClassifier.connectPotential(bwimg);
 	 		string nm = folder + "offline/" + testFiles[i];
 	 		nm = nm.replace( (nm.length()-3), 3, "png");
 	 		// cout << nm << endl;
 	 		cv::imwrite(nm, bwimg);
 	 	}

 	 	//train with adaptive
 	 	cout << "Retrain classifier with detected faces" << endl;
 	 	for(int i=0; i<trainFiles.size(); i++)
 	 	{
 	 		cv::Mat img = cv::imread( (folder+trainFiles[i]), CV_LOAD_IMAGE_COLOR);
 	 		if( !img.data ) continue;

 	 		//adaptive with face
 			vector< cv::Rect > faces;
			faceDetector.detectFace(img, faces);
			for(int f=0; f<faces.size(); f++)
			{
				cv::Mat face(img, faces[f]);
				skinClassifier.adaptive(face);
			}
 	 	}
 	 	cout << "Finished online training" << endl;
 	 	
 	 	//testing testFiles on Online probability
 	 	cout << "Test files with Online probability" << endl;
 	 	for(int i=0; i<testFiles.size(); i++)
 	 	{
 	 		cv::Mat img = cv::imread( (folder+testFiles[i]), CV_LOAD_IMAGE_COLOR);
 	 		if( !img.data ) continue;
 	 		cv::Mat bwimg(img.size(), CV_8UC1);
 	 		skinClassifier.detectSkin(img, bwimg);
 	 		skinClassifier.connectPotential(bwimg);
 	 		// imshow("original", img);
 	 		// imshow("offline Testing", bwimg);
 	 		string nm = folder + "online/" + testFiles[i];
 	 		nm = nm.replace( (nm.length()-3), 3, "png");
 	 		// cout << nm << endl;
 	 		cv::imwrite(nm, bwimg);
 	 	}

 	 	cout << "Finished all testing!" << endl;
	}

	void compare(string test_folder, string truth_folder)
	{
		cout << "Compare result with correct: " << endl;
		cout << test_folder << " VS " << truth_folder << endl;
		int TP = 0; int FN = 0;
		int FP = 0; int TN = 0;

		vector<string> testFiles;
		this->getFilenames(test_folder, testFiles);
		for(int i=0; i<testFiles.size(); i++)
		{
			string testFile = test_folder + testFiles[i];
			string truthFile = truth_folder + testFiles[i];
			cv::Mat test_image = cv::imread( testFile, CV_LOAD_IMAGE_GRAYSCALE);
			cv::Mat truth_image = cv::imread( truthFile, CV_LOAD_IMAGE_GRAYSCALE);
			if( (!test_image.data) || (!truth_image.data) ) continue;
			//compare with truth image
			for(int i=0; i < truth_image.rows; i++)
			{
				for( int j=0; j < truth_image.cols; j++)
				{
					int truth = (int)truth_image.at<uchar>(i, j);
					int pred = (int)test_image.at<uchar>(i, j);
					if( truth==255 ) //actual positive
					{
						if( pred==255 ) TP++; //true positive
						if( pred==0 ) FN++; //false positive
					}
					else if( truth==0 ) //actual negative
					{
						if( pred==255 ) FP++; //false positive
						if( pred==0 ) TN++; //true negative
					}
				}
			}
		}	
		//Evaluation
		int total = TP + FN + FP + TN;
		float accuracy = (float)(TP + TN) / (float)total;
		float recall = (float)TP / (float)(TP+FN);
		float precision = (float)TP / (float)(TP + FP);
		float f1 = (2 * precision * recall) / (precision + recall); 
		cout << "Accuracy: " << accuracy << '\n'
			<< "Recall Rate: " << recall << '\n'
			<< "Precision: " << precision << '\n'
			<< "F1 feature: " << f1 << endl;
	}

	void captureFrames(string path)
	{	
		string winName = "Hand Tracking";
		int winWidth = 480;
		int winHeight = 320;
		//get a handle to the video device, load video as fault
		cv::VideoCapture videoCap = cv::VideoCapture( 0 );
		//check if the video or camera is loaded successful.
		if( !videoCap.isOpened() )
		{
			std::cout << "Video or Camera cannot be opened." << endl;
			return;
		}
		
		cv::Mat frame; //Holds the current fram from the Video device
		int count = 0;
		for(;;)
		{
			videoCap >> frame;
			if( !frame.data )
				break;
			count++;
			//clone the current frame and resize to default size
			cv::Mat original = frame.clone();
			cv::resize( original, original, cv::Size(winWidth, winHeight), CV_INTER_AREA );
			cv::imshow("A frame", original);

			const string nm = path + to_string(count) + ".jpg";
			cout << nm << " ";
			cout << cv::imwrite(nm, original) << endl;

			//control flow
			char key = (char) cv::waitKey(20);
			//pause loop by pressing "space" key
			if( key==32 )
				cv::waitKey(0);
			if( key==32 )
				continue;
			if(key == 27)//Exit the loop bt pressing "Esc"
			{
				videoCap.release();
				break;
			}
		}// end of video tracking------------------------------
	}

	//online training
	void reTrain(string folder)
	{
		vector<string> filenames;
		this->getFilenames( (folder+"correct/"), filenames);
		for(int i=0; i<filenames.size(); i++)
 	 	{
 	 		string truth_file = folder+"correct/"+filenames[i];
 	 		cv::Mat truth_image = cv::imread( truth_file, CV_LOAD_IMAGE_GRAYSCALE );
 	 		string train_file = folder+filenames[i]; 
 	 		train_file = train_file.replace( (train_file.length()-3), 3, "jpg");
 	 		cv::Mat train_image = cv::imread( train_file, CV_LOAD_IMAGE_COLOR);
 	 		if( (!train_image.data) || (!truth_image.data) ) continue;
 	 		skinClassifier.updateStatistic(train_image, truth_image);
 	 		cout << "Retrian classifier with " << train_file << endl;
 	 	}
 	 	skinClassifier.calculProbab();
	}

	void testTracking(string folder, string retrian_folder)
	{
		this->reTrain(retrian_folder);
		cout << "retrain detector DONE \n";

		vector<string> sequen;
		this->getFilenames(folder, sequen, true);

		bool isFirst = true;
		float minSize = 0;
		for(int i=0; i<sequen.size(); i++)
		{
			//get a frame
			string file = folder + sequen[i];
			cv::Mat img = cv::imread(file, CV_LOAD_IMAGE_COLOR);
			if( !img.data ) continue;
			// cv::imshow("original", img);
			cout << sequen[i] << endl;

			//detect skin blobs
			cv::Mat bwimg(img.size(), CV_8UC1);
			skinClassifier.detectSkin(img, bwimg);
			skinClassifier.connectPotential(bwimg);
			// cv::imshow("skin bolbs", bwimg);
	 		string nm = folder + "detected/" + sequen[i];
 	 		nm = nm.replace( (nm.length()-3), 3, "png");
 	 		cv::imwrite(nm, bwimg);
 	 		cv::imshow("Detected Skin blobs", bwimg);
			//tracking blobs
			//Para for size filtering
			vector<cv::Rect> faces;
			faceDetector.detectFace( img, faces );
			float ave_face_size = 0;
			//update skin color probability with detected face areas
			for(int i=0; i<faces.size(); i++)
			{
				cv::Mat face(img, faces[i]);
				ave_face_size += ( face.rows * face.cols );
			}
			ave_face_size = (ave_face_size / (faces.size()) )*0.2;

			if( ave_face_size != 0 ) minSize = ave_face_size;	
			string filename = folder + "blobs/" + sequen[i];
 	 		filename = filename.replace( (filename.length()-3), 3, "png");
			//tracking
			if(isFirst)
			{
				tracking.firstFrame(bwimg, minSize, filename);
				isFirst = false;
			}
			else //tracking
				tracking.track(bwimg, minSize, filename);
			tracking.showEllipses(img, sequen[i]);

			// cv::imshow("Tracking", img);
			string tracked = folder + "tracked/" + sequen[i];
 	 		cv::imwrite(tracked, img);
 	 		cv::imshow("Tracking", img);

 	 		char key = (char) cv::waitKey(20);
			//pause loop by pressing "space" key
			if( key==32 )
				cv::waitKey(0);
			if( key==32 )
				continue;
			if(key == 27)//Exit the loop bt pressing "Esc"
				break;

		}
		tracking.writeTrajectory( (folder+"tracked/trajectory.txt") );
	}

};

HandTracking hand;

void on_upperThresh(int upper, void *userData)
{
	float upper_T = (float)upper / 100.0;
	float lower_T = (float)(*( static_cast<int*>(userData) )) / 100.0;
	hand.setThresholds(upper_T, lower_T);
}

void on_lowerThresh(int lower, void *userData)
{
	float lower_T = (float)lower / 100.0;
	float upper_T = (float)(*( static_cast<int*>(userData) )) / 100.0;
	hand.setThresholds(upper_T, lower_T);
}

void on_sizeThresh(int size, void *userData)
{
	float scale = (float) size / 100.0;
	hand.setSizeScale(scale);
}

//mian function, choice for either load camera or read local video
//capture frame for both skin detection and face detection
int main(int argc, const char *argv[])
{
	/*
	* ---------------------------------------------
	* tracking in video
	* 0 for camera, otherwise path of the video file
	* input rotateangle, for camera it is 0
	* Example: ./HandTracking 0 0
	* ./HandTracking ../dataset/video/video1.mp4 -90
	*/
	string videoName;
	float rotateAngle = 0;
	//check the input parameter for choosing the camera or loading video
	if(argc < 2)
	{
		cout << "start program with an arguments, 0 for camera or the path of video file" << endl;
		return -1;
	}
	else if(argc == 2)
	{
		videoName = argv[1];
	}
	else
	{
		videoName = argv[1];
		rotateAngle = atof( argv[2] );
	}

	string winName = "Hand Tracking";
	int winWidth = 480;
	int winHeight = 320;
	//create window for displaying images
	cv::namedWindow(winName, CV_WINDOW_AUTOSIZE);
	cv::resizeWindow(winName, winWidth, winHeight);
	//create trackbar to control thresholds
	int upper_thresh = 50;
	int lower_thresh = 15;
	int size_thresh = 20;
	cv::createTrackbar("Skin Colour Upper Threshold", winName, 
						&upper_thresh, 100, on_upperThresh, &lower_thresh);
	cv::createTrackbar("Skin Colour Lower Threshold", winName, 
						&lower_thresh, 100, on_lowerThresh, &upper_thresh);
	cv::createTrackbar("Size filter Threshold: ", winName, 
						&size_thresh, 100, on_sizeThresh);
	cout << "Start detection and tracking:\n";
	hand.cameraTest(videoName, rotateAngle, winWidth, winHeight, winName);
	// hand.ellipseTest();
	// hand.faceTest(videoName, rotateAngle);

	cv::waitKey(0);
	return 0;
}

//for evaluation
int main0(int argc, const char *argv[])
{
	string folder = "testing/cafe_1/";
	// hand.offlineTesting();
	// hand.captureFrames("testing/room_oom/room_");
	// hand.onlineTesting(folder);
	// hand.compare( (folder+"offline/"), (folder+"correct/") );
	// hand.compare( (folder+"online/"), (folder+"correct/"));
	string retrian_folder = "testing/cafe/";
	hand.testTracking(folder, retrian_folder);

	cv::waitKey(0);
	return 0;
}
