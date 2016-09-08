/*
* SkinClassifier.cpp
* Class for classifying the skin area of an RGB image using Naive Bayes Classifier and YUV channels
*/

#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <dirent.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;

class SkinClassifier
{

private:
	static const int num_of_bins = 256; //the number of bins for color histogram
	int color_Hist[num_of_bins][num_of_bins]; //counting the apparence of each color
	int skin_Hist[num_of_bins][num_of_bins]; //counting the apparence of skin color
	int num_color; //total number of all colors
	int num_skin; //total number of all skin colors
	double probab[num_of_bins][num_of_bins]; //probability of colro being skin color

	float Tmax;//upper threshold of skin color probability
	float Tmin;//low threshold of skin color probability
	string prob_file_path;//file name for storing the probability output.
	
public:

	SkinClassifier( string probability_file_path = "probability.txt", float upper_threshold = 0.5, float low_threshold = 0.15 )
	{
		std::cout<< "Bayesian Classifier for Skin-color Detection." << std::endl;
		Tmax = upper_threshold;
		Tmin = low_threshold;
		prob_file_path = probability_file_path;
		//load offline probability by default
		loadProbab();
		std::cout<< "Loading probability map successfully" << std::endl;
		cout << "Thresholds: " << Tmax << " -- " << Tmin << endl;
	}

	void setThresholds(float upper, float lower)
	{
		this->Tmax = upper;
		this->Tmin = lower;
		cout << "Set thresholds between: " << Tmax << " -- " << Tmin << endl;
	}

	//train classifier with a folders contianing both training image and ground truth image
	//default dataset folder is "../dataset/skin_dataset/familyPhoto/"
	void train( string dataset_folder = "../dataset/skin_dataset/FamilyPhoto/")
	{	

		cv::Mat train_image;
		cv::Mat truth_image;
		
		//load a pair of training image and turth image from the folder
		DIR *dir;
		struct dirent *entry;
		//load training images fold, get list of images name
		dir = opendir( dataset_folder.c_str() );
		if(dir)
		{
			while( entry = readdir(dir) )
			{
				if( strcmp( entry->d_name, "." ) != 0 && strcmp( entry->d_name, ".." ) != 0 )
				{
					std::string temp(entry->d_name);
					//concact train image and truth image file path
					std::string train_file_path = dataset_folder + temp;
					std::string truth_file_path = dataset_folder + "GroundTruth/" + temp.replace( (temp.length()-3),3, "png");
					//read image file to Mat
					train_image = cv::imread( train_file_path, CV_LOAD_IMAGE_COLOR);//8UC3 BGR format image
					truth_image = cv::imread( truth_file_path, CV_LOAD_IMAGE_GRAYSCALE);//8UC1 one intensity image
					//check both train image and truth image are loaded successfully
					if( ! train_image.data )
					{
						std::cout << "Could not read or find image: " << train_file_path << std::endl;
						continue;
					}
					if( ! truth_image.data )
					{
						std::cout << "Could not read or find image: " << truth_file_path << std::endl;
						continue;
					}
					//update the statistic of color hist and skin color hist
					// cout << "Update with " << train_file_path << endl;
					updateStatistic(train_image, truth_image);
				}
			}
			closedir(dir);
		} 
		else
		{
			std::cout << "open dataset folder failed, path: " << dataset_folder << std::endl;
		}
	}
	
	//input a pair of training image and truth image, update the statistic of pixels
	void updateStatistic( cv::Mat &train_image, cv::Mat &truth_image )
	{
		//convert train image to UV model
		cv::Mat uv;
		BGR2UV( train_image, uv );//get 8SC2 mat
		for(int i=0; i < uv.rows; i++)
		{
			for( int j=0; j < uv.cols; j++)
			{
				//get a training pixel
				cv::Vec2b intensity = uv.at<cv::Vec2b>( i, j );
				int u = intensity.val[0];
				int v = intensity.val[1];
				//update the color histogram statistic
				color_Hist[u][v]++;
				num_color++;
				//get corresponding label for this pixel
				int label = (int)truth_image.at<uchar>(i, j);
				//update the skin histogram if this pixel is a skin pixel
				if( label == 255 )	
				{
					skin_Hist[u][v]++;
					num_skin++;	
				} 
			}
		}
	}

	//convert 8UC3 BGR color image to 8UC2 image mat contains only the UV channels from YUV format
	void BGR2UV(cv::Mat &src_img, cv::Mat &dst_img)
	{
		cv::Mat yuv;
		//convert BGR color image to YCrCv color space
		cv::cvtColor(src_img, yuv, CV_BGR2YCrCb);
		vector<cv::Mat> channels;
		cv::split(yuv, channels);
		channels.erase( channels.begin() );//omit Y channel
		//merge UV channels into mat
		cv::merge(channels, dst_img);
	}

	/*	
	* write each pair of pixel and its statistic and probability being a skin color as one line.
	* format: U-component V-component color_histogram skin_histogram probability
	* format example: 128 128 10000 1000 0.1
	*/	
	void writeProbab(string filename = "probability.txt")
	{
		ofstream file;
		file.open( filename.c_str(), ios::trunc ); //open file and deleted existed data
		for(int i=0; i<num_of_bins; i++)
		{
			for(int j=0; j<num_of_bins; j++)
			{
				//calculate probability of each colro being a skin color
				int color_count = color_Hist[i][j];
				int skin_count = skin_Hist[i][j];
				//post probability of P(s|c) a color pixel being skin color
				if( color_count != 0 ) probab[i][j] = (double)skin_count / (double)color_count;			
				//write statistic and probability into file
				file << i << " " << j << " " << color_count << " " << skin_count << " " << probab[i][j] << "\n";
			}
		}
		file.close();
	}
	
	void calculProbab()
	{
		for(int i=0; i<num_of_bins; i++)
		{
			for(int j=0; j<num_of_bins; j++)
			{
				//calculate probability of each colro being a skin color
				int color_count = color_Hist[i][j];
				int skin_count = skin_Hist[i][j];
				//post probability of P(s|c) a color pixel being skin color
				if( color_count != 0 ) probab[i][j] = (double)skin_count / (double)color_count;
			}
		}
	}

	/*
	* load probab from file
	* file format: U-component V-component color_histogram skin_histogram probability
	*/
	void loadProbab(string filename = "probability.txt")
	{
		//clear array space
		clear_statistic();
		//open file, and read each lime once
		ifstream file( filename.c_str() );
		string line;
		if( file.is_open() )
		{
			while( getline(file, line))
			{		
				stringstream data(line);
				int u, v, color_count, skin_count;
				double p;
				data >> u >> v >> color_count >> skin_count >> p;
				color_Hist[u][v] = color_count;
				skin_Hist[u][v] = skin_count;
				num_color += color_count;
				num_skin += skin_count;
				probab[u][v] = p;
			}
			file.close();
		}
		else cout << "Unable to open " << filename << endl;	
	}
	
	/*
	* Detect skin color area and draw correspoding ellipses 
	*/
	void detect( cv::Mat &original, cv::Mat &bwimg )
	{	
		// cv::imshow("Original", original);
		cv::Mat dst( original.rows, original.cols, CV_8UC1);

		//detect skin color pixels
		detectSkin(original, dst);
		cv::imshow("Detected Skin Color", dst);

		//connect the potential skin color pixel(probability between Tman and Tmin)
		connectPotential(dst);
		cv::imshow("Connect potential pixels", dst);

		//eliminate noise and morphology operation
		smoothing(dst);
		// cv::imshow("smoothing", dst);
		// after smoothing, the output is a binary image where white areas are the skin areas
		bwimg = dst;		
	}
	
	/*
	* Classify the skin color pixel from the original
	*/
	void detectSkin(cv::Mat &original, cv::Mat &dst)
	{
		//convert to UV channels
		cv::Mat uv; 
		this->BGR2UV(original, uv);	 
		for(int i=0; i < uv.rows; i++)
		{
			for( int j=0; j < uv.cols; j++)
			{
				cv::Vec2b intensity = uv.at<cv::Vec2b>( i, j );
				int u = intensity.val[0];
				int v = intensity.val[1];
				//get probability from mapper
				double probability = probab[u][v];
				if( probability > Tmax )//skin color
				{
					dst.at<uchar>(i, j) = (uchar)255;//seed skin color point
				}
				else if( probability < Tmin )
				{
					dst.at<uchar>(i, j) = (uchar)0;//no skin color
				}
				else
				{
					dst.at<uchar>(i, j) = (uchar)128;//potential skin color
				}
			}
		}
	}

	/*
	* connect those potential skin color pixels
	*/
	void connectPotential(cv::Mat &img)
	{
		int radius = 1;
		for(int i=0; i < img.rows; i++)
		{
			for( int j=0; j < img.cols; j++)
			{
				int pixel = (int)img.at<uchar>(i, j);
				if(pixel == 128) //potential skin-color pixel
				{
					neightbourFiltering(img, i, j, radius);
				}
			}
		}
	}

	/**
	* neightbour filtering, three conditions
	* 1. immediate neighbour is skin color(255), set as skin colors
	* 2. immediate neighbours have not skin color, but has non-skin color(0), set as non-skin color
	* 3. immediate neighbours are all potential skin color(128), look for second neighbours, repeat these conditions.
	*/
	void neightbourFiltering(cv::Mat &img, int point_i, int point_j, int radius )
	{	
		// int num_skin_neightbour = 0;
		// int num_non_skin_neightbour = 0;
		bool is_skin = false;
		int i = point_i;
		int j = point_j;
		for(int m=i-radius; m<=i+radius; m++)
		{
			if( m >= img.rows || m < 0 ) continue; //check vertical boundary
			for(int n=j-radius; n<=j+radius; n++)
			{
				if( n >= img.cols || n < 0 ) continue; //check horizental boundary
				//immidiate neightbour is skin color
				int neightbour = (int)img.at<uchar>(m, n);
				if( neightbour == 255 ) 
				{
					img.at<uchar>(i, j) = (uchar)255;
					is_skin = true;
					break;
				}
				// if( neightbour == 255 ) num_skin_neightbour++;
				// if( neightbour == 0 ) num_non_skin_neightbour++;
			}
		}
		if(is_skin == false) img.at<uchar>(i, j) = (uchar)0;
	}

	void smoothing(cv::Mat &img)
	{
		//closing and opening by OPENCV
		int radius = 1;
		cv::Mat element = cv::getStructuringElement( cv::MORPH_RECT, cv::Size( 2*radius + 1, 2*radius+1 ), cv::Point( radius, radius ) );
		cv::Mat temp;
		//closing
		cv::dilate(img, temp, element);
		cv::erode(temp, temp, element);
		
		//opening
		cv::erode(temp, temp, element);
		cv::dilate(temp, img, element);

		radius = 2;
		element = cv::getStructuringElement( cv::MORPH_RECT, cv::Size( 2*radius + 1, 2*radius+1 ), cv::Point( radius, radius ) );
		//opening
		cv::erode(temp, temp, element);
		cv::dilate(temp, img, element);

		//closing
		cv::dilate(img, temp, element);
		cv::erode(temp, temp, element);
	}

	/*
	* Clear all statistic and probabilities
	*/
	void clear_statistic()
	{
		for(int i=0; i<num_of_bins; i++)
		{
			for(int j=0; j<num_of_bins; j++)
			{
				color_Hist[i][j] = 0;
				skin_Hist[i][j] = 0;
				probab[i][j] = 0;
			}
		}
		num_color = 0;
		num_skin = 0;
		cout << "All probabilities and statistics are set to 0." << endl;

	}

	void adaptive(cv::Mat face)
	{	
		cv::Mat fs(face.size(), CV_8UC1);
		this->detectSkin(face, fs);
		this->connectPotential(fs);
		cv::Mat uv; 
		this->BGR2UV(face, uv);
		// cv::imshow("Face", face);
		// cv::imshow("Face skin", fs);
		// iterate each pixel
		for(int r=0; r<fs.rows; r++)
		{
			for(int c=0; c<fs.cols; c++)
			{
				int skn = (int)(fs.at<uchar>(r, c));
				if(skn > 0)//skin color or potential skin color
				{	
					//get related UV value of this position
					cv::Vec2b intensity = uv.at<cv::Vec2b>( r, c );
					int u = intensity.val[0];
					int v = intensity.val[1];
					skin_Hist[u][v]++;
					color_Hist[u][v]++;
					num_color++;
					num_skin++; 
				}
			}
		}
		//update probability
		this->calculProbab();
	}

	//n-fold cross-validation for testing skin detector
	//total 78 skin images, about 15 images each fold
	void testSkinDetector(string dataset_folder = "../dataset/skin/", int num_fold = 5)
	{
		cout << '\n' << '\n';
		cout << "cross-validation for skin detector on images in " << dataset_folder << endl;
		this->clear_statistic();
		this->Tmax = 0.5;
		this->Tmin = 0.10;
		cout << "Thresholds between: " << Tmax << "----" << Tmin << endl;
		//read all file paths
		vector<string> train_files;
		DIR *dir;
		struct dirent *entry;
		dir = opendir( dataset_folder.c_str() );
		if(dir)
		{
			while( entry = readdir(dir) )
			{
				if( strcmp( entry->d_name, "." ) != 0 && strcmp( entry->d_name, ".." ) != 0 )
				{
					std::string temp(entry->d_name);
					//concact train image and truth image file path
					std::string train_file_path = dataset_folder + temp;
					std::string truth_file_path = dataset_folder + "GroundTruth/" + temp.replace( (temp.length()-3),3, "png");
					//read image file to Mat
					cv::Mat train_image = cv::imread( train_file_path, CV_LOAD_IMAGE_COLOR);//8UC3 BGR format image
					cv::Mat truth_image = cv::imread( truth_file_path, CV_LOAD_IMAGE_GRAYSCALE);//8UC1 one intensity image
					//check both train image and truth image are loaded successfully
					if( ! train_image.data )
					{
						std::cout << "Could not read or find train image: " << train_file_path << std::endl;
						continue;
					}
					if( ! truth_image.data )
					{
						std::cout << "Could not read or find truth image: " << truth_file_path << std::endl;
						continue;
					}
					train_files.push_back(temp);
				}
			}
			closedir(dir);
		} 
		else
		{
			std::cout << "open dataset folder failed, path: " << dataset_folder << std::endl;
		}
		cout << "Number of images files " << train_files.size() << endl;

		//randomizing the vector of file paths
 	 	std::random_shuffle ( train_files.begin(), train_files.end() );

 	 	//split vector into train files and test files in each iteration
 	 	int fold_size = train_files.size() / num_fold;
 	 	float ave_accuracy = 0;
 	 	float ave_recall = 0;
 	 	float ave_precision = 0;
 	 	float F1 = 0;
 	 	for(int i=0; i<num_fold; i++)
 	 	{
 	 		vector<string>::iterator first = train_files.begin() + i * fold_size;
			vector<string>::iterator last = first + fold_size;
			vector<string> test_files(first, last);

			//then train
			cout << "Training: ";
			int count =0;
			for(vector<string>::iterator it=train_files.begin(); it!=train_files.end(); ++it)
			{
			    if(it>=first && it<last)	continue;
				string train_file_path = dataset_folder + (*it).replace( ((*it).length()-3), 3, "jpg" );
				string truth_file_path = dataset_folder + + "GroundTruth/" + (*it).replace( ((*it).length()-3), 3, "png" );
				// cout << train_file_path << endl;
				// cout << truth_file_path << endl;
				// cout << '\n';
				//read image file to Mat
				cv::Mat train_image = cv::imread( train_file_path, CV_LOAD_IMAGE_COLOR);//8UC3 BGR format image
				cv::Mat truth_image = cv::imread( truth_file_path, CV_LOAD_IMAGE_GRAYSCALE);//8UC1 one intensity image
				if( (!train_image.data) || (!truth_image.data) ) continue;
				this->updateStatistic(train_image, truth_image);
				count++;
			}
			this->calculProbab();
			cout << "on " << count << " image files"<< endl;
			
			//then testing			
			cout << "Testing: ";
			int TP = 0; int FN = 0;
			int FP = 0; int TN = 0;	
			int test_count = 0;
			for( vector<string>::iterator it=test_files.begin(); it!=test_files.end(); ++it )
			{
				string train_file_path = dataset_folder + (*it).replace( ((*it).length()-3), 3, "jpg" );
				string truth_file_path = dataset_folder + + "GroundTruth/" + (*it).replace( ((*it).length()-3), 3, "png" );
				cv::Mat test_image = cv::imread( train_file_path, CV_LOAD_IMAGE_COLOR);//8UC3 BGR format image
				cv::Mat truth_image = cv::imread( truth_file_path, CV_LOAD_IMAGE_GRAYSCALE);//8UC1 one intensity image
				if( (!test_image.data) || (!truth_image.data) ) continue;
				cv::Mat dst( test_image.rows, test_image.cols, CV_8UC1);
				//detect skin color pixels
				this->detectSkin(test_image, dst);
				this->connectPotential(dst);
				//compare with truth image
				for(int i=0; i < truth_image.rows; i++)
				{
					for( int j=0; j < truth_image.cols; j++)
					{
						int truth = (int)truth_image.at<uchar>(i, j);
						int pred = (int)dst.at<uchar>(i, j);
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
				test_count++;
			}
			cout << "on " << test_count << "truth images" << endl;
			
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
			ave_accuracy += accuracy;
			ave_recall += recall;
			ave_precision += precision;
			F1 += f1;
 	 	}
 	 	//average evaluation of 5-fold
 	 	ave_accuracy /= num_fold;
 	 	ave_recall /= num_fold;
 	 	ave_precision /= num_fold;
 	 	F1 /= num_fold;
 	 	cout << "Average accuracy: " << ave_accuracy << "\n"
 	 		<< "Average recall: " << ave_recall << "\n"
 	 		<< "Average precision: " << ave_precision << "\n"
 	 		<< "Average F1 feature: " << F1 <<endl;
	}
};
