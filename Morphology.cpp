/**
	Morphology operation on Gray Image
	@author Lei Liu
*/

#include <iostream>

#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;

class Morphology
{

public:
	
	Morphology(){}
	
	void erode(cv::Mat &src, cv::Mat &dst, int radius=1)
	{
		dst = src;
		if( !src.data || !dst.data)
		{
			cout << "illegal src or dst parameter" << endl;
			return;
		}
		
		int width = src.rows;
		int height = src.cols;
		
		for(int i=0; i<height; i++)
		{
			for(int j=0; j<width; j++)
			{ 
				int min = (int)src.at<uchar>(i, j);
				for(int m=i-radius; m<=i+radius; m++)
				{
					if( m >= height) continue;
					
					for(int n=j-radius; n<=j+radius; n++)
					{
						if( n >= width ) continue;
						
						if( (int)src.at<uchar>(m, n) < min )
							min = (int)src.at<uchar>(m, n);
					}
				}
				dst.at<uchar>(i, j) = (uchar)min;
			}	
		}
	}
	
	void dilate(cv::Mat &src, cv::Mat &dst, int radius=1 )
	{
		dst = src;
		if( !src.data || !dst.data)
		{
			cout << "illegal src or dst parameter" << endl;
			return;
		}
		
		int width = src.rows;
		int height = src.cols;
		
		for(int i=0; i<height; i++)
		{
			for(int j=0; j<width; j++)
			{
				int max = (int)src.at<uchar>(i, j);
				for(int m=i-radius; m<=i+radius; m++)
				{
					if( m >= height) continue;
					
					for(int n=j-radius; n<=j+radius; n++)
					{
						if( n >= width ) continue;
						
						if( (int)src.at<uchar>(m, n) > max )
							max = (int)src.at<uchar>(m, n);
					}
				}
				dst.at<uchar>(i, j) = (uchar)max;
			}	
		}
	}
	
};


