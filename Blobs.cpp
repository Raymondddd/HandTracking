/**
	Blobs.cpp
 	produce current skin areas and corresdonding ellipses from current bwimg
 	@author Lei Liu
*/

#include <iostream>
#include <vector>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;

class Blobs
{
private:
	int numOfRuns;
	vector<int> stRun; //column position of start point of a connected pixels in a row
	vector<int> enRun; //column position of end point of a connected pixels in a row
	vector<int> rowRun; //row number of a set of connected pixels
	vector<int> runLabels; //label for a connected components
	int offset; //// 0- 4neighbourhoods, 1- 8neighoubrhoods
	cv::Mat bwimg;
	float minSize;
public:

	Blobs(cv::Mat &currentBwimg, int neighbourType=1, float sizeParam=0.005){
		numOfRuns = 0;
		stRun.clear();
		enRun.clear();
		rowRun.clear();
		runLabels.clear();
		offset = neighbourType;
		bwimg = cv::Mat(currentBwimg);
		minSize = bwimg.rows * bwimg.cols * sizeParam;
		this->connectedComponentLabelling();		
	}

	//fast connected components labelling algorithm
	// 8-neighbourhoods connnected-components labeling
	void connectedComponentLabelling()
	{
		//fill RUN vectors
		for(int i=0; i< bwimg.rows; i++)
		{
			//one row of image pixels
			const uchar* rowData = bwimg.ptr<uchar>(i);
			//first element
			if(rowData[0] == 255)
			{
				numOfRuns++;
				stRun.push_back(0);
				rowRun.push_back(i);
			}
			for(int j=1; j<bwimg.cols; j++)
			{
				if( rowData[j-1] ==0 && rowData[j] ==255 )
				{
					 // 4-neighbourhood ++;
					numOfRuns++;
					stRun.push_back(j);
					rowRun.push_back(i);
				}
				else if( rowData[j-1] == 255 && rowData[j] == 0 )
				{
					enRun.push_back(j-1);
				}
			}
			if( rowData[bwimg.cols-1] == 255)
			{
				enRun.push_back(bwimg.cols-1);
			}
		}

		//assign label for each RUNs and generate Equivalence List
		// vector<int> runLabels;
		vector< pair<int, int> > equivalences;
		runLabels.assign(numOfRuns, 0);
		int idxLabel = 1;
		int curRowIdx = 0;
		int firstRunOnCur = 0;
		int firstRunOnPre = 0;
		int lastRunOnPre = -1; 
		for(int i=0; i<numOfRuns; i++)
		{
			//first run in current row, update first run and last run in previous row
			if(rowRun[i] != curRowIdx)
			{
				curRowIdx = rowRun[i];
				firstRunOnPre = firstRunOnCur;
				lastRunOnPre = i-1;
				firstRunOnCur = i;
			}
			//for every run in last row, find overlay area with current row
			for(int j=firstRunOnPre; j<=lastRunOnPre; j++)
			{
				//has overlay area(connected) within neighbour rows
				if( stRun[i] <= enRun[j] + offset && 
					enRun[i] >= stRun[j] - offset &&
					rowRun[i] == rowRun[j] + 1 )
				{
					if(runLabels[i] == 0) //no labelled
					{
						runLabels[i] = runLabels[j];
					}
					else if( runLabels[i] != runLabels[j] ) //has labeled
					{
						equivalences.push_back( make_pair( runLabels[i], runLabels[j] ) );
					}
				}
			}
			if( runLabels[i] == 0 ) //no overlay with previous row
			{
				runLabels[i] = idxLabel++;
			}
		}

			if(runLabels.size() == 0)
			return;
		//handle equivalences and replace same label, find connected runs and assign same label
		//using deep-first searching for each graph(equivalences as tree)
		int maxLabel = *max_element( runLabels.begin(), runLabels.end() );
		vector< vector<bool> > eqTab(maxLabel, vector<bool>(maxLabel, false));
		vector< pair<int, int> >::iterator vecPairIt = equivalences.begin();
		//A Table representing the equivalence labels
		while(vecPairIt != equivalences.end())
		{
			eqTab[vecPairIt->first - 1][vecPairIt->second - 1] = true;
			eqTab[vecPairIt->second - 1][vecPairIt->first - 1] = true;
			vecPairIt++;
		}
		vector<int> labelFlag(maxLabel, 0);
		vector< vector<int> > equalList;
		vector<int> tempList;
		for(int i=1; i<=maxLabel; i++)
		{
			if(labelFlag[i-1])
			{
				continue;
			}
			labelFlag[i-1] = equalList.size() + 1;
			tempList.push_back(i);
			for( vector<int>::size_type j=0; j<tempList.size(); j++ )
			{
				for( vector<bool>::size_type k=0; k!= eqTab[tempList[j] -1 ].size(); k++ )
				{
					if( eqTab[ tempList[j]-1 ][k] && !labelFlag[k] )
					{
						tempList.push_back(k+1);
						labelFlag[k] = equalList.size() + 1;
					}
				}
			}
			equalList.push_back(tempList);
			tempList.clear();
		}
		for(vector<int>::size_type i=0; i != runLabels.size(); i++)
		{
			runLabels[i] = labelFlag[runLabels[i] - 1];
		}
	}

	//get all the points of each blob for a binary image
	// int- blob index
	// vector< pair<int,int> > - points of a blob
	map< int, vector< pair<int, int> > > getBlobs()
	{
		//blobs: map of all bolb, for each bolb
		//key is the label, value is the corresponding points
		map< int, vector< pair<int, int> > > blobs; 
		if(runLabels.size() != 0)
		{
			//generate blobs
			
			//for each bolb, points are stored in an vector containing all pairs of row index and column index
			for(int i=0; i<numOfRuns; i++)
			{
				vector< pair<int, int> > pointsInRow; 
				int rowIdx = rowRun[i];
				int stIdx = stRun[i];
				int enIdx = enRun[i];
				for(int idx= stIdx; idx <= enIdx; idx++ )
				{
					pointsInRow.push_back( make_pair(rowIdx, idx) );
				}
				int label = runLabels[i];
				vector< pair<int, int> > tempBlob;
				if( blobs.find(label) != blobs.end() ) //existed blob, push current point
				{
					tempBlob = blobs.at(label);
					tempBlob.insert( tempBlob.end(), pointsInRow.begin(), pointsInRow.end() );
					blobs.at(label) = tempBlob;
				}
				else //new label, new blob				
				{
					tempBlob = pointsInRow;
					blobs.insert( make_pair(label, tempBlob) );
				}		
			}
			this->blobFilter(blobs);
		}
		// cout << "getBlobs num of Blobs: " << blobs.size() << endl;
		// cv::imshow("Blobs undert tracking", bwimg);
		return blobs;
	}

	void blobFilter(map< int, vector< pair<int, int> > > &blobs)
	{
		//size filtering on blobs
		for(map< int, vector< pair<int, int> > >::iterator bl=blobs.begin(); bl!=blobs.end(); ++bl)
		{
			if( (bl->second).size() < (this->minSize) )
			{
				for(int i=0; i<(bl->second).size(); i++)
				{
					int r = ((bl->second).at(i)).first;
					int c = ((bl->second).at(i)).second;
					bwimg.at<uchar>(r,c) = (uchar)0;
				}
				blobs.erase(bl);
			}
			else
			{
				for(int i=0; i<(bl->second).size(); i++)
				{
					int r = ((bl->second).at(i)).first;
					int c = ((bl->second).at(i)).second;
					bwimg.at<uchar>(r,c) = (uchar) ( (bl->first) * 50 +50 );
				}
			}
		}
	}

	void setSize(float size)
	{
		if(size > minSize)
			this->minSize = size;
	}

	void write(string filename)
	{
		cv::imwrite(filename, this->bwimg);
	}
};