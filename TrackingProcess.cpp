/** 
	TrackingProcess.cpp
	Process of tracking objects over time with detected images

	@author Lei Liu
*/

#include <iostream>
#include <vector>
#include <string>
#include <math.h>
#include <list>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "Blobs.cpp"

using namespace std;

class TrackingProcess
{
	
private:
	map<int, cv::RotatedRect> history; //ellipse in the history at time t-1
	list<int> hypoth_labels; //mark all labels of current hypothesis, unassigned ellipse
	int newLabel = 1;

	map<int, cv::RotatedRect> pre_hitstory; //ellipses in the history at time t-2

	ostringstream trajectory; 

public:

	TrackingProcess()
	{
		history.clear();
		pre_hitstory.clear();
	}

	map<int, cv::RotatedRect> getHistory()
	{
		return this->history;
	}

	void firstFrame(cv::Mat &bwimg, float minBlobSize, string filename = "")
	{
		//get blobs of binary image from the first frame
		Blobs blobs(bwimg);
		blobs.setSize(minBlobSize);
		map< int, vector< pair<int, int> > > blob = blobs.getBlobs();
		if( filename != "" ) blobs.write(filename);
		//for each blob, fit an ellipse
		for(map< int, vector< pair<int, int> > >::iterator b=blob.begin(); b!=blob.end(); ++b)
			this->newHypothesis( b->second );
		pre_hitstory = history;
	}

	void newHypothesis(vector< pair<int, int> > &blob)
	{
		//fit a new Ellipse for this blob
		cv::RotatedRect ellipse = this->fitEllipse( blob );
		//add new Ellipse to the history of ellispses
		(this->history).insert( make_pair((this->newLabel), ellipse) );
		(this->newLabel)++; //increase once a new Ellipse is set
	}

	void assignHypothesis(vector< pair<int, int> > &blob, int el_label)
	{
		//fit a new Ellipse for this blob
		cv::RotatedRect ellipse = this->fitEllipse( blob );
		(this->history).at(el_label) = ellipse; //associate label to updated ellipse
		(this->hypoth_labels).remove( el_label );
		// cout << "assign hypothesis: " << el_label << endl;
	}

	/**
		fit an ellipse for a blob
		using covariance matrix of the distribution of position of points within the blob
		The major axes is the largest eigenvalue multipled by a factor (chi-square value represents the confidence level)
		@param blob, The vector contains the positions of points. Each iteration is a pair of row and column index of a point
		@return the Ellipse object as type cv::RotatedRect
	 */
	cv::RotatedRect fitEllipse( vector< pair<int, int> > &blob, float chisquare_val = 16	)
	{
		//initialize distribution of points
		int num = blob.size();
		float arr[num][2];
		for(int i=0; i<num; i++)
		{
			arr[i][1] = blob[i].first; //row index, y
			arr[i][0] = blob[i].second; //column index, x
		}

		//calculate covariance matrix
		cv::Mat mat = cv::Mat(num, 2, CV_32FC1, &arr);
		cv::Mat cov, mean;
		cv::calcCovarMatrix( mat, cov, mean, CV_COVAR_NORMAL | CV_COVAR_ROWS);
		cov = cov / (mat.rows - 1);
		//calculate eigen problem
		cv::Mat eigVec = cv::Mat(2, 2, CV_32FC1);
		cv::Mat eigVal = cv::Mat(1, 2, CV_32FC1);
		cv::eigen( cov, eigVal, eigVec);
		// cout << eigVec << endl;
		//calculate the major and minor axes
		float alpha = sqrt( eigVal.at<double>(0,0) ) * sqrt(chisquare_val);
		float beta = sqrt( eigVal.at<double>(0,1) ) * sqrt(chisquare_val);
		//calcualte the angle.
		double angle = atan2( eigVec.at<double>(0,1), eigVec.at<double>(0,0) ) * 180 / M_PI;
		// if( cov.at<double>(0,0) < cov.at<double>(1,1) )
		// {
		// 	float t = alpha;
		// 	alpha = beta;
		// 	beta = t;
		// 	angle = angle + 90;
		// }
		return cv::RotatedRect( cv::Point2f(mean.at<double>(0,0), mean.at<double>(0,1)), 
			cv::Size2f(alpha, beta), angle);
	}

	//calculate the distance of a point from an ellipse
	double distance(cv::RotatedRect &ellipse, cv::Point &point)
	{
		double center_x = ellipse.center.x;
		double center_y = ellipse.center.y;
		double alpha = ellipse.size.width / 2; //semi-major axes
		double beta = ellipse.size.height / 2; //semi-minor axes
		double angle = ellipse.angle * M_PI / 180; //rotation angle
		double p_x = point.x;
		double p_y = point.y;

		double new_x = (p_x - center_x) * cos(angle) + (p_y - center_y) * sin(angle);
		double new_y = (p_x - center_x) * (-sin(angle)) + (p_y - center_y) * cos(angle);
		double result = (pow(new_x, 2.0) / pow(alpha, 2.0)) + (pow( new_y, 2.0) / pow(beta, 2.0));		
		return result;
	}

	vector< pair<int, int> > maxContour(vector< pair<int, int> > &blob)
	{
		map< int, pair<int, int> > outContour;
		int min_x = 10000;
		int max_x = 0;
		for(int i=0; i<blob.size(); i++)
		{
			int x = blob[i].first;
			int y = blob[i].second;
			if(x<min_x)
				min_x = x;
			if(x>max_x)
				max_x = x;

			if(outContour.find(x) == outContour.end()) //new contour points
				outContour.insert( make_pair( x, make_pair(y, y) ) );
			else
			{
				int min_y = outContour.at(x).first;
				int max_y = outContour.at(x).second;
				if( y <= min_y)
					min_y = y;
				if( y >= max_y)
					max_y = y;
				outContour.at(x).first = min_y;
				outContour.at(x).second = max_y;
			}
		}

		vector< pair<int, int> > contour;
		for(map< int, pair<int, int> >::iterator out=outContour.begin(); out!=outContour.end(); ++out)
		{
			int x = out->first;
			pair<int, int> yy = out->second;
			int y1 = yy.first;
			int y2 = yy.second;
			contour.push_back( make_pair(x, y1) );
			contour.push_back( make_pair(x, y2) );
		}

		//add upper boundary
		for(int i=outContour.at(min_x).first; i<outContour.at(min_x).second; ++i)
			contour.push_back( make_pair(min_x, i) );
		for(int i=outContour.at(max_x).first; i<outContour.at(max_x).second; ++i)
			contour.push_back( make_pair(max_x, i) );

		return contour;
	}

	cv::RotatedRect contourEllipse(vector< pair<int, int> > &contour)
	{
		vector< cv::Point > cont;
		for(int i=0; i<contour.size(); i++)
		{
			int x = contour[i].first;
			int y = contour[i].second;
			cont.push_back( cv::Point( y, x ) );
		}
		cv::RotatedRect ellipse = cv::fitEllipse( cv::Mat(cont) );
		return ellipse;
	}

	void showEllipses(cv::Mat &original, string framename = "")
	{
		for(map<int, cv::RotatedRect>::iterator hist=(this->history).begin(); hist!=(this->history).end(); ++hist)
		{
			try
			{
				//draw an ellipse on the original image
				cv::ellipse(original, (hist->second), cv::Scalar( 0, 0, 255 ), 2, 8);
				//write corrresponding text
				stringstream text;
				text << (hist->first);
				cv::putText(original, text.str(), (hist->second).center, CV_FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(100,100,100), 2);
				if(framename != "")
				{

					trajectory << framename << " " << (hist->first) << " " << (hist->second).center << (hist->second).size << " " << (hist->second).angle << " " << "\n";
				}
			}
			catch (const std::exception& e) 
			{
				cout << "Fail to draw ellipse " << endl;
			}
			
		}
	}

	void writeTrajectory(string filename)
	{
		// cout << trajectory.str() << endl;
		ofstream file;
		file.open( filename.c_str(), ios::trunc ); //open file and deleted existed data
		if(file)
			file << trajectory.str();
		file.close();
	}

	//start from the second frame
	void track( cv::Mat &current_bwimg, float minBlobSize, string filename="" )
	{	
		//predict the ellipses before compare with new image
		this->predict();

		//find all labels of current ellipses
		this->hypothLabels();

		//get all the blobs of current binary image
		Blobs blobs(current_bwimg);
		blobs.setSize(minBlobSize);
		map<int, vector< pair<int, int> > > blob = blobs.getBlobs();
		if( filename != "" ) blobs.write(filename);

		map<int, vector<int>> blob_els;//map blob with all intersected ellipses

		//find all the intersection between blob with every ellipses
		this->findIntersection(blob, blob_els); //analyse the intersection between blobs and ellipses

		//assign ellipse for matched blob ans ellipses
		this->assign( blob, blob_els );

		//remove useless hypothesis, unassigned ellipses
		this->removeHypothe();
	}

	void findIntersection(map< int, vector< pair<int, int> > > &blob, map<int, vector<int>> &blob_els)
	{
		map< int, pair<int,int> > el_interesction; // map< ellipse label, pair<blob label, number of inter points> >
		for(map< int, vector< pair<int, int> > >::iterator b=blob.begin(); b!=blob.end(); ++b)
		{
			//get all the points of this blob
			vector< pair<int, int> > points = b->second;
			//get the label of this blob
			int bl_label = b->first;		
			vector<int> els; //mark ellipses that has intersection with this blob

			//compare with each ellipse
			for( map<int, cv::RotatedRect>::iterator hist=(this->history).begin(); hist!=(this->history).end(); ++hist)
			{
				int el_label = hist->first;
				cv::RotatedRect ellipse = hist->second;
				int num_inters=0; //number of points that within this ellipse
				//compare every point in this blob with this ellipse
				for(int i=0; i<points.size(); i++)
				{
					cv::Point p = cv::Point( points[i].second, points[i].first ); //point(column, row)		
					double distan = this->distance(ellipse, p);
					if( distan <= 1.0 ) //this blob has intersection with this ellipse
						num_inters++;
				}
				//blob has intersection with this ellipse
				if(num_inters != 0)
				{
					//check whether current ellipse has alrady intersection with other blobs
					if( el_interesction.find(el_label) != el_interesction.end() )
					{
						pair<int, int> bl_num = el_interesction.at(el_label);
						if( num_inters > bl_num.second )
						{
							//delete this ellipse from previous intersected blob
							vector<int> prevElls = blob_els.at( bl_num.first );
							for(int i=0; i<prevElls.size(); i++ )
							{
								if(el_label == prevElls[i])
									prevElls.erase(prevElls.begin()+i);
							}
							blob_els.at( bl_num.first ) = prevElls;
							//add current pair of intersection
							el_interesction.at(el_label) = make_pair( bl_label, num_inters );
							els.push_back(el_label);
						}
					}
					else
					{
						el_interesction.insert( make_pair( el_label, make_pair(bl_label, num_inters) ) );
						els.push_back(el_label);
					}
				}
			}
			blob_els.insert( make_pair( bl_label, els ) );
		}
	}

	void assign(map< int, vector< pair<int, int> > > &blob, map<int, vector<int>> &blob_els)
	{
		//for each blob, assign ellipses
		for(map<int, vector<int>>::iterator bl_els=blob_els.begin(); bl_els!=blob_els.end(); ++bl_els)
		{
			int bl_label = bl_els->first; //blob label
			vector<int> els = bl_els->second; //ellipse labels
			vector< pair<int, int> > points = blob.at(bl_label); //all points of this ellipse

			if(els.size()==0) //generate new ellipse and assign
				this->newHypothesis(points);
			else if( els.size() == 1 ) //assign the ellipse to this blob
				this->assignHypothesis( points, els.at(0) );//assign ellipse lable with current blob
			else //assign multiple ellipses to this blob
			{
				map<int, vector<pair<int, int>>> bl_points; //assign points to every ellipse
				for(int e=0; e<els.size(); e++)
					bl_points.insert( make_pair( els[e], vector<pair<int,int>>() ) );

				for(int i=0; i<points.size(); i++) //each point as pair<row, column>
				{
					cv::Point pnt = cv::Point( points[i].second, points[i].first );
					double minDist = 1.0/0.0;
					int minEll = -1;
					bool isAssigned = false;
					for(int e=0; e<els.size(); e++)
					{
						int el = els[e];
						double dist = this->distance( (this->history).at(el), pnt );
						if(dist<=1.0)
						{
							( bl_points.at(el) ).push_back( points[i] );
							isAssigned = true;
						}
						else
						{
							if( dist < minDist)
							{
								minDist = dist;
								minEll = el;
							}
						}
					}
					if( isAssigned == false )
						(bl_points.at(minEll)).push_back( points[i] );
				}
				//associate ellipse with points of this blob
				this->associateMultiEllp(bl_points);
			}
		}
	}

	void associateMultiEllp(map<int, vector<pair<int, int>>> &bl_points)
	{
		for(map<int, vector<pair<int, int>>>::iterator bl_pnt=bl_points.begin(); 
			bl_pnt!=bl_points.end(); ++bl_pnt)
		{
			this->assignHypothesis( bl_pnt->second, bl_pnt->first );
		}
	}

	void hypothLabels()
	{
		(this->hypoth_labels).clear();
		for( map<int, cv::RotatedRect>::iterator hist=(this->history).begin(); hist!=(this->history).end(); ++hist)
			(this->hypoth_labels).push_back( hist->first );
	}

	void removeHypothe()
	{
		for (list<int>::iterator it = (this->hypoth_labels).begin(); it != (this->hypoth_labels).end(); it++)
			(this->history).erase( *it );
	}

	void predict(float time_lag=1)
	{
		map<int, cv::RotatedRect> temp_history;
		for(map<int, cv::RotatedRect>::iterator hist = (this->history).begin(); hist!=(this->history).end(); ++hist)
		{
			int el_label = hist->first;
			cv::RotatedRect el = hist->second;
			cv::RotatedRect temp_el;
			if( (this->pre_hitstory).find(el_label) != (this->pre_hitstory).end() ) //find ellipse in previous history with same label, predict new location
			{
				cv::RotatedRect pre_el = (this->pre_hitstory).at(el_label);
				float new_x = 2 * (el.center.x) - (pre_el.center.x);
				float new_y = 2 * (el.center.y) - (pre_el.center.y);
				temp_el = cv::RotatedRect( cv::Point2f(new_x, new_y ), el.size, el.angle );
			}
			else //no prediction. Add to temp history			
				temp_el = el;

			temp_history.insert( make_pair(el_label, temp_el) );
		}
		this->pre_hitstory = this->history;
		this->history = temp_history;
	}
	
};
