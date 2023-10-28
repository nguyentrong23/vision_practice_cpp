#pragma once
#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <opencv2/highgui.hpp>

#include <math.h>
#include <iostream>

using namespace cv;
using namespace std;

#define D2R (CV_PI / 180.0)
#define R2D (180.0 / CV_PI)

#define INIT_CANNY_THRESH_1 (int)100
#define INIT_CANNY_THRESH_2 (int)200
#define INIT_CANNY_KERNEL_SIZE (int)3
#define INIT_MIN_SCORES (double)0.9
#define INIT_GREEDINESS (double)0.0
#define INIT_CONTOURS_TOLERANCE (int)1000
#define INIT_PCA_DIRETION_VECTOR_SCALE (double)0.02
class GeometricModel
{
public:
	struct ModelPatternInfo
	{
		Point2d Coordinates;
		Point2d Derivative;
		double Angle;
		double Magnitude;
		Point2d Center;
		Point2d Offset;
	};

	GeometricModel();
	GeometricModel(String pathTemplate, String modelName);
	GeometricModel(String pathTemplate, String modelName, int pyramidDownLevel);
	~GeometricModel();
	void InitParameter(void);
	bool readImage(String modelPath);
	bool readImage(String modelPath, int pyrDownLevel);
	bool learnPattern(bool bShowImage);
	void getPcaOrientation(const vector<Point>& pts, double& angleOutput, Point2f& centerOutput);
	void showInfoImage(String windowName);
	void drawPcaAxis(Mat& img, Point p, Point q, Scalar colour, const float scale = 0.2);

	bool imageEmpty(void);
	bool geometricLearned(void);

	void getThreshArea(int& lowThresh, int& highThresh);
	double getPcaAngle(void);

	// pattern infomation container
	vector<ModelPatternInfo> patternInfo;
	// matching min scores
	double scoresMin;
	// matching greediness scores
	double greediness;
	// center point find by PCA
	Point2f centerPCA;
	// image Height;
	int height;
	// image Width;
	int width;
	// model name
	String nameOfModel;
	// original template pattern image
	Mat imageOriginal;
	// distance of center pattern to center pca
	Point2f cPattern2cPca;
private:
	//// image of model in grayscale
	//Mat imageGray;
	//// edges image of model
	//Mat imageEdges;
	// canny threshold 1 parameter
	double cannyThresh_1;
	// canny threshold 2 parameter
	double cannyThresh_2;
	// canny kernel size
	int cannyKernelSize;

	// contours container
	vector<vector<Point>> contours;
	// contours hierarchy
	vector<Vec4i> hierarchy;
	// object select area
	int contoursSelectIndex;
	// object select area
	int contoursSelectArea;
	// object outer area tolerance
	int contoursAreaTolerance;
	// object angle find by PCA
	double originalPcaAngle;
	// pca angle offset in degrees unit
	double offsetPcaAngle;
	// pca direction draw point
	Point point1pca;
	// pca direction draw point
	Point point2pca;
	// Is model learned pattern
	bool bIsGeometricLearn;
};

