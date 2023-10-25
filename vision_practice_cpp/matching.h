#pragma once
#include "model.h"

using namespace cv;
using namespace std;

#define MAX_NUM_MODEL (int)10
#define OBJECT_ROI_BORDER_OFFSET (int)2

struct RotatedObject {
	Mat image;
	Point2f center;
	double angle;
};

struct ObjectInfo
{
	// angle find by pca
	double pcaAngle;
	// center find by pca
	Point2f pcaCenter;
	// contours index of contours source
	int conIndex;
	// contours area;
	int conArea;
	// contours center
	Point2f conCenter;
	// contours bouding rectangle
	Rect conBoundingRect;
	// contours min Rectange area
	RotatedRect conMinRectArea;
	// list model need check
	vector<int> modelCheckList;
	// rotated image with neative offset
	RotatedObject rotNegative;
	// rotated image with positive offset
	RotatedObject rotPositive;
	// rotated image with positive offset reserve 180 deg
	RotatedObject rotNegative_reverse;
	// rotated image with positive offset reserve 180 deg
	RotatedObject rotPositive_reverse;
};

enum class GetROI_MODE : int {
	ROI_POSITIVE,
	ROI_NEGATIVE,
	ROI_POSITIVE_REVERSE,
	ROI_NEGATIVE_REVERSE
};

struct MatchedObjects
{
	Point coordinates;
	double angle;
	Mat image;
};

class matching
{
public:
	matching();
	~matching();

	bool addmatchingModel(String pathTemplate, String modelName);
	bool addmatchingModel(String pathTemplate, String modelName, int pyrDownLevel);
	void clearMatchModel(void);
	void removeMatchModel(int index);
	void removeMatchModel(int startIndex, int endIndex);
	int GetModelSrcSize(void);

	void sourceStream(String path);
	void sourceStream(String path, int pyrDownLevel);
	void Matching(void);

	void getPcaOrientation(const vector<Point>& pts, double& angleOutput, Point2f& centerOutput);
	void getRotatedROI(Mat& matSrc, GetROI_MODE roiMode, model& model, ObjectInfo& object);
	void getRotatedROI(Mat& matSr, model& model, ObjectInfo& object, GetROI_MODE roiMode);
	bool matchingScores(RotatedObject& objectRotated, model& model, double& lastMaxScores);
	bool matchingScores(Mat& inputImage, model& model, double& lastMaxScores);
	Mat cropImageWithBorderOffset(Mat sourceImage, Rect boxBounding, int border);
private:
	vector<model> ModelSrc;
	Mat imageSrc;
	double lastCycleTime = 0;
};

