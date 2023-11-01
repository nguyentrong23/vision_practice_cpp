#pragma once
#include "GeoModel.h"

#include <chrono>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;

#define MAX_NUM_MODEL (int)10
#define OBJECT_ROI_BORDER_OFFSET (int)2
#define SOURCE_THERSHOLD_TOLERANCE  (double)0.05


struct  RotatedObj {
    Mat image;
    Point2f centerCrop;
    Point2f centerMaxScores;
    double angle;
};

struct PossibleObject
{
    // angle find by pca
    double pcaAngle;
    // center find by pca
    Point2f pcaCenter;
    // contours bouding rectangle
    Rect conBoundingRect;
    // contours min Rectange area
    RotatedRect conMinRectArea;
    // list model need check
    vector<int> modelCheckList;
    // rotated image with neative offset
     RotatedObj rotNegative;
    // rotated image with positive offset
     RotatedObj rotPositive;
    // rotated image with positive offset reserve 180 deg
     RotatedObj rotNegative_reverse;
    // rotated image with positive offset reserve 180 deg
     RotatedObj rotPositive_reverse;
};

struct MatchedObj
{
    std::string name;
    double scores;
    int indexOfSample;
    Point2f coordinates;
    double angle;
    RotatedRect pickingBox;
    Mat image;
    Rect rect;
};

class GeoMatch
{
public:
    GeoMatch();
    ~GeoMatch();

    bool addGeoMatchModel(String pathTemplate, String modelName);
    void clearMatchModel(void);
    void removeMatchModel(int index);
    void removeMatchModel(int startIndex, int endIndex);
    void modifyMatchModelAt(int index, GeoModel model);
    vector<GeoModel> getModelSource();
    int getModelSourceSize();
    void setImageSource(string path);
    void setImageSource(Mat img);
    bool haveObjectsInPlate();

    void matching(Mat image, bool boudingBoxChecking);
    void matching(bool boudingBoxChecking);

    void getRotatedROI(Mat& matSrc,  RotatedObj& object, Point center, RotatedRect minRectArea);
    bool matchingScores( RotatedObj& objectRotated, GeoModel& model, double& lastMaxScores);
    void saveMatchedObjectInfo(MatchedObj& matched,  RotatedObj& objectRotated, GeoModel model, double scores, Point2f pcaCenter);

    static Mat cropImageWithBorderOffset(Mat sourceImage, Rect boxBounding, int border);
    static void drawPickingBox(Mat& matSrc, RotatedRect rectRot, Scalar color);

    Mat resultImage;
private:
    // model source for matching
    vector<GeoModel> ModelSrc;
    // matched object list after matching
    vector<MatchedObj> matchedList;
    // image source for matching
    Mat imageSource;
    // picking box size
    Size2f pickingBoxSize;
    // flag indicator has any possible object in plate
    bool objectsInPlate;
    // last matching execution time (in millisecond)
    double lastExecuteTime = 0.0;
    // lower object threshold ratio
    const double lowerThreshRatio = 1.0 - SOURCE_THERSHOLD_TOLERANCE;
    // upper object threshold ratio
    const double upperThreshRatio = 1.0 + SOURCE_THERSHOLD_TOLERANCE;
};




