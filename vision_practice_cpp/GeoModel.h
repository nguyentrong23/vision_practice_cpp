#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <math.h>

#define D2R (CV_PI / 180.0)
#define R2D (180.0 / CV_PI)

#define CANNY_RANGE_MIN 0
#define CANNY_RANGE_MAX 100
#define CANNY_CONTRAST_MIN (double)1.0
#define CANNY_CONTRAST_MAX (double)3.0
#define MATCH_SCORES_MIN (double)0.5
#define MATCH_SCORES_MAX (double)0.99

#define INIT_CANNY_THRESH_1 (int)100
#define INIT_CANNY_THRESH_2 (int)200
#define INIT_CANNY_RANGE (int)50
#define INIT_CANNY_CONTRAST (double)1.0
#define INIT_CANNY_KERNEL_SIZE (int)3
#define INIT_MIN_SCORES (double)0.9
#define INIT_GREEDINESS (double)0.0
#define INIT_CONTOURS_TOLERANCE (int)5000
#define INIT_PCA_DIRETION_VECTOR_SCALE (double)0.02

#define DRAW_COLOR_RED      Scalar(0, 0, 255)
#define DRAW_COLOR_GREEN    Scalar(0, 255, 0)
#define DRAW_COLOR_BLUE     Scalar(255, 0, 0)

#define OBJECT_ANGLE_MAX    double(180.0)
#define OBJECT_ANGLE_MIN    double(-180.0)

using namespace cv;
using namespace std;

class GeoModel
    {
    public:
        struct ModelPattern
        {
            Point2d Coordinates;
            Point2d Derivative;
            double Angle;
            double Magnitude;
            Point2d Center;
            Point2d Offset;
        };

        /* PUBLIC FUNCTIONS */
        GeoModel();
        GeoModel(string path, string modelName);
        ~GeoModel();
        void InitModelParameter();
        void setImageOfModel(string modelPath);
        void modelLearnPattern();
        bool isImageEmpty();
        bool isGeometricLearned();
        string getModelFileName();
        Mat getImageOfModel();
        int getImageOfModelCols();
        int getImageOfModelRows();
        Mat getImageOfModelWithPickPosition();
        Mat getImageOfModelCannyThreshold();
        Mat getImageOfModelCannyThreshold(Mat image);
        vector<vector<Point>> getModelContours();
        int getModelContoursSelectedIndex();
        int getModelContoursSelectedArea();
        vector<Vec4i> getModelContoursHierarchy();
        void getModelThreshAreaTolerance(int& high, int& low);
        double getModelRawPcaAngle(bool radians = true);
        Point2f getModelCenterPCA();
        Point2f getDistanceCenterPattern();
        void setAngleOfModel(double value, bool radians = true);
        double getAngleOfModel(bool radians = false);
        vector<ModelPattern> getPatternOfModel();
        void showInfoImage(String windowName, Point2f center_of_temp);
        static void computePcaOrientation(const vector<Point>& pts, double& angleOutput, Point2f& centerOutput);
        static void drawPcaAxis(Mat& img, Point center, double angle, int length,
            Scalar xColour = DRAW_COLOR_RED, Scalar yColour = DRAW_COLOR_BLUE, Scalar centerColour = DRAW_COLOR_GREEN);
        static Mat adjustContrast(cv::Mat image, double alpha);
        static double median(cv::Mat Input);

        /* PUBLIC VARIABLES */
        String nameOfModel;
        double minScores;
        double greediness;
        Point2f pickPosition;
        int cannyRange;
        double cannyContrast;
        int contoursAreaTolerance;

    private:
        /* PRIVATE FUNCTIONS */

        /* PRIVATE VARIABLES */
        String nameOfFile;
        Mat imageOfMOdel;
        // angle of model always store in radians unit
        double angleOfModel;
        vector<ModelPattern> patternOfModel;
        bool geometricLearned;
        double cannyThreshLower;
        double cannyThreshUpper;
        int cannyKernelSize;
        vector<vector<Point>> contoursOfModel;
        vector<Vec4i> hierarchyOfContours;
        int contoursSelectIndex;
        int contoursSelectArea;
        double rawPcaAngle;
        Point2f centerOfPCA;
        Point point1;
        Point point2;
    };





