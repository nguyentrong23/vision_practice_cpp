#include "GeometricModel.h"

// constructor normal
GeometricModel::GeometricModel()
{
	InitParameter();
}

// constructor with normal initialize
GeometricModel::GeometricModel(String pathTemplate, String modelName)
{
	InitParameter();
	readImage(pathTemplate);
	nameOfModel = modelName;
}

// constructor init template image with pyramid down
GeometricModel::GeometricModel(String pathTemplate, String modelName, int pyramidDownLevel) {
	InitParameter();

	readImage(pathTemplate, pyramidDownLevel);
	nameOfModel = modelName;
}

// destructor
GeometricModel::~GeometricModel()
{

}

// init parameter
void GeometricModel::InitParameter(void) {
	cannyThresh_1 = INIT_CANNY_THRESH_1;
	cannyThresh_2 = INIT_CANNY_THRESH_2;
	cannyKernelSize = INIT_CANNY_KERNEL_SIZE;
	scoresMin = INIT_MIN_SCORES;
	greediness = INIT_GREEDINESS;
	contoursAreaTolerance = INIT_CONTOURS_TOLERANCE;
	bIsGeometricLearn = false;
}

// save model image
bool GeometricModel::readImage(String modelPath) {
	if (modelPath.empty()) {
		return false;
	}

	imageOriginal = imread(modelPath);

	if (imageOriginal.empty()) {
		return false;
	}

	height = imageOriginal.rows;
	width = imageOriginal.cols;

	return true;
}

bool GeometricModel::readImage(String modelPath, int pyrDownLevel) {
	if (modelPath.empty()) {
		return false;
	}

	imageOriginal = imread(modelPath);

	for (int downTime = 0; downTime < pyrDownLevel; downTime++) {
		pyrDown(imageOriginal, imageOriginal,
			Size(imageOriginal.cols / 2, imageOriginal.rows / 2));
	}

	if (imageOriginal.empty()) {
		return false;
	}

	height = imageOriginal.rows;
	width = imageOriginal.cols;

	return true;
}

bool GeometricModel::learnPattern(bool bShowImage) {
	if (imageOriginal.empty()) {
		return false;
	}

	Mat imgGray;
	Mat imgOutput;
	Mat gx, gy;
	Mat magnitude, angle;
	Point2d sumPoint = Point2d(0, 0);

	cvtColor(imageOriginal, imgGray, COLOR_RGB2GRAY);
	Canny(imgGray, imgOutput, cannyThresh_1, cannyThresh_2,
		cannyKernelSize);
	findContours(imgOutput, contours, hierarchy,
		cv::RETR_TREE, cv::CHAIN_APPROX_NONE);
	Sobel(imgGray, gx, CV_64F, 1, 0, 3);
	Sobel(imgGray, gy, CV_64F, 0, 1, 3);
	//compute the magnitude and direction(radians)
	cartToPolar(gx, gy, magnitude, angle);

	// duyệt qua từng điểm trong contours và chuẩn bị sumpoint để tính tâm cho contours
	patternInfo.clear();
	for (int contourIdx = 0; contourIdx < contours.size(); contourIdx++) {
		for (int pointInx = 0; pointInx < contours[contourIdx].size(); pointInx++)
		{
			ModelPatternInfo tempData;
			double mag;
			tempData.Coordinates = contours[contourIdx][pointInx];
			tempData.Derivative = Point2d(gx.at<double>(tempData.Coordinates)
				, gy.at<double>(tempData.Coordinates));
			tempData.Angle = angle.at<double>(tempData.Coordinates);
			mag = magnitude.at<double>(tempData.Coordinates);
			tempData.Magnitude = (mag == 0) ? 0 : (1 / mag);
			// push to container
			patternInfo.push_back(tempData);

			sumPoint += tempData.Coordinates;
		}
	}

	// tính tâm của contours trong template
	Point2f templateCenterPoint(sumPoint.x / patternInfo.size(), sumPoint.y / patternInfo.size());
	for (ModelPatternInfo& pointTemp : patternInfo) {
		pointTemp.Center = templateCenterPoint;
		pointTemp.Offset = pointTemp.Coordinates - pointTemp.Center;
	}


	// find contour area for matching process
	int maxNumPoint = 0;
	contoursSelectArea = 0;
	contoursSelectIndex = 0;
	for (int conCounter = 0; conCounter < contours.size(); conCounter++) {
		if (contours[conCounter].size() > maxNumPoint) {
			maxNumPoint = contours[conCounter].size();
			contoursSelectIndex = conCounter;
			contoursSelectArea = contourArea(contours[conCounter]);
		}
	}

	// tính hướng cho template
	getPcaOrientation(contours[contoursSelectIndex],
		originalPcaAngle, centerPCA);

	// khoảng cách giữa tâm của template và tâm của pca
	cPattern2cPca = templateCenterPoint - centerPCA;

	if (bShowImage) {
		showInfoImage(nameOfModel);
	}
	bIsGeometricLearn = true;
	return true;
}
// get pca orientation and pca center
// return angle in degrees unit
void GeometricModel::getPcaOrientation(const std::vector<Point>& pts, double& angleOutput, Point2f& centerOutput) {
	//Construct a buffer used by the pca analysis
	int sz = static_cast<int>(pts.size());
	Mat data_pts = Mat(sz, 2, CV_64F);
	for (int i = 0; i < data_pts.rows; i++)
	{
		data_pts.at<double>(i, 0) = pts[i].x;
		data_pts.at<double>(i, 1) = pts[i].y;
	}
	//Perform PCA analysis
	PCA pca_analysis(data_pts, Mat(), PCA::DATA_AS_ROW);
	//Store the center of the object
	Point cntr = Point(static_cast<int>(pca_analysis.mean.at<double>(0, 0)),
		static_cast<int>(pca_analysis.mean.at<double>(0, 1)));
	//Store the eigenvalues and eigenvectors
	vector<Point2d> eigen_vecs(2);
	vector<double> eigen_val(2);
	for (int i = 0; i < 2; i++)
	{
		eigen_vecs[i] = Point2d(pca_analysis.eigenvectors.at<double>(i, 0),
			pca_analysis.eigenvectors.at<double>(i, 1));
		eigen_val[i] = pca_analysis.eigenvalues.at<double>(i);
	}
	point1pca = cntr + INIT_PCA_DIRETION_VECTOR_SCALE * Point(static_cast<int>(eigen_vecs[0].x * eigen_val[0]), static_cast<int>(eigen_vecs[0].y * eigen_val[0]));
	point2pca = cntr - INIT_PCA_DIRETION_VECTOR_SCALE * Point(static_cast<int>(eigen_vecs[1].x * eigen_val[1]), static_cast<int>(eigen_vecs[1].y * eigen_val[1]));
	double angle = atan2(eigen_vecs[0].y, eigen_vecs[0].x); // orientation in radians
	angleOutput = angle * R2D;
	centerOutput = cntr;
}

// check model image empty or not
bool GeometricModel::imageEmpty(void) {
	return (imageOriginal.empty());
}

bool GeometricModel::geometricLearned(void) {
	return bIsGeometricLearn;
}

void GeometricModel::showInfoImage(String windowName) {
	if (windowName.empty()) {
		return;
	}

	Mat showImg;
	// avoid memory optimize of compiler
	imageOriginal.copyTo(showImg);

	// draw all contours
	drawContours(showImg, contours, -1, Scalar(255, 0, 0), 1);

	// draw pca center and direction
	circle(showImg, centerPCA, 3, Scalar(255, 0, 255), 2);
	drawPcaAxis(showImg, centerPCA, point1pca, Scalar(0, 255, 0));
	drawPcaAxis(showImg, centerPCA, point2pca, Scalar(255, 255, 0));

	imshow(windowName, showImg);
}

void GeometricModel::drawPcaAxis(Mat& img, Point p, Point q, Scalar colour, const float scale)
{
	double angle = atan2((double)p.y - q.y, (double)p.x - q.x); // angle in radians
	double hypotenuse = sqrt((double)(p.y - q.y) * (p.y - q.y) + (p.x - q.x) * (p.x - q.x));
	// Here we lengthen the arrow by a factor of scale
	q.x = (int)(p.x - scale * hypotenuse * cos(angle));
	q.y = (int)(p.y - scale * hypotenuse * sin(angle));
	line(img, p, q, colour, 1, LINE_AA);
}

void GeometricModel::getThreshArea(int& lowThresh, int& highThresh) {
	lowThresh = contoursSelectArea - contoursAreaTolerance;
	highThresh = contoursSelectArea + contoursAreaTolerance;
}

double GeometricModel::getPcaAngle(void) {
	return originalPcaAngle;
}
