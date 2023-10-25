#include "model.h"


// constructor normal
model::model()
{
	InitParameter();
}

// constructor with normal initialize
model::model(String pathTemplate, String modelName)
{
	InitParameter();
	readImage(pathTemplate);
	nameOfModel = modelName;
}

// constructor init template image with pyramid down
model::model(String pathTemplate, String modelName, int pyramidDownLevel) {
	InitParameter();

	readImage(pathTemplate, pyramidDownLevel);
	nameOfModel = modelName;
}

// destructor
model::~model()
{
}
void model::InitParameter(void)
{
	cannyThresh_1 = INIT_CANNY_THRESH_1;
	cannyThresh_2 = INIT_CANNY_THRESH_2;
	cannyKernelSize = INIT_CANNY_KERNEL_SIZE;
	scoresMin = INIT_MIN_SCORES;
	greediness = INIT_GREEDINESS;
	contoursAreaTolerance = INIT_CONTOURS_TOLERANCE;
	bIsGeometricLearn = false;
}
bool model::readImage(String modelPath)
{
	if (modelPath.empty())
	{
		return false;
	}
	imageOriginal = imread(modelPath);
	height = imageOriginal.rows;
	width = imageOriginal.cols;
	return true;
}
bool model::readImage(String modelPath, int pyrDownLevel)
{
	if (modelPath.empty())
	{
		return false;
	}
	imageOriginal = imread(modelPath);
	for (int i = 0; i < pyrDownLevel; i++)
	{
		pyrDown(imageOriginal, imageOriginal, Size(imageOriginal.rows/2, imageOriginal.cols/2));
	}
	height = imageOriginal.rows;
	width = imageOriginal.cols;
	return true;
}
bool model::learnPattern(bool bShowImage)
{
	if (imageOriginal.empty()) {
		return false;
	}

	Mat imgGray;
	Mat imgOutput;
	Mat gx, gy;
	Mat magnitude, angle;
	Point2d sumPoint = Point2d(0, 0);

	cvtColor(imageOriginal, imgGray, COLOR_RGB2GRAY);
	Canny(imgGray, imgOutput, cannyThresh_1, cannyThresh_2, cannyKernelSize);
	findContours(imgOutput,contours, hierarchy,cv::RETR_TREE, cv::CHAIN_APPROX_NONE);

	Sobel(imgGray, gx, CV_64F, 1, 0, 3);
	Sobel(imgGray, gy, CV_64F, 0, 1, 3);

	cartToPolar(gx, gy, magnitude, angle);
	patternInfo.clear();
	//tiền xử lý
	

	// thu thập dữ liệu vào pattern info
	for (int contourIdx = 0; contourIdx < contours.size(); contourIdx++) {
		for (int pointInx = 0; pointInx < contours[contourIdx].size(); pointInx++)
		{   // lấy data từ matran xuong phai dung at
			ModelPatternInfo temp_pixel;
			double mag;
			temp_pixel.Coordinates = contours[contourIdx][pointInx];
			temp_pixel.Angle = angle.at<double>(temp_pixel.Coordinates.x,temp_pixel.Coordinates.y);
			temp_pixel.Derivative = Point2d(gx.at<double>(temp_pixel.Coordinates), gy.at<double>(temp_pixel.Coordinates));
			mag = magnitude.at<double>(temp_pixel.Coordinates);
			temp_pixel.Magnitude = (mag == 0) ? 0 : 1 / mag;

			patternInfo.push_back(temp_pixel);
			sumPoint += temp_pixel.Coordinates;
		}
	}
	Point2f center_of_model(sumPoint.x / patternInfo.size(), sumPoint.y / patternInfo.size());
	for (ModelPatternInfo& tempattern : patternInfo)
	{
		tempattern.Center = center_of_model;
		tempattern.Offset = tempattern.Coordinates - tempattern.Center;
	}
	// tìm contours lớn nhất trong các contours để đoán hướng
	int maxNumPoint = 0;
	contoursSelectArea = 0;
	contoursSelectIndex = 0;
	for (int contourIdx = 0; contourIdx < contours.size(); contourIdx++) {
		if (contours[contourIdx].size() > maxNumPoint)
		{
			maxNumPoint = contours[contourIdx].size();
			contoursSelectIndex = contourIdx;
			contoursSelectArea = contourArea(contours[contourIdx]);
	}
	}

	getPcaOrientation(contours[contoursSelectIndex],
		originalPcaAngle, centerPCA);

	cPattern2cPca = center_of_model - centerPCA;

	if (bShowImage) {
		showInfoImage(nameOfModel);
	}

	bIsGeometricLearn = true;
	return true;
}
void model::getPcaOrientation(const vector<Point>& pts, double& angleOutput, Point2f& centerOutput)
{
	int size = static_cast<int>(pts.size());
	Mat data = Mat(size, 2, CV_64F);
	for (int i = 0; i < pts.size(); i++)
	{
		data.at<double>(i, 0) = pts[i].x;
		data.at<double>(i, 1) = pts[i].y;
	}
	PCA pca_analyer(data, Mat(), PCA::DATA_AS_ROW);
	Point2d  center_of_pca = Point2d(       static_cast<int>(pca_analyer.mean.at<double>(0, 0)),
		                          static_cast<int>(pca_analyer.mean.at<double>(0, 1)));
	vector<Point2d> eigen_vector(2);
	vector<double> eigen_val(2);
	for (int i = 0; i < 2; i++)
	{
		eigen_vector[i] = Point2d(static_cast<int>(pca_analyer.eigenvectors.at<double>(i, 0)),
								  static_cast<int>(pca_analyer.eigenvectors.at<double>(i, 1)));
		eigen_val[i] = static_cast<int>(pca_analyer.eigenvectors.at<double>(i));
	}

	point1pca = center_of_pca + INIT_PCA_DIRETION_VECTOR_SCALE * Point2d(static_cast<int>(eigen_vector[0].x * eigen_val[0]), static_cast<int>(eigen_vector[0].y * eigen_val[0]));
	point2pca = center_of_pca - INIT_PCA_DIRETION_VECTOR_SCALE * Point2d(static_cast<int>(eigen_vector[1].x * eigen_val[1]), static_cast<int>(eigen_vector[1].y * eigen_val[1]));
	angleOutput = atan2(eigen_vector[0].y, eigen_vector[0].x);
	angleOutput = angleOutput * R2D;
	centerOutput = center_of_pca;
}

void model::showInfoImage(String windowName)
{
	if (windowName.empty()) {
		return;
	}

	Mat showImg;

	imageOriginal.copyTo(showImg);
	drawContours(showImg, contours, -1, Scalar(255, 0, 0), 1);

	circle(showImg, centerPCA, 3, Scalar(255, 0, 255), 2);
	drawPcaAxis(showImg, centerPCA, point1pca, Scalar(0, 255, 0));
	drawPcaAxis(showImg, centerPCA, point2pca, Scalar(255, 255, 0));

	imshow(windowName, showImg);
}
void model::drawPcaAxis(Mat& img, Point p, Point q, Scalar colour, const float scale)
{
	double angle = atan2((double)p.y - q.y, (double)p.x - q.x); // angle in radians
	double hypotenuse = sqrt((double)(p.y - q.y) * (p.y - q.y) + (p.x - q.x) * (p.x - q.x));
	q.x = (int)(p.x - scale * hypotenuse * cos(angle));
	q.y = (int)(p.y - scale * hypotenuse * sin(angle));
	line(img, p, q, colour, 1, LINE_AA);
}
bool model::imageEmpty(void)
{
	return (imageOriginal.empty());
}
bool model::geometricLearned(void)
{
	return bIsGeometricLearn;
}

void model::getThreshArea(int& lowThresh, int& highThresh)
{
	lowThresh = contoursSelectArea - contoursAreaTolerance;
	highThresh = contoursSelectArea + contoursAreaTolerance;
}
double model::getPcaAngle(void)
{
	return originalPcaAngle;
}
