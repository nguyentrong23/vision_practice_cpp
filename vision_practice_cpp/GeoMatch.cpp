#include "GeoMatch.h"

GeoMatch::GeoMatch()
{

}

GeoMatch::~GeoMatch()
{

}

// lear pattern
bool GeoMatch::addGeoMatchModel(String pathTemplate, String modelName) {

	model tempModel(pathTemplate, modelName);
	// check image model read success or not
	if (tempModel.imageEmpty()) {
		return false;
	}

	tempModel.learnPattern(true);

	// debug
	std::cout << modelName << ": PCA angle - " << tempModel.getPcaAngle() << endl;

	// push to list at bottom
	ModelSrc.push_back(tempModel);
}

bool GeoMatch::addGeoMatchModel(String pathTemplate, String modelName, int pyrDownLevel) {

	model tempModel(pathTemplate, modelName, pyrDownLevel);
	// check image model read success or not
	if (tempModel.imageEmpty()) {
		return false;
	}

	tempModel.learnPattern(true);

	// push to list at bottom
	ModelSrc.push_back(tempModel);
}


// clear model source data
void GeoMatch::clearMatchModel(void) {
	ModelSrc.clear();
}

// remove one model from source with index
void GeoMatch::removeMatchModel(int index) {

	if ((index < 0) || (index >= MAX_NUM_MODEL)) {
		return;
	}

	ModelSrc.erase(ModelSrc.begin() + index);
}

// remove multi model from source
void GeoMatch::removeMatchModel(int startIndex, int endIndex) {

	if (startIndex >= endIndex) {
		return;
	}

	if (((startIndex < 0) || (startIndex >= MAX_NUM_MODEL))
		|| ((endIndex < 0) || (endIndex >= MAX_NUM_MODEL))) {
		return;
	}

	ModelSrc.erase(ModelSrc.begin() + startIndex, ModelSrc.begin() + endIndex);
}

// Get source model size
int GeoMatch::GetModelSrcSize(void) {
	return ModelSrc.size();
}

void GeoMatch::sourceStream(String path) {
	imageSrc = imread(path);
}

void GeoMatch::sourceStream(String path, int pyrDownLevel) {
	imageSrc = imread(path);

	for (int downTime = 0; downTime < pyrDownLevel; downTime++) {
		pyrDown(imageSrc, imageSrc,
			Size(imageSrc.cols / 2, imageSrc.rows / 2));
	}
}

void GeoMatch::Matching(void) {

	if (imageSrc.empty()) {
		return;
	}

	// elapsed time counter
	double startClock;

	Mat imageGray;
	Mat imageThreshold;
	vector<vector<Point>> srcContours;
	vector<Vec4i> srcHierarchy;
	vector<ObjectInfo> objects;

	Mat imageShow;

	imageSrc.copyTo(imageShow);

	/// MATCHING START
	startClock = clock();

	/// Stage 1 find contours in range of models and get this angle
	cvtColor(imageSrc, imageGray, COLOR_RGB2GRAY);
	threshold(imageGray, imageThreshold, 100, 160, THRESH_BINARY_INV);
	//Canny(imageGray, imageThreshold, cannyThresh_1, cannyThresh_2,cannyKernelSize);
	findContours(imageThreshold, srcContours, srcHierarchy,
		cv::RETR_TREE, cv::CHAIN_APPROX_NONE);

	//cout << "Contours size: " << srcContours.size() << endl;
	//imshow("thresh", imageThreshold);

	// find valid contours
	for (int conCounter = 0; conCounter < srcContours.size(); conCounter++) {
		ObjectInfo tempObject;
		int area = contourArea(srcContours[conCounter]);
		for (int listCounter = 0; listCounter < ModelSrc.size(); listCounter++) {
			int highThresh, lowThresh;
			ModelSrc[listCounter].getThreshArea(lowThresh, highThresh);

			if ((area >= lowThresh) && (area <= highThresh)) {
				tempObject.modelCheckList.push_back(listCounter);
			}
		}

		if (tempObject.modelCheckList.size() > 0) {
			tempObject.conArea = area;
			tempObject.conIndex = conCounter;
			tempObject.conBoundingRect = boundingRect(srcContours[conCounter]);
			tempObject.conMinRectArea = minAreaRect(srcContours[conCounter]);
			Moments tempMoment = moments(srcContours[conCounter]);
			tempObject.conCenter.x = tempMoment.m10 / tempMoment.m00;
			tempObject.conCenter.y = tempMoment.m01 / tempMoment.m00;
			objects.push_back(tempObject);
		}
	}

	vector<MatchedObjects> matchedList;
	// get angle of valid contours
	for (int pcaCounter = 0; pcaCounter < objects.size(); pcaCounter++) {
		int conIdx = objects[pcaCounter].conIndex;
		getPcaOrientation(srcContours[conIdx],
			objects[pcaCounter].pcaAngle, objects[pcaCounter].pcaCenter);

		double lastMaxScores = 0.0;
		bool matchedFound = false;
		MatchedObjects tempMatched;

		for (int modelCounter = 0; modelCounter < ModelSrc.size(); modelCounter++) {



			getRotatedROI(imageSrc, ModelSrc[modelCounter], objects[pcaCounter], GetROI_MODE::ROI_POSITIVE);
			matchedFound = matchingScores(objects[pcaCounter].rotPositive, ModelSrc[modelCounter], lastMaxScores);
			//cout << lastMaxScores << endl;
			if (matchedFound) {
				tempMatched.coordinates = objects[pcaCounter].pcaCenter;
				tempMatched.angle = objects[pcaCounter].rotPositive.angle;
			}

			getRotatedROI(imageSrc, ModelSrc[modelCounter], objects[pcaCounter], GetROI_MODE::ROI_NEGATIVE);
			matchedFound = matchingScores(objects[pcaCounter].rotNegative, ModelSrc[modelCounter], lastMaxScores);
			//cout << lastMaxScores << endl;
			if (matchedFound) {
				tempMatched.coordinates = objects[pcaCounter].pcaCenter;
				tempMatched.angle = objects[pcaCounter].rotNegative.angle;
			}

			getRotatedROI(imageSrc, ModelSrc[modelCounter], objects[pcaCounter], GetROI_MODE::ROI_POSITIVE_REVERSE);
			matchedFound = matchingScores(objects[pcaCounter].rotPositive_reverse, ModelSrc[modelCounter], lastMaxScores);
			//cout << lastMaxScores << endl;
			if (matchedFound) {
				tempMatched.coordinates = objects[pcaCounter].pcaCenter;
				tempMatched.angle = objects[pcaCounter].rotPositive_reverse.angle;
			}

			getRotatedROI(imageSrc, ModelSrc[modelCounter], objects[pcaCounter], GetROI_MODE::ROI_NEGATIVE_REVERSE);
			matchedFound = matchingScores(objects[pcaCounter].rotNegative_reverse, ModelSrc[modelCounter], lastMaxScores);
			//cout << lastMaxScores << endl;
			if (matchedFound) {
				tempMatched.coordinates = objects[pcaCounter].pcaCenter;
				tempMatched.angle = objects[pcaCounter].rotNegative_reverse.angle;
			}

			// debug
			//imshow("Positive rotated", objects[pcaCounter].rotPositive.image);
			//imshow("Negative rotated", objects[pcaCounter].rotNegative.image);
			//imshow("Positive reverse rotated", objects[pcaCounter].rotPositive_reverse.image);
			//imshow("Negative reverse rotated", objects[pcaCounter].rotNegative_reverse.image);
			//waitKey(0);
		}

		if (lastMaxScores > ModelSrc[0].scoresMin) {
			matchedList.push_back(tempMatched);
		}
	}


	for (int i = 0; i < matchedList.size(); i++) {
		circle(imageShow, matchedList[i].coordinates, 2, Scalar(255, 255, 0), 2);
		String textPut = to_string(matchedList[i].angle);
		putText(imageShow, textPut, matchedList[i].coordinates, FONT_HERSHEY_COMPLEX, 1.0, Scalar(0, 255, 255), 2);
	}

	imshow("Image show", imageShow);

	// for debug
	//std::cout << "Number contours need check: " << objects.size() << std::endl;
	//for (int i = 0; i < objects.size(); i++) {
	//	cout << "[" << objects[i].contoursIndex << "]" << " ";
	//	cout << "Angle: " << objects[i].angle << " ";
	//	cout << "Center: " << objects[i].center << " ";
	//	cout << "Model index need check: ";
	//	for (int j = 0; j < objects[i].modelCheckList.size(); j++) {
	//		cout << objects[i].modelCheckList[j] << " ";
	//	}
	//	cout << endl;
	//}
	// end debug

	// done
	lastCycleTime = (clock() - startClock) / double(CLOCKS_PER_SEC);
	lastCycleTime *= 1000;
	cout << "elasped time: " << lastCycleTime << " ms" << endl;
}

// get pca orientation and pca center
// return angle in degrees unit
void GeoMatch::getPcaOrientation(const std::vector<Point>& pts, double& angleOutput, Point2f& centerOutput) {
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
	double angle = atan2(eigen_vecs[0].y, eigen_vecs[0].x); // orientation in radians
	angleOutput = angle * R2D;
	centerOutput = cntr;
}

void GeoMatch::getRotatedROI(Mat& matSr, model& model, ObjectInfo& object, GetROI_MODE roiMode) {
	int iCols = matSr.cols;
	int iRows = matSr.rows;
	Mat rotationMatrix;
	double rotateAngle = 0.0;
	Point2f centerOfObject = object.pcaCenter;
	switch (roiMode)
	{
	case GetROI_MODE::ROI_POSITIVE:
		rotateAngle = object.pcaAngle + model.getPcaAngle();
		object.rotPositive.angle = rotateAngle;
		rotationMatrix = getRotationMatrix2D(centerOfObject, rotateAngle, 1.0);
		break;
	case GetROI_MODE::ROI_NEGATIVE:
		rotateAngle = object.pcaAngle - model.getPcaAngle();
		object.rotNegative.angle = rotateAngle;
		rotationMatrix = getRotationMatrix2D(centerOfObject, rotateAngle, 1.0);
		break;
	case GetROI_MODE::ROI_POSITIVE_REVERSE:
		rotateAngle = object.pcaAngle + model.getPcaAngle() + 180.0;
		object.rotPositive_reverse.angle = rotateAngle;
		rotationMatrix = getRotationMatrix2D(centerOfObject, rotateAngle, 1.0);
		break;
	case GetROI_MODE::ROI_NEGATIVE_REVERSE:
		rotateAngle = object.pcaAngle - model.getPcaAngle() + 180.0;
		object.rotNegative_reverse.angle = rotateAngle;
		rotationMatrix = getRotationMatrix2D(centerOfObject, rotateAngle, 1.0);
		break;
	}


	// calculate for resize matrix
	double xVar[4] = { rotationMatrix.at<double>(0, 0),
						rotationMatrix.at<double>(0, 1),
						rotationMatrix.at<double>(1, 0),
						rotationMatrix.at<double>(1, 1) };
	// find image rotated rectangle verticies
	vector<Point> rotatedRect;
	rotatedRect.push_back(Point2i(0, 0));
	rotatedRect.push_back(Point2i(xVar[0] * iCols, xVar[2] * iCols));
	rotatedRect.push_back(Point2i((xVar[0] * iCols + xVar[1] * iRows), (xVar[2] * iCols + xVar[3] * iRows)));
	rotatedRect.push_back(Point2i(xVar[1] * iRows, xVar[3] * iRows));
	Rect BoundingRect = boundingRect(rotatedRect);

	// resize matrix to wrap all image
	rotationMatrix.at<double>(0, 2) = -BoundingRect.x;
	rotationMatrix.at<double>(1, 2) = -BoundingRect.y;
	// rotated image
	switch (roiMode)
	{
	case GetROI_MODE::ROI_POSITIVE:
		warpAffine(matSr, object.rotPositive.image, rotationMatrix, Size(BoundingRect.width, BoundingRect.height));
		break;
	case GetROI_MODE::ROI_NEGATIVE:
		warpAffine(matSr, object.rotNegative.image, rotationMatrix, Size(BoundingRect.width, BoundingRect.height));
		break;
	case GetROI_MODE::ROI_POSITIVE_REVERSE:
		warpAffine(matSr, object.rotPositive_reverse.image, rotationMatrix, Size(BoundingRect.width, BoundingRect.height));
		break;
	case GetROI_MODE::ROI_NEGATIVE_REVERSE:
		warpAffine(matSr, object.rotNegative_reverse.image, rotationMatrix, Size(BoundingRect.width, BoundingRect.height));
		break;
	}

	// transform vector
	vector<Point2f> transformBoundingBox;
	object.conMinRectArea.points(transformBoundingBox);
	transform(transformBoundingBox, transformBoundingBox, rotationMatrix);
	BoundingRect = boundingRect(transformBoundingBox);

	vector<Point2f> transformCenterPoint;
	transformCenterPoint.push_back((Point2f)object.pcaCenter);
	transform(transformCenterPoint, transformCenterPoint, rotationMatrix);
	Point2f topleft = BoundingRect.tl();
	/*Point2f pcaCenterPoint = transformCenterPoint[0] - topleft;*/
	/*cout << BoundingRect.size() << " - " << pcaCenterPoint << endl;*/

	switch (roiMode)
	{
	case GetROI_MODE::ROI_POSITIVE:
		object.rotPositive.image = cropImageWithBorderOffset(object.rotPositive.image, BoundingRect, 2);
		object.rotPositive.center = transformCenterPoint[0] - topleft;
		object.rotPositive.center += model.cPattern2cPca;
		//circle(object.rotPositive.image, object.rotPositive.center, 2, Scalar(255, 255, 0), 2);
		break;
	case GetROI_MODE::ROI_NEGATIVE:
		object.rotNegative.image = cropImageWithBorderOffset(object.rotNegative.image, BoundingRect, 2);
		object.rotNegative.center = transformCenterPoint[0] - topleft;
		object.rotNegative.center += model.cPattern2cPca;
		//circle(object.rotNegative.image, object.rotNegative.center, 2, Scalar(255, 255, 0), 2);
		break;
	case GetROI_MODE::ROI_POSITIVE_REVERSE:
		object.rotPositive_reverse.image = cropImageWithBorderOffset(object.rotPositive_reverse.image, BoundingRect, 2);
		object.rotPositive_reverse.center = transformCenterPoint[0] - topleft;
		object.rotPositive_reverse.center += model.cPattern2cPca;
		//circle(object.rotPositive_reverse.image, object.rotPositive_reverse.center, 2, Scalar(255, 255, 0), 2);
		break;
	case GetROI_MODE::ROI_NEGATIVE_REVERSE:
		object.rotNegative_reverse.image = cropImageWithBorderOffset(object.rotNegative_reverse.image, BoundingRect, 2);
		object.rotNegative_reverse.center = transformCenterPoint[0] - topleft;
		object.rotNegative_reverse.center += model.cPattern2cPca;
		//circle(object.rotNegative_reverse.image, object.rotNegative_reverse.center, 2, Scalar(255, 255, 0), 2);
		break;
	}
}


void GeoMatch::getRotatedROI(Mat& matSrc, GetROI_MODE roiMode, model& model, ObjectInfo& object) {
	// calculate top left and bottom right coordinates
	Point2i topLeft = object.conBoundingRect.tl();
	Point2i botRight = object.conBoundingRect.br();
	topLeft.x -= OBJECT_ROI_BORDER_OFFSET;
	topLeft.y -= OBJECT_ROI_BORDER_OFFSET;
	botRight.x += OBJECT_ROI_BORDER_OFFSET;
	botRight.y += OBJECT_ROI_BORDER_OFFSET;
	// crop image
	Mat cropImg = matSrc(Range(topLeft.y, botRight.y), Range(topLeft.x, botRight.x));
	// find center of object after crop
	Point2f centerOfObject;
	centerOfObject.x = object.pcaCenter.x - topLeft.x;
	centerOfObject.y = object.pcaCenter.y - topLeft.y;

	int iCols = cropImg.cols;
	int iRows = cropImg.rows;
	Mat rotationMatrix;
	double rotateAngle = 0.0;

	switch (roiMode)
	{
	case GetROI_MODE::ROI_POSITIVE:
		rotateAngle = object.pcaAngle + model.getPcaAngle();
		rotationMatrix = getRotationMatrix2D(centerOfObject, rotateAngle, 1.0);
		break;
	case GetROI_MODE::ROI_NEGATIVE:
		rotateAngle = object.pcaAngle - model.getPcaAngle();
		rotationMatrix = getRotationMatrix2D(centerOfObject, rotateAngle, 1.0);
		break;
	case GetROI_MODE::ROI_POSITIVE_REVERSE:
		rotateAngle = object.pcaAngle + model.getPcaAngle() + 180.0;
		rotationMatrix = getRotationMatrix2D(centerOfObject, rotateAngle, 1.0);
		break;
	case GetROI_MODE::ROI_NEGATIVE_REVERSE:
		rotateAngle = object.pcaAngle - model.getPcaAngle() + 180.0;
		rotationMatrix = getRotationMatrix2D(centerOfObject, rotateAngle, 1.0);
		break;
	}

	// calculate for resize matrix
	double xVar[4] = { rotationMatrix.at<double>(0, 0),
						rotationMatrix.at<double>(0, 1),
						rotationMatrix.at<double>(1, 0),
						rotationMatrix.at<double>(1, 1) };

	// find image rotated rectangle verticies
	vector<Point> rotatedRect;
	rotatedRect.push_back(Point2i(0, 0));
	rotatedRect.push_back(Point2i(xVar[0] * iCols, xVar[2] * iCols));
	rotatedRect.push_back(Point2i((xVar[0] * iCols + xVar[1] * iRows), (xVar[2] * iCols + xVar[3] * iRows)));
	rotatedRect.push_back(Point2i(xVar[1] * iRows, xVar[3] * iRows));
	Rect BoundingRect = boundingRect(rotatedRect);

	// resize matrix to wrap all image
	rotationMatrix.at<double>(0, 2) = -BoundingRect.x;
	rotationMatrix.at<double>(1, 2) = -BoundingRect.y;
	// rotated image
	switch (roiMode)
	{
	case GetROI_MODE::ROI_POSITIVE:
		warpAffine(cropImg, object.rotPositive.image, rotationMatrix, Size(BoundingRect.width, BoundingRect.height));
		break;
	case GetROI_MODE::ROI_NEGATIVE:
		warpAffine(cropImg, object.rotNegative.image, rotationMatrix, Size(BoundingRect.width, BoundingRect.height));
		break;
	case GetROI_MODE::ROI_POSITIVE_REVERSE:
		warpAffine(cropImg, object.rotPositive_reverse.image, rotationMatrix, Size(BoundingRect.width, BoundingRect.height));
		break;
	case GetROI_MODE::ROI_NEGATIVE_REVERSE:
		warpAffine(cropImg, object.rotNegative_reverse.image, rotationMatrix, Size(BoundingRect.width, BoundingRect.height));
		break;
	}

	// transform to vector
}

bool GeoMatch::matchingScores(RotatedObject& objectRotated, model& model, double& lastMaxScores) {
	// add check pattern learn and image empty

	bool matchingResult = false;
	Mat imgDest;
	Mat gx;
	Mat gy;
	Mat magnitude;
	Mat angle;

	cvtColor(objectRotated.image, imgDest, COLOR_RGB2GRAY);

	Sobel(imgDest, gx, CV_64F, 1, 0, 3);
	Sobel(imgDest, gy, CV_64F, 0, 1, 3);

	cartToPolar(gx, gy, magnitude, angle);

	// ncc match search
	long noOfCordinates = model.patternInfo.size();
	// normalized min score
	double normMinScore = model.scoresMin / noOfCordinates;
	double normGreediness = ((1 - model.greediness * model.scoresMin) / (1 - model.greediness)) / noOfCordinates;
	double partialScore = 0;
	double resultScore = 0;

	int checkCounter = 0;

	int startRowIdx = objectRotated.center.y - 20;
	int endRowIdx = objectRotated.center.y + 20;
	int startColIdx = objectRotated.center.x - 20;
	int endColIdx = objectRotated.center.x + 20;

	for (int rowIdx = startRowIdx; rowIdx < endRowIdx; rowIdx++)
	{
		for (int colIdx = startColIdx; colIdx < endColIdx; colIdx++)
		{
			double partialSum = 0;
			for (int count = 0; count < noOfCordinates; count++)
			{
				model::ModelPatternInfo tempPoint = model.patternInfo[count];

				int CoorX = (int)(colIdx + tempPoint.Offset.x);
				int CoorY = (int)(rowIdx + tempPoint.Offset.y);

				double iTx = tempPoint.Derivative.x;
				double iTy = tempPoint.Derivative.y;

				// ignore invalid pixel
				if (CoorX < 0 || CoorY < 0 || CoorY >(imgDest.rows - 1) || CoorX >(imgDest.cols - 1)) {
					continue;
				}

				double iSx = gx.at<double>(CoorY, CoorX);
				double iSy = gy.at<double>(CoorY, CoorX);

				if ((iSx != 0 || iSy != 0) && (iTx != 0 || iTy != 0))
				{
					double mag = magnitude.at<double>(CoorY, CoorX);
					double matGradMag = (mag == 0) ? 0 : 1 / mag;
					partialSum += ((iSx * iTx) + (iSy * iTy)) * (tempPoint.Magnitude * matGradMag);
				}

				int sumOfCoords = count + 1;

				partialScore = partialSum / sumOfCoords;

				double minBreakScores = std::min((model.scoresMin - 1) + normGreediness * sumOfCoords, normMinScore * sumOfCoords);

				if (partialScore < minBreakScores) {
					break;

					if (partialScore < lastMaxScores) {
						break;
					}
				}
			}

			if (partialScore > resultScore) {
				resultScore = partialScore;
			}
		}
	}

	if (resultScore > lastMaxScores) {
		lastMaxScores = resultScore;
		return true;
	}

	return false;
}

bool GeoMatch::matchingScores(Mat& inputImage, model& model, double& lastMaxScores) {
	// add check pattern learn and image empty

	bool matchingResult = false;
	Mat imgDest;
	Mat gx;
	Mat gy;
	Mat magnitude;
	Mat angle;

	cvtColor(inputImage, imgDest, COLOR_RGB2GRAY);

	Sobel(imgDest, gx, CV_64F, 1, 0, 3);
	Sobel(imgDest, gy, CV_64F, 0, 1, 3);

	cartToPolar(gx, gy, magnitude, angle);

	// ncc match search
	long noOfCordinates = model.patternInfo.size();
	// normalized min score
	double normMinScore = model.scoresMin / noOfCordinates;
	double normGreediness = ((1 - model.greediness * model.scoresMin) / (1 - model.greediness)) / noOfCordinates;
	double partialScore = 0;
	double resultScore = 0;

	int checkCounter = 0;

	int startRowIdx = (inputImage.rows / 2) - 50;
	int endRowIdx = (inputImage.rows / 2) + 50;
	int startColIdx = (inputImage.rows / 2) - 50;
	int endColIdx = (inputImage.rows / 2) + 50;

	for (int rowIdx = startRowIdx; rowIdx < endRowIdx; rowIdx++)
	{
		for (int colIdx = startColIdx; colIdx < endColIdx; colIdx++)
		{
			double partialSum = 0;
			for (int count = 0; count < noOfCordinates; count++)
			{
				model::ModelPatternInfo tempPoint = model.patternInfo[count];

				int CoorX = (int)(colIdx + tempPoint.Offset.x);
				int CoorY = (int)(rowIdx + tempPoint.Offset.y);

				double iTx = tempPoint.Derivative.x;
				double iTy = tempPoint.Derivative.y;

				// ignore invalid pixel
				if (CoorX < 0 || CoorY < 0 || CoorY >(imgDest.rows - 1) || CoorX >(imgDest.cols - 1)) {
					continue;
				}

				double iSx = gx.at<double>(CoorY, CoorX);
				double iSy = gy.at<double>(CoorY, CoorX);

				if ((iSx != 0 || iSy != 0) && (iTx != 0 || iTy != 0))
				{
					double mag = magnitude.at<double>(CoorY, CoorX);
					double matGradMag = (mag == 0) ? 0 : 1 / mag;
					partialSum += ((iSx * iTx) + (iSy * iTy)) * (tempPoint.Magnitude * matGradMag);
				}

				int sumOfCoords = count + 1;

				partialScore = partialSum / sumOfCoords;

				double minBreakScores = std::min((model.scoresMin - 1) + normGreediness * sumOfCoords, normMinScore * sumOfCoords);

				if (partialScore < minBreakScores) {
					break;

					if (partialScore < lastMaxScores) {
						break;
					}
				}
			}

			if (partialScore > resultScore) {
				resultScore = partialScore;
			}
		}
	}

	if (resultScore > lastMaxScores) {
		lastMaxScores = resultScore;
		return true;
	}

	return false;
}

Mat GeoMatch::cropImageWithBorderOffset(Mat sourceImage, Rect boxBounding, int border) {

	Mat outputMat;
	int minOffset_X, maxOffset_X;
	int minOffset_Y, maxOffset_Y;

	Point topLeft = boxBounding.tl();
	Point botRight = boxBounding.br();

	minOffset_X = topLeft.x - border;
	maxOffset_X = botRight.x + border;

	minOffset_Y = topLeft.y - border;
	maxOffset_Y = botRight.y + border;

	if (minOffset_X < 0) {
		minOffset_X = 0;
	}

	if (maxOffset_X > sourceImage.cols) {
		maxOffset_X = sourceImage.cols;
	}

	if (minOffset_Y < 0) {
		minOffset_Y = 0;
	}

	if (maxOffset_Y > sourceImage.rows) {
		maxOffset_Y = sourceImage.rows;
	}

	outputMat = sourceImage(Range(minOffset_Y, maxOffset_Y), Range(minOffset_X, maxOffset_X));

	return outputMat;
}