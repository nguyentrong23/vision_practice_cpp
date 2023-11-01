#include "GeoMatch.h"

    GeoMatch::GeoMatch()
    {

    }
    GeoMatch::~GeoMatch()
    {

    }

    bool GeoMatch::addGeoMatchModel(String pathTemplate, String modelName) {
        GeoModel tempModel(pathTemplate, modelName);
        if (tempModel.isImageEmpty()) {
            return false;
        }
        tempModel.modelLearnPattern();

        ModelSrc.push_back(tempModel);
        return true;
    }

    void GeoMatch::clearMatchModel(void) {
        ModelSrc.clear();
    }

    void GeoMatch::removeMatchModel(int index) {
        if ((index < 0) || (index >= MAX_NUM_MODEL)) {
            return;
        }
        ModelSrc.erase(ModelSrc.begin() + index);
    }

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

    void GeoMatch::modifyMatchModelAt(int index, GeoModel model) {
        ModelSrc[index] = model;
    }

    vector<GeoModel> GeoMatch::getModelSource() {
        return ModelSrc;
    }

    int GeoMatch::getModelSourceSize(void) {
        return (int)ModelSrc.size();
    }

    void GeoMatch::setImageSource(string path) {
        imageSource = imread(path);
    }

    void GeoMatch::setImageSource(Mat img) {
        img.copyTo(imageSource);
    }

    bool GeoMatch::haveObjectsInPlate() {
        return objectsInPlate;
    }

    void GeoMatch::matching(Mat image, bool boudingBoxChecking) {
        image.copyTo(imageSource);
        matching(boudingBoxChecking);
    }

    void GeoMatch::matching(bool boudingBoxChecking) {
        if (imageSource.empty()) {
            return;
        }

        resultImage = imageSource.clone();

        if (ModelSrc.empty()) {
            return;
        }


        double startClock;
        auto start = std::chrono::high_resolution_clock::now();  // = clock()

        /// MATCHING START
        Mat imageGray;
        Mat imageThresh;
        vector<vector<Point>> srcContours;
        vector<Vec4i> srcHierarchy;
        // pre-processing
        cvtColor(imageSource, imageGray, COLOR_RGB2GRAY);
        GaussianBlur(imageGray, imageGray, cv::Size(3, 3), 0);
        // find contours
        threshold(imageGray, imageThresh, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
        findContours(imageThresh, srcContours, srcHierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);

        // cờ
        objectsInPlate = false;

        // chẹc contours phù hợp về area
        vector<PossibleObject> objects;   
        vector<vector<Point>> possibleCollisionBox;
        const int minNoiseArea = (int)(imageSource.cols * imageSource.rows * 0.05);// noise min
        const int maxNoiseArea = (int)(imageSource.cols * imageSource.rows * 0.95);// noise max


        // xứ lý kỹ contours để xác định đường viền con và cha
        for (int conCounter = 0; conCounter < srcContours.size(); conCounter++) {
            PossibleObject tempObject;
            int area = contourArea(srcContours[conCounter]);

            // nếu object lớn hơn ngưỡng dưới
            if (area >= minNoiseArea) {
                objectsInPlate = true;
            }

            // object lớn hơn ngưỡng trên
            if (area >= maxNoiseArea) {
                for (int index = 0; index < srcHierarchy.size(); index++) { // duyệt qua Hierarchy của tất cả contours
                    if (srcHierarchy[index][3] == conCounter) { // srcHierarchy[index][3] chỉ index của đường viền cha
                        srcHierarchy[index][3] = -1;  // tóm lại là kiểm tra xem đường viền hiện tại có đường viền con nào k 
                    }                                // nếu có cha là thằng contours hiện tại thì cho nó thành mồ côi
                }
                continue;
            }

            // nếu   countours hiện tại không có cha thì add nó vào list possible 
            if (srcHierarchy[conCounter][3] < 0) {
                possibleCollisionBox.push_back(srcContours[conCounter]);
            }

            // kiểm tra diện tích có valid với temlate nào không
            for (int modelCounter = 0; modelCounter < ModelSrc.size(); modelCounter++) {
                // get area in limit lower and upper based model area
                int lowerArea = ModelSrc[modelCounter].getModelContoursSelectedArea() * lowerThreshRatio;
                int upperArea = ModelSrc[modelCounter].getModelContoursSelectedArea() * upperThreshRatio;
                if ((area <= lowerArea) || (area >= upperArea)) {     
                    continue;
                }
                tempObject.modelCheckList.push_back(modelCounter); // thêm những model nào  phù hợp với contours vào checklist
            }

            if (tempObject.modelCheckList.size() > 0) { 

                tempObject.conBoundingRect = boundingRect(srcContours[conCounter]);
                tempObject.conMinRectArea = minAreaRect(srcContours[conCounter]);
                GeoModel::computePcaOrientation(srcContours[conCounter], tempObject.pcaAngle, tempObject.pcaCenter);
                objects.push_back(tempObject); // thêm contours phù hợp với ít nhất 1 template vào danh sách object
            }
        }

        // matching with input model
        vector<MatchedObj> matchObjList;
        double time = 0;

        startClock = clock();
        for (int objCounter = 0; objCounter < objects.size(); objCounter++) 
        { // duyệt qua từng object
            double maxScores = 0.0;
            MatchedObj matchObj;
            // trong mõi object duyệt qua những template trong modelchecklist
            for (int modelCounter = 0; modelCounter < objects[objCounter].modelCheckList.size(); modelCounter++)
            {
                int modelIndex = objects[objCounter].modelCheckList[modelCounter]; // index of model
                double rawAngle = ModelSrc[modelIndex].getModelRawPcaAngle(); // get pca angel
                Point2f ROICenter = objects[objCounter].pcaCenter;// center of contours
                
               // xoay từng góc  và matching


                double startClock1 = clock();
                objects[objCounter].rotPositive.angle = objects[objCounter].pcaAngle + rawAngle;
                getRotatedROI(imageSource, objects[objCounter].rotPositive, ROICenter, objects[objCounter].conMinRectArea);
                double  CycleTime1 = (clock() - startClock1) / double(CLOCKS_PER_SEC);
                time = time + CycleTime1;
                if (matchingScores(objects[objCounter].rotPositive, ModelSrc[modelIndex], maxScores)) {
                    saveMatchedObjectInfo(matchObj, objects[objCounter].rotPositive,  ModelSrc[modelIndex], maxScores, objects[objCounter].pcaCenter);
                    matchObj.rect = objects[objCounter].conBoundingRect;
                    break;
                }


                double startClock2 = clock();
                objects[objCounter].rotNegative.angle = objects[objCounter].pcaAngle - rawAngle;
                getRotatedROI(imageSource, objects[objCounter].rotNegative, ROICenter, objects[objCounter].conMinRectArea);
                double  CycleTime2 = (clock() - startClock2) / double(CLOCKS_PER_SEC);
                time = time + CycleTime2;
                if (matchingScores(objects[objCounter].rotNegative, ModelSrc[modelIndex], maxScores)) {
                    saveMatchedObjectInfo(matchObj, objects[objCounter].rotNegative, ModelSrc[modelIndex], maxScores, objects[objCounter].pcaCenter);
                    matchObj.rect = objects[objCounter].conBoundingRect;
                    break;
                }


                double startClock3 = clock();
                objects[objCounter].rotPositive_reverse.angle = objects[objCounter].pcaAngle + rawAngle + CV_PI;
                getRotatedROI(imageSource, objects[objCounter].rotPositive_reverse, ROICenter, objects[objCounter].conMinRectArea);
                double  CycleTime3 = (clock() - startClock3) / double(CLOCKS_PER_SEC);
                time = time + CycleTime3;
                if (matchingScores(objects[objCounter].rotPositive_reverse, ModelSrc[modelIndex], maxScores)) {
                    saveMatchedObjectInfo(matchObj, objects[objCounter].rotPositive_reverse,  ModelSrc[modelIndex], maxScores, objects[objCounter].pcaCenter);
                    matchObj.rect = objects[objCounter].conBoundingRect;
                    break;
                }

                double startClock4 = clock();
                objects[objCounter].rotNegative_reverse.angle = objects[objCounter].pcaAngle - rawAngle + CV_PI;
                getRotatedROI(imageSource, objects[objCounter].rotNegative_reverse, ROICenter, objects[objCounter].conMinRectArea);
                double  CycleTime4 = (clock() - startClock4) / double(CLOCKS_PER_SEC);
                time = time + CycleTime4;
                if (matchingScores(objects[objCounter].rotNegative_reverse, ModelSrc[modelIndex], maxScores)) {
                    saveMatchedObjectInfo(matchObj, objects[objCounter].rotNegative_reverse, ModelSrc[modelIndex], maxScores, objects[objCounter].pcaCenter);
                    matchObj.rect = objects[objCounter].conBoundingRect;
                    break;
                }
            }

            pickingBoxSize.height = 60;
            pickingBoxSize.width = 20;
            if (!matchObj.image.empty()) 
            {
                //// CHECK PICKING BOX HAS INTERSECTION WITH ANOTHER BOUNDING BOX
                bool pickingBoxCollision = false;
                // create picking box
                matchObj.pickingBox = RotatedRect(matchObj.coordinates, pickingBoxSize, matchObj.angle * R2D);
                // create picking box collision with any possible object contours.
                if (boudingBoxChecking) {
                    vector<Point2f> pickingBouding;
                    matchObj.pickingBox.points(pickingBouding);
                    for (int boxCounter = 0; boxCounter < possibleCollisionBox.size(); boxCounter++) {
                        vector<Point> convex;
                        intersectConvexConvex(possibleCollisionBox[boxCounter], pickingBouding, convex, false);
                        if (convex.size() > 0) {
                            pickingBoxCollision = true;
                            break;
                        }
                    }
                }
                if (!pickingBoxCollision) {
                    matchObjList.push_back(matchObj);
                }
            }
        }

        time *= 1000;
        cout << "rotate time :" << time << " ms" << endl;
        double lastCycleTime = (clock() - startClock) / double(CLOCKS_PER_SEC);
        lastCycleTime *= 1000;
        cout << "detect time  : " << lastCycleTime << " ms" << endl;

        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<chrono::milliseconds>(stop - start);
        lastExecuteTime = duration.count();
        std::cout << "Time execution: " << duration.count() << "ms" << std::endl;



        for (int i = 0; i < matchObjList.size(); i++) {
            GeoModel::drawPcaAxis(resultImage, matchObjList[i].coordinates, matchObjList[i].angle, 20);
            rectangle(resultImage, matchObjList[i].rect, Scalar(0, 255, 0), 1);
            drawPickingBox(resultImage, matchObjList[i].pickingBox, DRAW_COLOR_RED);
            std::cout << "Max scores: " << matchObjList[i].scores << ", Name: " << matchObjList[i].name
                << ", Pick position: " << matchObjList[i].coordinates << std::endl;
        }

        imshow("resultImage", resultImage);
    }

    void GeoMatch::saveMatchedObjectInfo(MatchedObj& matched, RotatedObj& objectRotated ,GeoModel model, double scores, Point2f pcaCenter) {
        matched.name = model.nameOfModel;
        matched.image = objectRotated.image;
        matched.angle = objectRotated.angle + model.getAngleOfModel(true);
        matched.scores = scores;
        Point2f modelPickDistance;
        modelPickDistance.x = model.pickPosition.x - model.getPatternOfModel()[0].Center.x;
        modelPickDistance.y = model.pickPosition.y - model.getPatternOfModel()[0].Center.y;

        Point2f srcPatternDistance;
        srcPatternDistance.x = objectRotated.centerMaxScores.x - objectRotated.centerCrop.x;
        srcPatternDistance.y = objectRotated.centerMaxScores.y - objectRotated.centerCrop.y;

        int xDist = srcPatternDistance.x + modelPickDistance.x;
        int yDist = srcPatternDistance.y + modelPickDistance.y;

        matched.coordinates.x = pcaCenter.x + ((cos(objectRotated.angle) * xDist) - (sin(objectRotated.angle) * yDist));
        matched.coordinates.y = pcaCenter.y + ((sin(objectRotated.angle) * xDist) + (cos(objectRotated.angle) * yDist));
    }

    void GeoMatch::getRotatedROI(Mat& matSrc, RotatedObj& object, Point center, RotatedRect minRectArea) {
        int srcCols = matSrc.cols;
        int srcRows = matSrc.rows;
        Mat rotationMatrix;
        rotationMatrix = getRotationMatrix2D(center, object.angle * R2D, 1.0);
        // calculate for resize matrix
        double xVar[4] = { rotationMatrix.at<double>(0, 0),
                          rotationMatrix.at<double>(0, 1),
                          rotationMatrix.at<double>(1, 0),
                          rotationMatrix.at<double>(1, 1) };
        // find image rotated rectangle verticies
        vector<Point> rotatedRect;
        rotatedRect.push_back(Point2i(0, 0));
        rotatedRect.push_back(Point2i(xVar[0] * srcCols, xVar[2] * srcCols));
        rotatedRect.push_back(Point2i((xVar[0] * srcCols + xVar[1] * srcRows), (xVar[2] * srcCols + xVar[3] * srcRows)));
        rotatedRect.push_back(Point2i(xVar[1] * srcRows, xVar[3] * srcRows));
        Rect BoundingRect = boundingRect(rotatedRect);
        // resize matrix to wrap all image
        rotationMatrix.at<double>(0, 2) = -BoundingRect.x;
        rotationMatrix.at<double>(1, 2) = -BoundingRect.y;
        // rotated image
        warpAffine(matSrc, object.image, rotationMatrix, Size(BoundingRect.width, BoundingRect.height));
        // transform vector
        vector<Point2f> transformBoundingBox;
        minRectArea.points(transformBoundingBox);
        transform(transformBoundingBox, transformBoundingBox, rotationMatrix);
        BoundingRect = boundingRect(transformBoundingBox);

        object.image = cropImageWithBorderOffset(object.image, BoundingRect, 2);

        // get center off PCA for transform pickposition
        vector<Point2f> transformCenterPoint;
        transformCenterPoint.push_back(center);
        transform(transformCenterPoint, transformCenterPoint, rotationMatrix);
        Point2f topleft = BoundingRect.tl();
        object.centerCrop = transformCenterPoint[0] - topleft;
    }

    bool GeoMatch::matchingScores(RotatedObj& objectRotated, GeoModel& model, double& lastMaxScores) {

        Mat imgDest;
        Mat gx;
        Mat gy;
        Mat magnitude;
        Mat angle;
        Point2f center;

        const int centerTolerantRow = (int)(objectRotated.image.rows * 0.02);
        const int centerTolerantCol = (int)(objectRotated.image.cols * 0.02);

        cvtColor(objectRotated.image, imgDest, COLOR_RGB2GRAY);
        Sobel(imgDest, gx, CV_64F, 1, 0, 3);
        Sobel(imgDest, gy, CV_64F, 0, 1, 3);
        cartToPolar(gx, gy, magnitude, angle);

        vector<GeoModel::ModelPattern> modelPattern = model.getPatternOfModel();

        // ncc match search
        long noOfCordinates = modelPattern.size();
        // normalized min score
        double normMinScore = model.minScores / noOfCordinates;
        double normGreediness = ((1 - model.greediness * model.minScores) / (1 - model.greediness)) / noOfCordinates;
        double partialScore = 0;
        double resultScore = 0;

        Point2f offset = model.getDistanceCenterPattern();
        int startRowIdx = objectRotated.centerCrop.y + offset.y - centerTolerantRow;
        int endRowIdx = objectRotated.centerCrop.y + offset.y + centerTolerantRow;
        int startColIdx = objectRotated.centerCrop.x + offset.x - centerTolerantCol;
        int endColIdx = objectRotated.centerCrop.x + offset.x + centerTolerantCol;

        for (int rowIdx = startRowIdx; rowIdx < endRowIdx; rowIdx++)
        {
            for (int colIdx = startColIdx; colIdx < endColIdx; colIdx++)
            {
                double partialSum = 0;
                for (int count = 0; count < noOfCordinates; count++)
                {
                    GeoModel::ModelPattern tempPoint = modelPattern[count];

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

                    double minBreakScores = std::min((model.minScores - 1) + normGreediness * sumOfCoords, normMinScore * sumOfCoords);
                    if (partialScore < minBreakScores) {
                        break;
                    }

                    if (partialScore < lastMaxScores) {
                        break;
                    }
                }

                if (partialScore > resultScore) {
                    resultScore = partialScore;
                    center.x = colIdx;
                    center.y = rowIdx;
                }
            }
        }

        if (resultScore > model.minScores) {
            if (resultScore > lastMaxScores) {
                lastMaxScores = resultScore;
                objectRotated.centerMaxScores = center;
                return true;
            }
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

    void GeoMatch::drawPickingBox(Mat& matSrc, RotatedRect rectRot, Scalar color) {
        cv::Point2f vertices2f[4];
        rectRot.points(vertices2f);

        cv::Point vertices[4];
        for (int i = 0; i < 4; ++i) {
            vertices[i] = vertices2f[i];
        }

        line(matSrc, vertices[0], vertices[1], color, 1, LineTypes::LINE_AA);
        line(matSrc, vertices[1], vertices[2], color, 1, LineTypes::LINE_AA);
        line(matSrc, vertices[2], vertices[3], color, 1, LineTypes::LINE_AA);
        line(matSrc, vertices[3], vertices[0], color, 1, LineTypes::LINE_AA);
    }

