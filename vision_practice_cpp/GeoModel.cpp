#include "GeoModel.h"


    GeoModel::GeoModel() {
        InitModelParameter();
    }

    GeoModel::GeoModel(string path, string modelName) {
        InitModelParameter();
        setImageOfModel(path);
        nameOfModel = modelName;
    }

    GeoModel::~GeoModel() {

    }

    void GeoModel::InitModelParameter() {
        cannyThreshLower = INIT_CANNY_THRESH_1;
        cannyThreshUpper = INIT_CANNY_THRESH_2;
        cannyRange = INIT_CANNY_RANGE;
        cannyContrast = INIT_CANNY_CONTRAST;
        cannyKernelSize = INIT_CANNY_KERNEL_SIZE;
        minScores = INIT_MIN_SCORES;
        greediness = INIT_GREEDINESS;
        contoursAreaTolerance = INIT_CONTOURS_TOLERANCE;
        geometricLearned = false;
    }

    void GeoModel::setImageOfModel(string modelPath) {
        if (modelPath.empty()) {
            return;
        }

        // read image from file
        imageOfMOdel = imread(modelPath);
        if (imageOfMOdel.empty()) {
            return;
        }

        // get file name from model path
        size_t splitIndex = modelPath.find_last_of('/');
        if (splitIndex == String::npos) {
            return;
        }
        nameOfFile = modelPath.substr(splitIndex + 1, modelPath.length() - splitIndex);

        // init pick position in center of image
        pickPosition.x = imageOfMOdel.cols / 2.0;
        pickPosition.y = imageOfMOdel.rows / 2.0;
        return;
    }

    void GeoModel::modelLearnPattern() {
        if (imageOfMOdel.empty()) {
            return;
        }

        geometricLearned = false;
        patternOfModel.clear();
        contoursOfModel.clear();
        hierarchyOfContours.clear();

        Mat imgGray;
        Mat imgOutput;
        Mat gx, gy;
        Mat magnitude, angle;
        Point2d sumPoint = Point2d(0, 0);

        //// tiền xử lý
        cvtColor(imageOfMOdel, imgGray, COLOR_RGB2GRAY);
        GaussianBlur(imgGray, imgGray, cv::Size(3, 3), 0);

        // Threshold with canny method
        double medianValue = median(imgGray);
        double sigma = (0.1 / (double)CANNY_RANGE_MAX) * cannyRange;
        cannyThreshLower = std::max(0.0, ((1.0 - sigma) * medianValue));
        cannyThreshUpper = std::min(255.0, ((1.0 + sigma) * medianValue));
        Canny(imgGray, imgOutput, cannyThreshLower, cannyThreshUpper,
            cannyKernelSize);
        // find contours
        findContours(imgOutput, contoursOfModel, hierarchyOfContours,
            cv::RETR_TREE, cv::CHAIN_APPROX_NONE);


        // gx,gy , mag 
        Sobel(imgGray, gx, CV_64F, 1, 0, 3);
        Sobel(imgGray, gy, CV_64F, 0, 1, 3);
        cartToPolar(gx, gy, magnitude, angle);

        // save pattern info
        for (int contourIdx = 0; contourIdx < contoursOfModel.size(); contourIdx++) {
            for (int pointInx = 0; pointInx < contoursOfModel[contourIdx].size(); pointInx++)
            {
                ModelPattern tempData;
                double mag;
                tempData.Coordinates = contoursOfModel[contourIdx][pointInx];
                tempData.Derivative = Point2d(gx.at<double>(tempData.Coordinates),
                    gy.at<double>(tempData.Coordinates));
                tempData.Angle = angle.at<double>(tempData.Coordinates);
                mag = magnitude.at<double>(tempData.Coordinates);
                tempData.Magnitude = (mag == 0) ? 0 : (1 / mag);
                // push to container
                patternOfModel.push_back(tempData);

                sumPoint += tempData.Coordinates;
            }
        }

        // tính trung bình của contours template
        Point2f templateCenterPoint(sumPoint.x / patternOfModel.size(),
            sumPoint.y / patternOfModel.size());
        for (ModelPattern& pointTemp : patternOfModel) {  // cập nhât center và offset
            pointTemp.Center = templateCenterPoint;
            pointTemp.Offset = pointTemp.Coordinates - pointTemp.Center;
        }

        //// ?????????????? target
        Mat threshImage;
        vector<vector<Point>> pcaContours;
        vector<Vec4i> pcaHierarchy;
        threshold(imgGray, threshImage, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);// otsu để xác định ngưỡng cho nhiều trường hợp
        findContours(threshImage, pcaContours, pcaHierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);// tìm contours lại
        int limitArea = ((double)(imgGray.cols * imgGray.rows) * 0.98);// ngướng để loại bỏ contours lớn nhất quanh img khi có kết hợp với otsu
        int maxArea = 0;
        for (int conCounter = 0; conCounter < pcaContours.size(); conCounter++) {
            int area = contourArea(pcaContours[conCounter]);
            if (area >= limitArea) { // loại bỏ contours bao
                continue;
            }

            if (area > maxArea) {  // lấy contour to nhất còn lại trong template
                maxArea = area;
                contoursSelectIndex = conCounter;
                contoursSelectArea = maxArea;
            }
        }

        computePcaOrientation(pcaContours[contoursSelectIndex], rawPcaAngle, centerOfPCA);// hướng của template
        geometricLearned = true;
        showInfoImage(nameOfModel, templateCenterPoint);
    }

    bool GeoModel::isImageEmpty() {
        return imageOfMOdel.empty();
    }

    bool GeoModel::isGeometricLearned() {
        return geometricLearned;
    }

    string GeoModel::getModelFileName() {
        return nameOfFile;
    }

    Mat GeoModel::getImageOfModel() {
        return imageOfMOdel.clone();
    }

    int GeoModel::getImageOfModelCols() {
        return imageOfMOdel.cols;
    }

    int GeoModel::getImageOfModelRows() {
        return imageOfMOdel.rows;
    }

    Mat GeoModel::getImageOfModelWithPickPosition() {
        cv::Mat newImage = imageOfMOdel.clone();
        drawPcaAxis(newImage, pickPosition, angleOfModel, 20);
        return newImage;
    }

    Mat GeoModel::getImageOfModelCannyThreshold() {  // get canny image

        cv::Mat cannyImage = imageOfMOdel.clone();
        cvtColor(cannyImage, cannyImage, COLOR_RGB2GRAY);
        GaussianBlur(cannyImage, cannyImage, cv::Size(3, 3), 0);
        // tiền xử lý


        double medianValue = median(cannyImage);
        double sigma = (0.1 / (double)CANNY_RANGE_MAX) * cannyRange;
        cannyThreshLower = std::max(0.0, ((1.0 - sigma) * medianValue));
        cannyThreshUpper = std::min(255.0, ((1.0 + sigma) * medianValue));
        // Threshold with canny method
        Canny(cannyImage, cannyImage, cannyThreshLower, cannyThreshUpper,
            cannyKernelSize);
        return cannyImage;
    }

    Mat GeoModel::getImageOfModelCannyThreshold(Mat image) { // có input

        cv::Mat cannyImage = imageOfMOdel.clone();
        cvtColor(cannyImage, cannyImage, COLOR_RGB2GRAY);
        GaussianBlur(cannyImage, cannyImage, cv::Size(3, 3), 0);
        // calculate threshold value
        double medianValue = median(cannyImage);
        double sigma = (0.1 / (double)CANNY_RANGE_MAX) * cannyRange;
        cannyThreshLower = std::max(0.0, ((1.0 - sigma) * medianValue));
        cannyThreshUpper = std::min(255.0, ((1.0 + sigma) * medianValue));
        // Threshold with canny method
        Canny(cannyImage, cannyImage, cannyThreshLower, cannyThreshUpper,
            cannyKernelSize, true);
        return cannyImage;
    }



    vector<vector<Point>> GeoModel::getModelContours() {
        return contoursOfModel;
    }

    vector<Vec4i> GeoModel::getModelContoursHierarchy() {
        return hierarchyOfContours;
    }

    int GeoModel::getModelContoursSelectedIndex() {
        return contoursSelectIndex;
    }

    int GeoModel::getModelContoursSelectedArea() {
        return contoursSelectArea;
    }

    void GeoModel::getModelThreshAreaTolerance(int& high, int& low) {
        high = contoursSelectArea + contoursAreaTolerance;
        low = contoursSelectArea - contoursAreaTolerance;
    }

    double GeoModel::getModelRawPcaAngle(bool radians) {
        if (radians) {
            return rawPcaAngle;
        }

        return rawPcaAngle * R2D;
    }

    Point2f GeoModel::getModelCenterPCA() {
        return centerOfPCA;
    }

    Point2f GeoModel::getDistanceCenterPattern() {
        return Point2f((patternOfModel[0].Center.x - centerOfPCA.x), (patternOfModel[0].Center.y - centerOfPCA.y));
    }

    void GeoModel::setAngleOfModel(double value, bool radians) {
        angleOfModel = value;
        if (!radians) {
            angleOfModel *= D2R;
        }
    }

    double GeoModel::getAngleOfModel(bool radians) {
        if (radians) {
            return angleOfModel;
        }
        return angleOfModel * R2D;
    }

    vector<GeoModel::ModelPattern> GeoModel::getPatternOfModel() {
        return patternOfModel;
    }

    void GeoModel::computePcaOrientation(const vector<Point>& pts, double& angleOutput, Point2f& centerOutput) {
        // Construct a buffer used by the pca analysis
        int sz = static_cast<int>(pts.size());
        Mat data_pts = Mat(sz, 2, CV_64F);
        for (int i = 0; i < data_pts.rows; i++)
        {
            data_pts.at<double>(i, 0) = pts[i].x;
            data_pts.at<double>(i, 1) = pts[i].y;
        }
        // Perform PCA analysis
        PCA pca_analysis(data_pts, Mat(), PCA::DATA_AS_ROW);
        // Store the center of the object
        Point cntr = Point(static_cast<int>(pca_analysis.mean.at<double>(0, 0)),
            static_cast<int>(pca_analysis.mean.at<double>(0, 1)));
        // Store the eigenvalues and eigenvectors
        vector<Point2d> eigen_vecs(2);
        vector<double> eigen_val(2);
        for (int i = 0; i < 2; i++)
        {
            eigen_vecs[i] = Point2d(pca_analysis.eigenvectors.at<double>(i, 0),
                pca_analysis.eigenvectors.at<double>(i, 1));
            eigen_val[i] = pca_analysis.eigenvalues.at<double>(i);
        }

        double angle = atan2(eigen_vecs[0].y, eigen_vecs[0].x); // orientation in radians
        angleOutput = angle;
        centerOutput = cntr;
    }

    void GeoModel::drawPcaAxis(Mat& img, Point center, double angle, int length,
        Scalar xColour, Scalar yColour, Scalar centerColour) {
        const double yAxisRatio = 0.8;
        // create the axis
        Point xPoint((int)(center.x + cos(angle) * length), (int)(center.y + sin(angle) * length));
        Point yPoint((int)(center.x - sin(angle) * length * yAxisRatio), (int)(center.y + cos(angle) * length * yAxisRatio));
        line(img, center, xPoint, xColour, 1, LINE_AA, 0);
        line(img, center, yPoint, yColour, 1, LINE_AA, 0);
        circle(img, center, 2, centerColour, 2);
    }

    Mat GeoModel::adjustContrast(cv::Mat image, double alpha) {
        cv::Mat new_image = cv::Mat::zeros(image.size(), image.type());
        // double beta = 1.0;
        for (int y = 0; y < image.rows; y++) {
            for (int x = 0; x < image.cols; x++) {
                for (int c = 0; c < image.channels(); c++) {
                    new_image.at<Vec3b>(y, x)[c] =
                        saturate_cast<uchar>(alpha * image.at<Vec3b>(y, x)[c] /*+ beta*/);
                }
            }
        }
        return new_image;
    }

    double GeoModel::median(cv::Mat Input) { // giá trị trung bình cảu ảnh theo histogram

        double m = (Input.rows * Input.cols) / 2;
        int bin = 0;
        double med = -1.0;

        int histSize = 256;
        float range[] = { 0, 256 };
        const float* histRange = { range };
        bool uniform = true;
        bool accumulate = false;
        cv::Mat hist;
        cv::calcHist(&Input, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);

        for (int i = 0; i < histSize && med < 0.0; ++i)
        {
            bin += cvRound(hist.at< float >(i));
            if (bin > m && med < 0.0)
                med = i;
        }

        return med;
    }

    void GeoModel::showInfoImage(String windowName, Point2f center_of_temp)
    {
        if (windowName.empty()) {
            return;
        }

        Mat showImg;
        // avoid memory optimize of compiler
        showImg = imageOfMOdel.clone();

        // draw all contours
        drawContours(showImg, contoursOfModel, -1, Scalar(255, 0, 0), 1);

        // draw pca center and direction
        circle(showImg, centerOfPCA, 3, Scalar(255, 0, 255), 2);
        circle(showImg, center_of_temp, 3, Scalar(0, 255, 255), 2);


        drawPcaAxis(showImg, centerOfPCA, rawPcaAngle,100);
      /*  drawPcaAxis(showImg, centerOfPCA,angel , Scalar(255, 255, 0));*/

        imshow(windowName, showImg);
    }

