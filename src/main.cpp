/*
  Copyright 2019 <BrownieAlice>
*/

#include <opencv2/opencv.hpp>
#include <cmath>

constexpr int minDisparity = 32;
constexpr int numDisparities = 112;
constexpr int SADWindowSize = 16;
constexpr int P1 = 8 * 3 * 5;
constexpr int P2 = 32 * 3 * 5;
constexpr int disp12MaxDiff = 20;
constexpr int preFilterCap = 0;
constexpr int uniquenessRatio = 10;
constexpr int speckleWindowSize = 100;
constexpr int speckleRange = 32;
constexpr bool fullDp = true;
cv::Ptr<cv::StereoSGBM> ssgbm = cv::StereoSGBM::create(
  minDisparity, numDisparities, SADWindowSize, P1, P2,
  disp12MaxDiff, preFilterCap, uniquenessRatio,
  speckleWindowSize, speckleRange, fullDp);

int main() {
  cv::VideoCapture cap1(0);
  /*
    cap1.set(cv::CAP_PROP_FOURCC,
      cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    cap1.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap1.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
  */
  cap1.set(cv::CAP_PROP_FPS, 30);
  cap1.set(cv::CAP_PROP_FRAME_WIDTH, 640);
  cap1.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
  cap1.set(cv::CAP_PROP_EXPOSURE, 0.0);

  cv::VideoCapture cap2(3);
  /*
    cap2.set(cv::CAP_PROP_FOURCC,
      cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    cap2.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap2.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
  */
  cap2.set(cv::CAP_PROP_FPS, 30);
  cap2.set(cv::CAP_PROP_FRAME_WIDTH, 640);
  cap2.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
  cap2.set(cv::CAP_PROP_EXPOSURE, 0.0);

  if (!cap1.isOpened()) {
    std::cout << "canot open cap1" << std::endl;
    return -1;
  }

  if (!cap2.isOpened()) {
    std::cout << "canot open cap2" << std::endl;
    return -1;
  }

  cv::Mat frame1, frame2, calc_depth, depth;
  // 取得したフレーム
  while (true) {
    cap1 >> frame1;
    cap2 >> frame2;
    calc_depth = cv::Mat(frame1.rows, frame1.cols, CV_16S);
    depth = cv::Mat(frame1.rows, frame1.cols, CV_16S);

    cv::imshow("Camera Left", frame1);
    cv::imshow("Camera Right", frame2);
    //画像を表示．

    cv::cvtColor(frame1, frame1, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(frame1, frame1);
    cv::GaussianBlur(frame1, frame1, cv::Size(5, 5), 0);
    cv::cvtColor(frame2, frame2, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(frame2, frame2);
    cv::GaussianBlur(frame2, frame2, cv::Size(5, 5), 0);

    ssgbm->compute(frame1, frame2, calc_depth);
    double min, max;
    cv::minMaxLoc(calc_depth, &min, &max);
    calc_depth.convertTo(depth, CV_8UC1, 255.0 / (max - min),
      -255.0 * min / (max - min));
    cv::imshow("result", depth);

    cv::waitKey(1);
  }
  cv::destroyAllWindows();
  return 0;
}
