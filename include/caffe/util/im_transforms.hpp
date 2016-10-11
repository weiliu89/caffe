#ifdef USE_OPENCV
#ifndef IM_TRANSFORMS_HPP
#define IM_TRANSFORMS_HPP

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe_pb.h"

namespace caffe {

// Generate random number given the probablities for each number.
DLL_EXPORT int roll_weighted_die(const std::vector<float>& probabilities);

template <typename T>
DLL_EXPORT bool is_border(const cv::Mat& edge, T color);

// Auto cropping image.
template <typename T>
DLL_EXPORT cv::Rect CropMask(const cv::Mat& src, T point, int padding = 2);

DLL_EXPORT cv::Mat colorReduce(const cv::Mat& image, int div = 64);

DLL_EXPORT void fillEdgeImage(const cv::Mat& edgesIn, cv::Mat* filledEdgesOut);

DLL_EXPORT void CenterObjectAndFillBg(const cv::Mat& in_img, const bool fill_bg,
                           cv::Mat* out_img);

DLL_EXPORT cv::Mat AspectKeepingResizeAndPad(const cv::Mat& in_img,
                                  const int new_width, const int new_height,
                                  const int pad_type = cv::BORDER_CONSTANT,
                                  const cv::Scalar pad = cv::Scalar(0, 0, 0),
                                  const int interp_mode = cv::INTER_LINEAR);

DLL_EXPORT cv::Mat AspectKeepingResizeBySmall(const cv::Mat& in_img,
                                   const int new_width, const int new_height,
                                   const int interp_mode = cv::INTER_LINEAR);

DLL_EXPORT void constantNoise(const int n, const vector<uchar>& val, cv::Mat* image);

DLL_EXPORT void UpdateBBoxByResizePolicy(const ResizeParameter& param,
                              const int old_width, const int old_height,
                              NormalizedBBox* bbox);

DLL_EXPORT cv::Mat ApplyResize(const cv::Mat& in_img, const ResizeParameter& param);

DLL_EXPORT cv::Mat ApplyNoise(const cv::Mat& in_img, const NoiseParameter& param);

}  // namespace caffe

#endif  // IM_TRANSFORMS_HPP
#endif  // USE_OPENCV
