#ifndef CAFFE_UNPOOLING_LAYER_HPP_
#define CAFFE_UNPOOLING_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief UnPools the input image by assigning fixed, bilinear interpolation,
 * etc. within regions.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class UnPoolingLayer : public Layer<Dtype> {
 public:
  explicit UnPoolingLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "UnPooling"; }
  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MaxBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  // fill mask for different unpool type
  void FillMask();

  int out_kernel_h_, out_kernel_w_;
  int out_stride_h_, out_stride_w_;
  int out_pad_h_, out_pad_w_;
  int num_, channels_;
  int height_, width_;
  int unpooled_height_, unpooled_width_;
  Blob<int> mask_;
};

}  // namespace caffe

#endif  // CAFFE_UNPOOLING_LAYER_HPP_
