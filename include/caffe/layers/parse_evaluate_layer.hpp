#ifndef CAFFE_PARSE_EVALUATE_LAYER_HPP_
#define CAFFE_PARSE_EVALUATE_LAYER_HPP_

#include <set>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Count the prediction and ground truth statistics for each datum.
 *
 * NOTE: This does not implement Backwards operation.
 */
template <typename Dtype>
class ParseEvaluateLayer : public Layer<Dtype> {
 public:
  /**
   * @param param provides ParseEvaluateParameter parse_evaluate_param,
   *     with ParseEvaluateLayer options:
   *   - num_labels (\b optional int32.).
   *     number of labels. must provide!!
   *   - ignore_label (\b repeated int32).
   *     If any, ignore evaluating the corresponding label for each
   *     image.
   */
  explicit ParseEvaluateLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ParseEvaluate"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  /**
   * @param bottom input Blob vector (length 2)
   *   -# @f$ (N \times 1 \times H \times W) @f$
   *      the prediction label @f$ x @f$
   *   -# @f$ (N \times 1 \times H \times W) @f$
   *      the ground truth label @f$ x @f$
   * @param top output Blob vector (length 1)
   *   -# @f$ (N \times C \times 1 \times 3) @f$
   *      the counts for different class @f$
   */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  /// @brief Not implemented (non-differentiable function)
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
  }

  // number of total labels
  int num_labels_;
  // store ignored labels
  std::set<Dtype> ignore_labels_;
};

}  // namespace caffe

#endif  // CAFFE_PARSE_EVALUATE_LAYER_HPP_
