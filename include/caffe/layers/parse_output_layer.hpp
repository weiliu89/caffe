#ifndef CAFFE_PARSE_OUTPUT_LAYER_HPP_
#define CAFFE_PARSE_OUTPUT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Compute the segmentation label of the @f$ H \times W @f$ for each datum across
 *        all channels @f$ C @f$.
 *
 * Intended for use after a classification layer to produce a prediction of
 * segmentation label.
 * If parameter out_max_val is set to true, also output the predicted value for
 * the corresponding label for each image.
 *
 * NOTE: does not implement Backwards operation.
 */
template <typename Dtype>
class ParseOutputLayer : public Layer<Dtype> {
 public:
  /**
   * @param param provides ParseOutputParameter parse_output_param,
   *     with ParseOutputLayer options:
   *   - out_max_val (\b optional bool, default false).
   *     if set, output the predicted value for the corresponding label for each
   *     image.
   */
  explicit ParseOutputLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ParseOutput"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

 protected:
  /**
   * @param bottom input Blob vector (length 1)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the inputs @f$ x @f$
   * @param top output Blob vector (length 1)
   *   -# @f$ (N \times 1 \times H \times W) @f$ or, if out_max_val
   *      @f$ (N \times 2 \times H \times W) @f$
   *      the computed outputs @f$
   */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  /// @brief Not implemented (non-differentiable function)
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
  }

  bool out_max_val_;

  // max_prob_ is used to store the maximum probability value
  Blob<Dtype> max_prob_;
};


}  // namespace caffe

#endif  // CAFFE_PARSE_OUTPUT_LAYER_HPP_
