#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/parse_evaluate_layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ParseEvaluateLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const ParseEvaluateParameter& parse_evaluate_param =
      this->layer_param_.parse_evaluate_param();
  CHECK(parse_evaluate_param.has_num_labels()) << "Must have num_labels!!";
  num_labels_ = parse_evaluate_param.num_labels();
  ignore_labels_.clear();
  int num_ignore_label = parse_evaluate_param.ignore_label().size();
  for (int i = 0; i < num_ignore_label; ++i) {
    ignore_labels_.insert(parse_evaluate_param.ignore_label(i));
  }
}

template <typename Dtype>
void ParseEvaluateLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
      << "The data and label should have the same number.";
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->channels(), 1);
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_GE(bottom[0]->width(), bottom[1]->width());
  top[0]->Reshape(1, num_labels_, 1, 3);
}

template <typename Dtype>
void ParseEvaluateLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                            const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  CHECK_EQ(bottom[0]->count(), bottom[1]->count());
  const Dtype* bottom_pred = bottom[0]->cpu_data();
  const Dtype* bottom_gt = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_set(top[0]->count(), Dtype(0), top_data);
  int num = bottom[0]->num();
  int spatial_dim = bottom[0]->height() * bottom[0]->width();
  for (int i = 0; i < num; ++i) {
    // count the number of ground truth labels, the predicted labels, and
    // predicted labels happens to be ground truth labels
    for (int j = 0; j < spatial_dim; ++j) {
      int gt_label = bottom_gt[j];
      int pred_label = bottom_pred[j];
      CHECK_LT(pred_label, num_labels_);
      if (ignore_labels_.find(gt_label) != ignore_labels_.end()) {
        continue;
      }
      if (gt_label == pred_label) {
        top_data[gt_label * 3]++;
      }
      top_data[gt_label * 3 + 1]++;
      top_data[pred_label * 3 + 2]++;
    }
    bottom_pred += bottom[0]->offset(1);
    bottom_gt += bottom[1]->offset(1);
  }
  // ParseEvaluate layer should not be used as a loss function.
}

INSTANTIATE_CLASS(ParseEvaluateLayer);
REGISTER_LAYER_CLASS(ParseEvaluate);

}  // namespace caffe
