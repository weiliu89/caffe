#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/parse_output_layer.hpp"

namespace caffe {

template <typename Dtype>
void ParseOutputLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  out_max_val_ = top.size() > 1;
}

template <typename Dtype>
void ParseOutputLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Produces max_ind and max_val
  top[0]->Reshape(bottom[0]->num(), 1, bottom[0]->height(), bottom[0]->width());
  if (out_max_val_) {
    top[1]->ReshapeLike(*top[0]);
  }
  max_prob_.Reshape(1, 1, bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
void ParseOutputLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_label_data = top[0]->mutable_cpu_data();
  Dtype* top_prob_data = NULL;
  if (out_max_val_) {
    top_prob_data = top[1]->mutable_cpu_data();
  }
  Dtype* max_prob_data = max_prob_.mutable_cpu_data();
  int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int spatial_dim = bottom[0]->height() * bottom[0]->width();
  for (int i = 0; i < num; ++i) {
    caffe_set(spatial_dim, Dtype(0), top_label_data);
    // initialize max value from first plane
    caffe_copy(spatial_dim, bottom_data, max_prob_data);
    bottom_data += bottom[0]->offset(0, 1);
    for (int j = 1; j < channels; ++j) {
      for (int k = 0; k < spatial_dim; ++k) {
        Dtype prob = bottom_data[k];
        if (prob > max_prob_data[k]) {
          max_prob_data[k] = prob;
          top_label_data[k] = j;
        }
      }
      bottom_data += bottom[0]->offset(0, 1);
    }
    top_label_data += top[0]->offset(1);
    if (out_max_val_) {
      caffe_copy(spatial_dim, max_prob_data, top_prob_data);
      top_prob_data += top[1]->offset(1);
    }
  }
}

INSTANTIATE_CLASS(ParseOutputLayer);
REGISTER_LAYER_CLASS(ParseOutput);

}  // namespace caffe
