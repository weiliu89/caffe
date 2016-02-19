#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/unpooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void FixedUnPoolForward(const int nthreads, const Dtype* bottom_data,
    const int num, const int channels, const int height, const int width,
    const int unpooled_height, const int unpooled_width, const int out_kernel_h,
    const int out_kernel_w, const int out_stride_h, const int out_stride_w,
    const int out_pad_h, const int out_pad_w, Dtype* top_data) {
  CUDA_KERNEL_LOOP(unpool_index, nthreads) {
    int uw = unpool_index % unpooled_width;
    int uh = (unpool_index / unpooled_width) % unpooled_height;
    int c = (unpool_index / unpooled_width / unpooled_height) % channels;
    int n = unpool_index / unpooled_width / unpooled_height / channels;
    int hstart = (uh + out_pad_h < out_kernel_h) ? 0 :
      (uh + out_pad_h - out_kernel_h) / out_stride_h + 1;
    int hend = min((uh + out_pad_h) / out_stride_h + 1, height);
    int wstart = (uw + out_pad_w < out_kernel_w) ? 0 :
      (uw + out_pad_w - out_kernel_w) / out_stride_w + 1;
    int wend = min((uw + out_pad_w) / out_stride_w + 1, width);
    int offset = (n * channels + c) * height * width;
    int unpool_offset = (n * channels + c) * unpooled_height * unpooled_width;
    bottom_data += offset;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        int uhstart = h * out_stride_h - out_pad_h;
        int uwstart = w * out_stride_w - out_pad_w;
        int uhend = uhstart + out_kernel_h;
        int uwend = uwstart + out_kernel_w;
        int uhmid = (uhstart + uhend - 1) / 2;
        int uwmid = (uwstart + uwend - 1) / 2;
        uhmid = min(max(uhmid, 0), unpooled_height);
        uwmid = min(max(uwmid, 0), unpooled_width);
        if (unpool_offset + uhmid * unpooled_width + uwmid == unpool_index) {
          // find the mapping, assign & return
          int index = h * width + w;
          top_data[unpool_index] = bottom_data[index];
          return;
        }
      }
    }
  }
}

template <typename Dtype>
__global__ void DivUnPoolForward(const int nthreads, const Dtype* bottom_data,
    const int* mask, const int num, const int channels, const int height,
    const int width, const int unpooled_height, const int unpooled_width,
    const int out_kernel_h, const int out_kernel_w, const int out_stride_h,
    const int out_stride_w, const int out_pad_h, const int out_pad_w,
    Dtype* top_data) {
  CUDA_KERNEL_LOOP(unpool_index, nthreads) {
    int uw = unpool_index % unpooled_width + out_pad_w;
    int uh = (unpool_index / unpooled_width) % unpooled_height + out_pad_h;
    int c = (unpool_index / unpooled_width / unpooled_height) % channels;
    int n = unpool_index / unpooled_width / unpooled_height / channels;
    int spatial_dim = unpooled_height * unpooled_width;
    int hstart = (uh < out_kernel_h) ? 0 :
      (uh - out_kernel_h) / out_stride_h + 1;
    int hend = min(uh / out_stride_h + 1, height);
    int wstart = (uw < out_kernel_w) ? 0 :
      (uw - out_kernel_w) / out_stride_w + 1;
    int wend = min(uw / out_stride_w + 1, width);
    Dtype divval = 0;
    bottom_data += (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        int uhstart = h * out_stride_h - out_pad_h;
        int uwstart = w * out_stride_w - out_pad_w;
        int uhend = min(uhstart + out_kernel_h, unpooled_height + out_pad_h);
        int uwend = min(uwstart + out_kernel_w, unpooled_width + out_pad_w);
        int unpool_size = (uhend - uhstart) * (uwend - uwstart);
        divval += bottom_data[h * width + w] / unpool_size;
      }
    }
    top_data[unpool_index] = divval / mask[unpool_index % spatial_dim];
  }
}

template <typename Dtype>
__global__ void RepUnPoolForward(const int nthreads, const Dtype* bottom_data,
    const int* mask, const int num, const int channels, const int height,
    const int width, const int unpooled_height, const int unpooled_width,
    const int out_kernel_h, const int out_kernel_w, const int out_stride_h,
    const int out_stride_w, const int out_pad_h, const int out_pad_w,
    Dtype* top_data) {
  CUDA_KERNEL_LOOP(unpool_index, nthreads) {
    int uw = unpool_index % unpooled_width + out_pad_w;
    int uh = (unpool_index / unpooled_width) % unpooled_height + out_pad_h;
    int c = (unpool_index / unpooled_width / unpooled_height) % channels;
    int n = unpool_index / unpooled_width / unpooled_height / channels;
    int spatial_dim = unpooled_height * unpooled_width;
    int hstart = (uh < out_kernel_h) ? 0 :
      (uh - out_kernel_h) / out_stride_h + 1;
    int hend = min(uh / out_stride_h + 1, height);
    int wstart = (uw < out_kernel_w) ? 0 :
      (uw - out_kernel_w) / out_stride_w + 1;
    int wend = min(uw / out_stride_w + 1, width);
    Dtype val = 0;
    bottom_data += (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        int uhstart = h * out_stride_h - out_pad_h;
        int uwstart = w * out_stride_w - out_pad_w;
        int uhend = min(uhstart + out_kernel_h, unpooled_height + out_pad_h);
        int uwend = min(uwstart + out_kernel_w, unpooled_width + out_pad_w);
        val += bottom_data[h * width + w];
      }
    }
    top_data[unpool_index] = val / mask[unpool_index % spatial_dim];
  }
}

template <typename Dtype>
void UnPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = top[0]->count();
  caffe_gpu_set(count, Dtype(0), top_data);
  const int* mask = mask_.gpu_data();
  switch (this->layer_param_.unpooling_param().unpool()) {
  case UnPoolingParameter_UnPoolMethod_FIXED:
    FixedUnPoolForward<Dtype>
        // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, num_, channels_, height_, width_,
        unpooled_height_, unpooled_width_, out_kernel_h_, out_kernel_w_,
        out_stride_h_, out_stride_w_, out_pad_h_, out_pad_w_, top_data);
    break;
  case UnPoolingParameter_UnPoolMethod_DIV:
    DivUnPoolForward<Dtype>
        // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, mask, num_, channels_, height_, width_,
        unpooled_height_, unpooled_width_, out_kernel_h_, out_kernel_w_,
        out_stride_h_, out_stride_w_, out_pad_h_, out_pad_w_, top_data);
    break;
  case UnPoolingParameter_UnPoolMethod_REP:
    RepUnPoolForward<Dtype>
        // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, mask, num_, channels_, height_, width_,
        unpooled_height_, unpooled_width_, out_kernel_h_, out_kernel_w_,
        out_stride_h_, out_stride_w_, out_pad_h_, out_pad_w_, top_data);
    break;
  default:
    LOG(FATAL) << "Unknown unpooling method.";
  }
  CUDA_POST_KERNEL_CHECK;
}


template <typename Dtype>
__global__ void FixedUnPoolBackward(const int nthreads, const Dtype* top_diff,
    const int* mask, const int num, const int channels, const int height,
    const int width, const int unpooled_height, const int unpooled_width,
    const int out_kernel_h, const int out_kernel_w, const int out_stride_h,
    const int out_stride_w, const int out_pad_h, const int out_pad_w,
    Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;
    int uhstart = h * out_stride_h - out_pad_h;
    int uwstart = w * out_stride_w - out_pad_w;
    int uhend = uhstart + out_kernel_h;
    int uwend = uwstart + out_kernel_w;
    int uhmid = (uhstart + uhend - 1) / 2;
    int uwmid = (uwstart + uwend - 1) / 2;
    uhmid = min(max(uhmid, 0), unpooled_height-1);
    uwmid = min(max(uwmid, 0), unpooled_width-1);
    int offset = (n * channels + c) * unpooled_height * unpooled_width;
    int unpool_index = uhmid * unpooled_width + uwmid;
    Dtype gradient = 0;
    if (mask[unpool_index] == h * width + w) {
      gradient += top_diff[unpool_index + offset];
    }
    bottom_diff[index] = gradient;
  }
}

template <typename Dtype>
__global__ void DivUnPoolBackward(const int nthreads, const Dtype* top_diff,
    const int* mask, const int num, const int channels, const int height,
    const int width, const int unpooled_height, const int unpooled_width,
    const int out_kernel_h, const int out_kernel_w, const int out_stride_h,
    const int out_stride_w, const int out_pad_h, const int out_pad_w,
    Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;
    int uhstart = h * out_stride_h - out_pad_h;
    int uwstart = w * out_stride_w - out_pad_w;
    int uhend = min(uhstart + out_kernel_h, unpooled_height + out_pad_h);
    int uwend = min(uwstart + out_kernel_w, unpooled_width + out_pad_w);
    int unpool_size = (uhend - uhstart) * (uwend - uwstart);
    uhstart = max(uhstart, 0);
    uwstart = max(uwstart, 0);
    uhend = min(uhend, unpooled_height);
    uwend = min(uwend, unpooled_width);
    Dtype gradient = 0;
    int offset = (n * channels + c) * unpooled_height * unpooled_width;
    for (int uh = uhstart; uh < uhend; ++uh) {
      for (int uw = uwstart; uw < uwend; ++uw) {
        int unpool_index = uh * unpooled_width + uw;
        gradient += top_diff[unpool_index + offset] / mask[unpool_index];
      }
    }
    bottom_diff[index] = gradient / unpool_size;
  }
}

template <typename Dtype>
__global__ void RepUnPoolBackward(const int nthreads, const Dtype* top_diff,
    const int* mask, const int num, const int channels, const int height,
    const int width, const int unpooled_height, const int unpooled_width,
    const int out_kernel_h, const int out_kernel_w, const int out_stride_h,
    const int out_stride_w, const int out_pad_h, const int out_pad_w,
    Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;
    int uhstart = h * out_stride_h - out_pad_h;
    int uwstart = w * out_stride_w - out_pad_w;
    int uhend = min(uhstart + out_kernel_h, unpooled_height + out_pad_h);
    int uwend = min(uwstart + out_kernel_w, unpooled_width + out_pad_w);
    uhstart = max(uhstart, 0);
    uwstart = max(uwstart, 0);
    uhend = min(uhend, unpooled_height);
    uwend = min(uwend, unpooled_width);
    Dtype gradient = 0;
    int offset = (n * channels + c) * unpooled_height * unpooled_width;
    for (int uh = uhstart; uh < uhend; ++uh) {
      for (int uw = uwstart; uw < uwend; ++uw) {
        int unpool_index = uh * unpooled_width + uw;
        gradient += top_diff[unpool_index + offset] / mask[unpool_index];
      }
    }
    bottom_diff[index] = gradient;
  }
}

template <typename Dtype>
void UnPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);
  const int* mask = mask_.gpu_data();
  switch (this->layer_param_.unpooling_param().unpool()) {
  case UnPoolingParameter_UnPoolMethod_FIXED:
    FixedUnPoolBackward<Dtype>
        // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, mask, num_, channels_, height_, width_,
        unpooled_height_, unpooled_width_, out_kernel_h_, out_kernel_w_,
        out_stride_h_, out_stride_w_, out_pad_h_, out_pad_w_, bottom_diff);
    break;
  case UnPoolingParameter_UnPoolMethod_DIV:
    DivUnPoolBackward<Dtype>
        // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, mask, num_, channels_, height_, width_,
        unpooled_height_, unpooled_width_, out_kernel_h_, out_kernel_w_,
        out_stride_h_, out_stride_w_, out_pad_h_, out_pad_w_, bottom_diff);
    break;
  case UnPoolingParameter_UnPoolMethod_REP:
    RepUnPoolBackward<Dtype>
        // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, mask, num_, channels_, height_, width_,
        unpooled_height_, unpooled_width_, out_kernel_h_, out_kernel_w_,
        out_stride_h_, out_stride_w_, out_pad_h_, out_pad_w_, bottom_diff);
    break;
  default:
    LOG(FATAL) << "Unknown unpooling method.";
  }
  CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_LAYER_GPU_FUNCS(UnPoolingLayer);


}  // namespace caffe
