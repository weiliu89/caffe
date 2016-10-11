#pragma once

#include "caffe/layers/loss_layer.hpp"

namespace caffe{
    template <typename Dtype>
    class SmoothL1LossLayer : public LossLayer<Dtype> {
    public:
        explicit SmoothL1LossLayer(const LayerParameter& param)
            : LossLayer<Dtype>(param), diff_() {}
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);

        virtual inline const char* type() const { return "SmoothL1Loss"; }

        virtual inline int ExactNumBottomBlobs() const { return -1; }
        virtual inline int MinBottomBlobs() const { return 2; }
        virtual inline int MaxBottomBlobs() const { return 3; }

        /**
        * Unlike most loss layers, in the SmoothL1LossLayer we can backpropagate
        * to both inputs -- override to return true and always allow force_backward.
        */
        virtual inline bool AllowForceBackward(const int bottom_index) const {
            return true;
        }

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
        virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);

        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
        virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

        Blob<Dtype> diff_;
        Blob<Dtype> errors_;
        bool has_weights_;
    };
}