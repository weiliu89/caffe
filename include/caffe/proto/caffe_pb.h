#pragma once

#ifndef CAFFE_EXPORTS
#if defined libcaffe_EXPORTS && defined _MSC_VER
#define CAFFE_EXPORTS __declspec(dllexport)
#else
#define CAFFE_EXPORTS
#endif
#endif
#include "caffe/proto/caffe.pb.h"