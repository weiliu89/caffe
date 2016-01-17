#pragma once

#ifndef CAFFE_EXPORTS
#if defined libcaffe_EXPORTS && defined _MSC_VER
#define CAFFE_EXPORTS __declspec(dllexport)
#else
#define CAFFE_EXPORTS
#endif
#endif

#ifndef PROTO_EXPORTS
#if defined proto_EXPORTS && defined _MSC_VER
#define PROTO_EXPORTS __declspec(dllexport)
#else
#define PROTO_EXPORTS
#endif
#endif

#include "caffe/proto/caffe.pb.h"