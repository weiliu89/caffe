#ifndef CAFFE_UTIL_IO_H_
#define CAFFE_UTIL_IO_H_

#include <unistd.h>
#include <string>

#include "google/protobuf/message.h"
#include "hdf5.h"
#include "hdf5_hl.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "mkstemp.h"

#define HDF5_NUM_DIMS 4

namespace caffe {

using ::google::protobuf::Message;

inline void DLL_EXPORT MakeTempFilename(string* temp_filename) {
  temp_filename->clear();
  *temp_filename = "/tmp/caffe_test.XXXXXX";
  char* temp_filename_cstr = new char[temp_filename->size() + 1];
  // NOLINT_NEXT_LINE(runtime/printf)
  strcpy(temp_filename_cstr, temp_filename->c_str());

  int fd = mkstemp(temp_filename_cstr);

  CHECK_GE(fd, 0) << "Failed to open a temporary file at: " << *temp_filename;
  close(fd);
  *temp_filename = temp_filename_cstr;
  delete[] temp_filename_cstr;
}

inline void DLL_EXPORT MakeTempDir(string* temp_dirname) {
  temp_dirname->clear();
  *temp_dirname = "/tmp/caffe_test.XXXXXX";
  char* temp_dirname_cstr = new char[temp_dirname->size() + 1];
  // NOLINT_NEXT_LINE(runtime/printf)
  strcpy(temp_dirname_cstr, temp_dirname->c_str());
#ifndef _MSC_VER 
  char* mkdtemp_result = mkdtemp(temp_dirname_cstr); 
#else 
  errno_t mkdtemp_result = _mktemp_s(temp_dirname_cstr, sizeof(temp_dirname_cstr)); 
#endif
//  char* mkdtemp_result = mkdtemp(temp_dirname_cstr);

  CHECK(mkdtemp_result != NULL)
      << "Failed to create a temporary directory at: " << *temp_dirname;
  *temp_dirname = temp_dirname_cstr;
  delete[] temp_dirname_cstr;
}

bool DLL_EXPORT ReadProtoFromTextFile(const char* filename, Message* proto);

inline bool DLL_EXPORT ReadProtoFromTextFile(const string& filename, Message* proto) {
  return ReadProtoFromTextFile(filename.c_str(), proto);
}

inline void DLL_EXPORT ReadProtoFromTextFileOrDie(const char* filename, Message* proto) {
  CHECK(ReadProtoFromTextFile(filename, proto));
}

inline void DLL_EXPORT ReadProtoFromTextFileOrDie(const string& filename, Message* proto) {
  ReadProtoFromTextFileOrDie(filename.c_str(), proto);
}

void DLL_EXPORT WriteProtoToTextFile(const Message& proto, const char* filename);
inline void DLL_EXPORT WriteProtoToTextFile(const Message& proto, const string& filename) {
  WriteProtoToTextFile(proto, filename.c_str());
}

bool DLL_EXPORT ReadProtoFromBinaryFile(const char* filename, Message* proto);

inline bool DLL_EXPORT ReadProtoFromBinaryFile(const string& filename, Message* proto) {
  return ReadProtoFromBinaryFile(filename.c_str(), proto);
}

inline void DLL_EXPORT ReadProtoFromBinaryFileOrDie(const char* filename, Message* proto) {
  CHECK(ReadProtoFromBinaryFile(filename, proto));
}

inline void DLL_EXPORT ReadProtoFromBinaryFileOrDie(const string& filename,
                                         Message* proto) {
  ReadProtoFromBinaryFileOrDie(filename.c_str(), proto);
}


void DLL_EXPORT WriteProtoToBinaryFile(const Message& proto, const char* filename);
inline void DLL_EXPORT WriteProtoToBinaryFile(
    const Message& proto, const string& filename) {
  WriteProtoToBinaryFile(proto, filename.c_str());
}

bool DLL_EXPORT ReadFileToDatum(const string& filename, const int label, Datum* datum);

inline bool DLL_EXPORT ReadFileToDatum(const string& filename, Datum* datum) {
  return ReadFileToDatum(filename, -1, datum);
}

bool DLL_EXPORT ReadImageToDatum(const string& filename, const int label,
    const int height, const int width, const bool is_color,
    const std::string & encoding, Datum* datum);

inline bool DLL_EXPORT ReadImageToDatum(const string& filename, const int label,
    const int height, const int width, const bool is_color, Datum* datum) {
  return ReadImageToDatum(filename, label, height, width, is_color,
                          "", datum);
}

inline bool DLL_EXPORT ReadImageToDatum(const string& filename, const int label,
    const int height, const int width, Datum* datum) {
  return ReadImageToDatum(filename, label, height, width, true, datum);
}

inline bool DLL_EXPORT ReadImageToDatum(const string& filename, const int label,
    const bool is_color, Datum* datum) {
  return ReadImageToDatum(filename, label, 0, 0, is_color, datum);
}

inline bool DLL_EXPORT ReadImageToDatum(const string& filename, const int label,
    Datum* datum) {
  return ReadImageToDatum(filename, label, 0, 0, true, datum);
}

inline bool DLL_EXPORT ReadImageToDatum(const string& filename, const int label,
    const std::string & encoding, Datum* datum) {
  return ReadImageToDatum(filename, label, 0, 0, true, encoding, datum);
}

bool DLL_EXPORT DecodeDatumNative(Datum* datum);
bool DLL_EXPORT DecodeDatum(Datum* datum, bool is_color);

cv::Mat DLL_EXPORT ReadImageToCVMat(const string& filename,
    const int height, const int width, const bool is_color);

cv::Mat DLL_EXPORT ReadImageToCVMat(const string& filename,
    const int height, const int width);

cv::Mat DLL_EXPORT ReadImageToCVMat(const string& filename,
    const bool is_color);

cv::Mat DLL_EXPORT ReadImageToCVMat(const string& filename);

cv::Mat DLL_EXPORT DecodeDatumToCVMatNative(const Datum& datum);
cv::Mat DLL_EXPORT DecodeDatumToCVMat(const Datum& datum, bool is_color);

void DLL_EXPORT CVMatToDatum(const cv::Mat& cv_img, Datum* datum);

template <typename Dtype>
void DLL_EXPORT hdf5_load_nd_dataset_helper(
    hid_t file_id, const char* dataset_name_, int min_dim, int max_dim,
    Blob<Dtype>* blob);

template <typename Dtype>
void DLL_EXPORT hdf5_load_nd_dataset(
    hid_t file_id, const char* dataset_name_, int min_dim, int max_dim,
    Blob<Dtype>* blob);

template <typename Dtype>
void DLL_EXPORT hdf5_save_nd_dataset(
    const hid_t file_id, const string& dataset_name, const Blob<Dtype>& blob);

}  // namespace caffe

#endif   // CAFFE_UTIL_IO_H_
