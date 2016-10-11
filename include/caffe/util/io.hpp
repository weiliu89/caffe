#ifndef CAFFE_UTIL_IO_H_
#define CAFFE_UTIL_IO_H_

#include <boost/filesystem.hpp>
#include <iomanip>
#include <iostream>  // NOLINT(readability/streams)
#include <string>

#include "google/protobuf/message.h"

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/format.hpp"


#ifndef CAFFE_TMP_DIR_RETRIES
#define CAFFE_TMP_DIR_RETRIES 100
#endif

namespace caffe {

using ::google::protobuf::Message;
using ::boost::filesystem::path;

inline void MakeTempDir(string* temp_dirname) {
  temp_dirname->clear();
  const path& model =
    boost::filesystem::temp_directory_path()/"caffe_test.%%%%-%%%%";
  for ( int i = 0; i < CAFFE_TMP_DIR_RETRIES; i++ ) {
    const path& dir = boost::filesystem::unique_path(model).string();
    bool done = boost::filesystem::create_directory(dir);
    if ( done ) {
      *temp_dirname = dir.string();
      return;
    }
  }
  LOG(fatal) << "Failed to create a temporary directory.";
}

inline void MakeTempFilename(string* temp_filename) {
  static path temp_files_subpath;
  static uint64_t next_temp_file = 0;
  temp_filename->clear();
  if ( temp_files_subpath.empty() ) {
    string path_string="";
    MakeTempDir(&path_string);
    temp_files_subpath = path_string;
  }
  *temp_filename =
    (temp_files_subpath/caffe::format_int(next_temp_file++, 9)).string();
}

inline void GetTempDirname(string* temp_dirname) {
    temp_dirname->clear();
    const path& model =
        boost::filesystem::temp_directory_path() / "caffe_test.%%%%-%%%%";
    for (int i = 0; i < CAFFE_TMP_DIR_RETRIES; i++) {
        const path& dir = boost::filesystem::unique_path(model).string();
        bool done = boost::filesystem::create_directory(dir);
        if (done) {
            bool remove_done = boost::filesystem::remove(dir);
            if (remove_done) {
                *temp_dirname = dir.string();
                return;
            }
            LOG(fatal) << "Failed to remove a temporary directory.";
        }
    }
    LOG(fatal) << "Failed to create a temporary directory.";
}

inline void GetTempFilename(string* temp_filename) {
    static path temp_files_subpath;
    static uint64_t next_temp_file = 0;
    temp_filename->clear();
    if (temp_files_subpath.empty()) {
        string path_string = "";
        GetTempDirname(&path_string);
        temp_files_subpath = path_string;
    }
    *temp_filename =
        (temp_files_subpath / caffe::format_int(next_temp_file++, 9)).string();
}

bool DLL_EXPORT ReadProtoFromTextFile(const char* filename, Message* proto);

inline bool ReadProtoFromTextFile(const string& filename, Message* proto) {
  return ReadProtoFromTextFile(filename.c_str(), proto);
}

inline void ReadProtoFromTextFileOrDie(const char* filename, Message* proto) {
  CHECK(ReadProtoFromTextFile(filename, proto));
}

inline void ReadProtoFromTextFileOrDie(const string& filename, Message* proto) {
  ReadProtoFromTextFileOrDie(filename.c_str(), proto);
}

void DLL_EXPORT WriteProtoToTextFile(const Message& proto, const char* filename);
inline void DLL_EXPORT WriteProtoToTextFile(const Message& proto, const string& filename) {
  WriteProtoToTextFile(proto, filename.c_str());
}

bool DLL_EXPORT ReadProtoFromBinaryFile(const char* filename, Message* proto);

inline bool ReadProtoFromBinaryFile(const string& filename, Message* proto) {
  return ReadProtoFromBinaryFile(filename.c_str(), proto);
}

inline void ReadProtoFromBinaryFileOrDie(const char* filename, Message* proto) {
  CHECK(ReadProtoFromBinaryFile(filename, proto));
}

inline void ReadProtoFromBinaryFileOrDie(const string& filename,
                                         Message* proto) {
  ReadProtoFromBinaryFileOrDie(filename.c_str(), proto);
}


void DLL_EXPORT WriteProtoToBinaryFile(const Message& proto, const char* filename);
inline void WriteProtoToBinaryFile(
    const Message& proto, const string& filename) {
  WriteProtoToBinaryFile(proto, filename.c_str());
}

bool DLL_EXPORT ReadFileToDatum(const string& filename, const int label, Datum* datum);

inline bool ReadFileToDatum(const string& filename, Datum* datum) {
  return ReadFileToDatum(filename, -1, datum);
}

bool DLL_EXPORT ReadImageToDatum(const string& filename, const int label,
    const int height, const int width, const bool is_color,
    const std::string & encoding, Datum* datum);

inline bool  ReadImageToDatum(const string& filename, const int label,
    const int height, const int width, const bool is_color, Datum* datum) {
  return ReadImageToDatum(filename, label, height, width, is_color,
                          "", datum);
}

inline bool ReadImageToDatum(const string& filename, const int label,
    const int height, const int width, Datum* datum) {
  return ReadImageToDatum(filename, label, height, width, true, datum);
}

inline bool ReadImageToDatum(const string& filename, const int label,
    const bool is_color, Datum* datum) {
  return ReadImageToDatum(filename, label, 0, 0, is_color, datum);
}

inline bool ReadImageToDatum(const string& filename, const int label,
    Datum* datum) {
  return ReadImageToDatum(filename, label, 0, 0, true, datum);
}

inline bool ReadImageToDatum(const string& filename, const int label,
    const std::string & encoding, Datum* datum) {
  return ReadImageToDatum(filename, label, 0, 0, true, encoding, datum);
}

bool DLL_EXPORT DecodeDatumNative(Datum* datum);
bool DLL_EXPORT DecodeDatum(Datum* datum, bool is_color);


bool DLL_EXPORT ReadRichImageToAnnotatedDatum(const string& filename,
    const string& labelname, const int height, const int width,
    const int min_dim, const int max_dim, const bool is_color,
    const std::string& encoding, const AnnotatedDatum_AnnotationType type,
    const string& labeltype, const std::map<string, int>& name_to_label,
    AnnotatedDatum* anno_datum);

inline bool ReadRichImageToAnnotatedDatum(const string& filename,
    const string& labelname, const int height, const int width,
    const bool is_color, const std::string & encoding,
    const AnnotatedDatum_AnnotationType type, const string& labeltype,
    const std::map<string, int>& name_to_label, AnnotatedDatum* anno_datum) {
    return ReadRichImageToAnnotatedDatum(filename, labelname, height, width, 0, 0,
        is_color, encoding, type, labeltype, name_to_label,
        anno_datum);
}

bool DLL_EXPORT ReadXMLToAnnotatedDatum(const string& labelname, const int img_height,
    const int img_width, const std::map<string, int>& name_to_label,
    AnnotatedDatum* anno_datum);

bool DLL_EXPORT ReadJSONToAnnotatedDatum(const string& labelname, const int img_height,
    const int img_width, const std::map<string, int>& name_to_label,
    AnnotatedDatum* anno_datum);

bool DLL_EXPORT ReadTxtToAnnotatedDatum(const string& labelname, const int height,
    const int width, AnnotatedDatum* anno_datum);

bool DLL_EXPORT ReadLabelFileToLabelMap(const string& filename, bool include_background,
    const string& delimiter, LabelMap* map);

inline bool ReadLabelFileToLabelMap(const string& filename,
    bool include_background, LabelMap* map) {
    return ReadLabelFileToLabelMap(filename, include_background, " ", map);
}

inline bool ReadLabelFileToLabelMap(const string& filename, LabelMap* map) {
    return ReadLabelFileToLabelMap(filename, true, map);
}

bool DLL_EXPORT MapNameToLabel(const LabelMap& map, const bool strict_check,
    std::map<string, int>* name_to_label);

inline bool MapNameToLabel(const LabelMap& map,
    std::map<string, int>* name_to_label) {
    return MapNameToLabel(map, true, name_to_label);
}

bool DLL_EXPORT MapLabelToName(const LabelMap& map, const bool strict_check,
    std::map<int, string>* label_to_name);

inline bool MapLabelToName(const LabelMap& map,
    std::map<int, string>* label_to_name) {
    return MapLabelToName(map, true, label_to_name);
}

bool DLL_EXPORT MapLabelToDisplayName(const LabelMap& map, const bool strict_check,
    std::map<int, string>* label_to_display_name);

inline bool MapLabelToDisplayName(const LabelMap& map,
    std::map<int, string>* label_to_display_name) {
    return MapLabelToDisplayName(map, true, label_to_display_name);
}


#ifdef USE_OPENCV
cv::Mat DLL_EXPORT ReadImageToCVMat(const string& filename,
    const int height, const int width, const bool is_color);

cv::Mat DLL_EXPORT ReadImageToCVMat(const string& filename,
    const int height, const int width);

cv::Mat DLL_EXPORT ReadImageToCVMat(const string& filename,
    const bool is_color);

cv::Mat DLL_EXPORT ReadImageToCVMat(const string& filename);

cv::Mat DLL_EXPORT DecodeDatumToCVMatNative(const Datum& datum);
cv::Mat DLL_EXPORT DecodeDatumToCVMat(const Datum& datum, bool is_color);

void DLL_EXPORT EncodeCVMatToDatum(const cv::Mat& cv_img, const string& encoding,
    Datum* datum);

void DLL_EXPORT CVMatToDatum(const cv::Mat& cv_img, Datum* datum);

void DLL_EXPORT GetImageSize(const string& filename, int* height, int* width);
#endif  // USE_OPENCV
}  // namespace caffe

#endif   // CAFFE_UTIL_IO_H_
