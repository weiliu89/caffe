// This program converts a set of images and annotations to a lmdb/leveldb by
// storing them as AnnotatedDatum proto buffers.
// Usage:
//   convert_annoset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME
//
// where ROOTFOLDER is the root folder that holds all the images and
// annotations, and LISTFILE should be a list of files as well as their labels
// or label files.
// For classification task, the file should be in the format as
//   imgfolder1/img1.JPEG 7
//   ....
// For detection task, the file should be in the format as
//   imgfolder1/img1.JPEG annofolder1/anno1.xml
//   ....

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "boost/variant.hpp"
#include "boost/program_options.hpp"

#include "caffe/proto/caffe_pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using boost::scoped_ptr;


cv::Mat ReadImageToCVMat(const string& filename, const int height,
    const int width, const int min_dim, const int max_dim,
    const bool is_color) {
    cv::Mat cv_img;
    int cv_read_flag = (is_color ? cv::IMREAD_ANYCOLOR :
        cv::IMREAD_GRAYSCALE);
    
    cv::Mat cv_img_origin = cv::imread(filename, cv_read_flag);
    if (!cv_img_origin.data) {
        LOG(error) << "Could not open or find file " << filename;
        return cv_img_origin;
    }
    if (min_dim > 0 || max_dim > 0) {
        int num_rows = cv_img_origin.rows;
        int num_cols = cv_img_origin.cols;
        int min_num = std::min(num_rows, num_cols);
        int max_num = std::max(num_rows, num_cols);
        float scale_factor = 1;
        if (min_dim > 0 && min_num < min_dim) {
            scale_factor = static_cast<float>(min_dim) / min_num;
        }
        if (max_dim > 0 && static_cast<int>(scale_factor * max_num) > max_dim) {
            // Make sure the maximum dimension is less than max_dim.
            scale_factor = static_cast<float>(max_dim) / max_num;
        }
        if (scale_factor == 1) {
            cv_img = cv_img_origin;
        }
        else {
            cv::resize(cv_img_origin, cv_img, cv::Size(0, 0),
                scale_factor, scale_factor);
        }
    }
    else if (height > 0 && width > 0) {
        cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
    }
    else {
        cv_img = cv_img_origin;
    }
    return cv_img;
}
static bool matchExt(const std::string & fn,
    std::string en) {
    size_t p = fn.rfind('.') + 1;
    std::string ext = p != fn.npos ? fn.substr(p) : fn;
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    std::transform(en.begin(), en.end(), en.begin(), ::tolower);
    if (ext == en)
        return true;
    if (en == "jpg" && ext == "jpeg")
        return true;
    return false;
}
bool ReadImageToDatum(const string& filename, const int label,
    const int height, const int width, const int min_dim, const int max_dim,
    const bool is_color, const std::string & encoding, Datum* datum) {
    cv::Mat cv_img = ReadImageToCVMat(filename, height, width, min_dim, max_dim,
        is_color);
    if (cv_img.data) {
        if (encoding.size()) {
            if ((cv_img.channels() == 3) == is_color && !height && !width &&
                !min_dim && !max_dim && matchExt(filename, encoding))
                return ReadFileToDatum(filename, label, datum);
            EncodeCVMatToDatum(cv_img, encoding, datum);
            datum->set_label(label);
            return true;
        }
        CVMatToDatum(cv_img, datum);
        datum->set_label(label);
        return true;
    }
    else {
        return false;
    }
}


int main(int argc, char** argv) {
#if USE_OPENCV
    boost::program_options::options_description desc("Convert a set of images and annotations to the "
        "leveldb/lmdb format used as input for Caffe.\n"
        "Usage:\n"
        "    convert_annoset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME\n");
    desc.add_options()
        ("grey", boost::program_options::value<bool>()->default_value(false), "When this option is on, treat images as grayscale ones")
        ("shuffle", boost::program_options::value<bool>()->default_value(false), "Randomly shuffle the order of images and their labels")
        ("backend", boost::program_options::value<string>()->default_value("lmdb"), "The backend {lmdb, leveldb} for storing the result")
        ("anno_type", boost::program_options::value<string>()->default_value("classifcation"), "The type of annotation {classification, detection}.")
        ("label_type", boost::program_options::value<string>()->default_value("xml"), "The type of annotation file format.")
        ("label_map_file", boost::program_options::value<string>()->default_value(""), "A file with LabelMap protobuf message.")
        ("check_label", boost::program_options::value<bool>()->default_value(false), "When this option is on, check that there is no duplicated name/label.")
        ("min_dim", boost::program_options::value<int>()->default_value(0), "Minimum dimension images are resized to (keep same aspect ratio)")
        ("max_dim", boost::program_options::value<int>()->default_value(0), "Maximum dimension images are resized to (keep same aspect ratio)")
        ("resize_width", boost::program_options::value<int>()->default_value(0), "Width images are resized to.")
        ("resize_height", boost::program_options::value<int>()->default_value(0), "Height images are resized to.")
        ("check_size", boost::program_options::value<bool>()->default_value(false), "When this option is on, check that all the datum have the same size")
        ("encoded", boost::program_options::value<bool>()->default_value(false), "When this option is on, the encoded image will be save in datum")
        ("encode_type", boost::program_options::value<string>()->default_value(""), "Optional: What type should we encode the image as ('png','jpg',...).");

    boost::program_options::variables_map vm;
    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);

  if (argc < 4) {
    std::cout << desc;
    return 1;
  }

  const bool is_color = !vm["grey"].as<bool>();
  const bool check_size = vm["check_size"].as<bool>();
  const bool encoded = vm["encoded"].as<bool>();
  const string encode_type = vm["encode_type"].as<string>();
  const string anno_type = vm["anno_type"].as<string>();
  AnnotatedDatum_AnnotationType type;
  const string label_type = vm["label_type"].as<string>();
  const string label_map_file = vm["label_map_file"].as<string>();
  const bool check_label = vm["check_label"].as<bool>();
  std::map<std::string, int> name_to_label;

  std::ifstream infile(argv[2]);
  std::vector<std::pair<std::string, boost::variant<int, std::string> > > lines;
  std::string filename;
  int label;
  std::string labelname;
  if (anno_type == "classification") {
    while (infile >> filename >> label) {
      lines.push_back(std::make_pair(filename, label));
    }
  } else if (anno_type == "detection") {
    type = AnnotatedDatum_AnnotationType_BBOX;
    LabelMap label_map;
    CHECK(ReadProtoFromTextFile(label_map_file, &label_map))
        << "Failed to read label map file.";
    CHECK(MapNameToLabel(label_map, check_label, &name_to_label))
        << "Failed to convert name to label.";
    while (infile >> filename >> labelname) {
      lines.push_back(std::make_pair(filename, labelname));
    }
  }
  if (vm["shuffle"].as<bool>()) {
    // randomly shuffle data
    LOG(info) << "Shuffling data";
    shuffle(lines.begin(), lines.end());
  }
  LOG(info) << "A total of " << lines.size() << " images.";

  if (encode_type.size() && !encoded)
    LOG(info) << "encode_type specified, assuming encoded=true.";

  int min_dim = std::max<int>(0, vm["min_dim"].as<int>());
  int max_dim = std::max<int>(0, vm["max_dim"].as<int>());
  int resize_height = std::max<int>(0, vm["resize_height"].as<int>());
  int resize_width = std::max<int>(0, vm["resize_width"].as<int>());

  // Create new DB
  scoped_ptr<db::DB> db(db::GetDB(vm["backend"].as<string>()));
  db->Open(argv[3], db::NEW);
  scoped_ptr<db::Transaction> txn(db->NewTransaction());

  // Storing to db
  std::string root_folder(argv[1]);
  AnnotatedDatum anno_datum;
  Datum* datum = anno_datum.mutable_datum();
  int count = 0;
  int data_size = 0;
  bool data_size_initialized = false;

  for (int line_id = 0; line_id < lines.size(); ++line_id) {
    bool status = true;
    std::string enc = encode_type;
    if (encoded && !enc.size()) {
      // Guess the encoding type from the file name
      string fn = lines[line_id].first;
      size_t p = fn.rfind('.');
      if ( p == fn.npos )
        LOG(warning) << "Failed to guess the encoding of '" << fn << "'";
      enc = fn.substr(p);
      std::transform(enc.begin(), enc.end(), enc.begin(), ::tolower);
    }
    filename = root_folder + lines[line_id].first;
    if (anno_type == "classification") {
      label = boost::get<int>(lines[line_id].second);
      status = ReadImageToDatum(filename, label, resize_height, resize_width,
          min_dim, max_dim, is_color, enc, datum);
    } else if (anno_type == "detection") {
      labelname = root_folder + boost::get<std::string>(lines[line_id].second);
      status = ReadRichImageToAnnotatedDatum(filename, labelname, resize_height,
          resize_width, min_dim, max_dim, is_color, enc, type, label_type,
          name_to_label, &anno_datum);
      anno_datum.set_type(AnnotatedDatum_AnnotationType_BBOX);
    }
    if (status == false) {
      LOG(warning) << "Failed to read " << lines[line_id].first;
      continue;
    }
    if (check_size) {
      if (!data_size_initialized) {
        data_size = datum->channels() * datum->height() * datum->width();
        data_size_initialized = true;
      } else {
        const std::string& data = datum->data();
        CHECK_EQ(data.size(), data_size) << "Incorrect data field size "
            << data.size();
      }
    }
    // sequential
    string key_str = caffe::format_int(line_id, 8) + "_" + lines[line_id].first;

    // Put in db
    string out;
    CHECK(anno_datum.SerializeToString(&out));
    txn->Put(key_str, out);

    if (++count % 1000 == 0) {
      // Commit db
      txn->Commit();
      txn.reset(db->NewTransaction());
      LOG(info) << "Processed " << count << " files.";
    }
  }
  // write the last batch
  if (count % 1000 != 0) {
    txn->Commit();
    LOG(info) << "Processed " << count << " files.";
  }
#else
  LOG(fatal) << "This tool requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  return 0;
}
