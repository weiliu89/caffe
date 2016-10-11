// This program retrieves the sizes of a set of images.
// Usage:
//   get_image_size [FLAGS] ROOTFOLDER/ LISTFILE OUTFILE
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

#include <fstream>  // NOLINT(readability/streams)
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe_pb.h"
#include <boost/program_options.hpp>
#include <opencv2/imgcodecs.hpp>

#include "caffe/util/io.hpp"

using namespace caffe;  // NOLINT(build/namespaces)


int main(int argc, char** argv) {
    boost::program_options::options_description desc("Allowed options");
    desc.add_options()
        ("name_id_file", boost::program_options::value<std::string>()->default_value(""), "A file which maps image_name to image_id.")
        ("also_log_to_stderr", boost::program_options::value<bool>()->default_value(true), "Print log messages to stderror as well");
    boost::program_options::variables_map vm;
    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
#ifdef USE_OPENCV
  

  if (argc < 4) {
    return 1;
  }

  std::ifstream infile(argv[2]);
  if (!infile.good()) {
    LOG(fatal) << "Failed to open file: " << argv[2];
  }
  std::vector<std::pair<std::string, std::string> > lines;
  std::string filename, label;
  while (infile >> filename >> label) {
    lines.push_back(std::make_pair(filename, label));
  }
  infile.close();
  LOG(info) << "A total of " << lines.size() << " images.";
  
  const string name_id_file = vm["name_id_file"].as<std::string>();
  std::map<string, int> map_name_id;
  if (!name_id_file.empty()) {
    std::ifstream nameidfile(name_id_file.c_str());
    if (!nameidfile.good()) {
      LOG(fatal) << "Failed to open name_id_file: " << name_id_file;
    }
    std::string name;
    int id;
    while (nameidfile >> name >> id) {
      CHECK(map_name_id.find(name) == map_name_id.end());
      map_name_id[name] = id;
    }
    CHECK_EQ(map_name_id.size(), lines.size());
  }

  // Storing to outfile
  boost::filesystem::path root_folder(argv[1]);
  std::ofstream outfile(argv[3]);
  if (!outfile.good()) {
    LOG(fatal) << "Failed to open file: " << argv[3];
  }
  int height, width;
  int count = 0;
  for (int line_id = 0; line_id < lines.size(); ++line_id) {
    boost::filesystem::path img_file = root_folder / lines[line_id].first;
    GetImageSize(img_file.string(), &height, &width);
    std::string img_name = img_file.stem().string();
    if (map_name_id.size() == 0) {
      outfile << img_name << " " << height << " " << width << std::endl;
    } else {
      CHECK(map_name_id.find(img_name) != map_name_id.end());
      int img_id = map_name_id.find(img_name)->second;
      outfile << img_id << " " << height << " " << width << std::endl;
    }

    if (++count % 1000 == 0) {
      LOG(info) << "Processed " << count << " files.";
    }
  }
  // write the last batch
  if (count % 1000 != 0) {
    LOG(info) << "Processed " << count << " files.";
  }
  outfile.flush();
  outfile.close();
#else
  LOG(FATAL) << "This tool requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  return 0;
}
