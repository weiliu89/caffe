#include <vector>
#include <string>
#include <boost/filesystem.hpp>
#include "caffe/caffe.hpp"

int main(int argc, char* argv[]) {
  // check number of args
  if (argc != 4) {
    std::cout << "usage: predict_video <model_name> <weight_name> <video_filename>" << std::endl;
    return 1;
  }

  // read args
  std::string model_filename = argv[1];
  std::string weight_filename = argv[2];
  std::string video_filename = argv[3];

  // check file exist
  if (!boost::filesystem::exists( model_filename)) { std::cout << "Can not find file \"" <<  model_filename << '"' << std::endl; return 2; }
  if (!boost::filesystem::exists(weight_filename)) { std::cout << "Can not find file \"" << weight_filename << '"' << std::endl; return 2; }
  if (!boost::filesystem::exists( video_filename)) { std::cout << "Can not find file \"" <<  video_filename << '"' << std::endl; return 2; }

  // read net according to args
  int level = 0;
  std::vector<std::string> stages;
  caffe::Net<float> caffe_net(model_filename, caffe::TEST, level, &stages);
  caffe_net.CopyTrainedLayersFrom(weight_filename);
  
}
