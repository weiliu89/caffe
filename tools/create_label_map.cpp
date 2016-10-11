// This program reads in pairs label names and optionally ids and display names
// and store them in LabelMap proto buffer.
// Usage:
//   create_label_map [FLAGS] MAPFILE OUTFILE
// where MAPFILE is a list of label names and optionally label ids and
// displaynames, and OUTFILE stores the information in LabelMap proto.
// Example:
//   ./build/tools/create_label_map --delimiter=" " --include_background=true
//   data/VOC2007/map.txt data/VOC2007/labelmap_voc.prototxt
// The format of MAPFILE is like following:
//   class1 [1] [someclass1]
//   ...
// The format of OUTFILE is like following:
//   item {
//     name: "class1"
//     label: 1
//     display_name: "someclass1"
//   }
//   ...

#include <fstream>  // NOLINT(readability/streams)
#include <string>

#include <boost/program_options.hpp>

#include "caffe/proto/caffe_pb.h"
#include "caffe/util/io.hpp"

using namespace caffe;  // NOLINT(build/namespaces)


int main(int argc, char** argv) {
    boost::program_options::options_description desc("Read in pairs label names and optionally ids and "
        "display names and store them in LabelMap proto buffer.\n"
        "Usage:\n"
        "    create_label_map [FLAGS] MAPFILE OUTFILE\n");
    desc.add_options()
        ("include_background", boost::program_options::value<bool>()->default_value(false), "When this option is on, include none_of_the_above as class 0.")
        ("delimiter", boost::program_options::value<std::string>()->default_value(" "), "The delimiter used to separate fields in label_map_file.");
  
    boost::program_options::variables_map vm;
    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
  

  if (argc < 3) {
    std::cout << desc;
    return 1;
  }

  const bool include_background = vm["include_background"].as<bool>();
  const string delimiter = vm["delimiter"].as<std::string>();

  const string& map_file = argv[1];
  LabelMap label_map;
  ReadLabelFileToLabelMap(map_file, include_background, delimiter, &label_map);

  WriteProtoToTextFile(label_map, argv[2]);
}
