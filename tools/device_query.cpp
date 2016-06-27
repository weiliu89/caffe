#include "caffe/common.hpp"

int main(int argc, char** argv) {
  LOG(fatal) << "Deprecated. Use caffe device_query "
                "[--device_id=0] instead.";
  return 0;
}
