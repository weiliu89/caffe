#include "caffe/caffe.hpp"

int main(int argc, char** argv) {
  LOG(fatal) << "Deprecated. Use caffe train --solver=... "
                "[--snapshot=...] instead.";
  return 0;
}
