#include <H5Cpp.h>
#ifndef H5_NO_NAMESPACE
using namespace H5;
#endif
int main(int argc, char **argv) {
  char const* info_ver = "INFO" ":" H5_VERSION;
  H5File file("foo.h5", H5F_ACC_TRUNC);
  return 0;
}