#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "caffe" for configuration "Release"
set_property(TARGET caffe APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(caffe PROPERTIES
  IMPORTED_LINK_INTERFACE_LIBRARIES_RELEASE "proto;proto;/usr/local/lib/libboost_system-mt.dylib;/usr/local/lib/libboost_thread-mt.dylib;/usr/local/lib/libboost_filesystem-mt.dylib;/usr/local/lib/libboost_regex-mt.dylib;/usr/local/lib/libglog.dylib;/usr/local/lib/libgflags.dylib;/usr/local/lib/libprotobuf.dylib;/usr/local/lib/libhdf5_cpp.dylib;/usr/local/lib/libhdf5.dylib;/usr/lib/libpthread.dylib;/usr/lib/libz.dylib;/usr/lib/libdl.dylib;/usr/lib/libm.dylib;/usr/local/lib/libhdf5_hl_cpp.dylib;/usr/local/lib/libhdf5_hl.dylib;/usr/local/lib/liblmdb.dylib;/usr/local/lib/libleveldb.dylib;/usr/local/lib/libsnappy.dylib;/Developer/NVIDIA/CUDA-7.0/lib/libcudart.dylib;/Developer/NVIDIA/CUDA-7.0/lib/libcurand.dylib;/Developer/NVIDIA/CUDA-7.0/lib/libcublas.dylib;/Developer/NVIDIA/CUDA-7.0/lib/libcublas_device.a;/usr/local/cuda/lib;opencv_core;opencv_highgui;opencv_imgproc;-lcblas;-framework Accelerate;/usr/lib/libpython2.7.dylib;/usr/local/lib/libboost_python-mt.dylib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libcaffe.1.0.0-rc3.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/libcaffe.1.0.0-rc3.dylib"
  )

list(APPEND _IMPORT_CHECK_TARGETS caffe )
list(APPEND _IMPORT_CHECK_FILES_FOR_caffe "${_IMPORT_PREFIX}/lib/libcaffe.1.0.0-rc3.dylib" )

# Import target "proto" for configuration "Release"
set_property(TARGET proto APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(proto PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libproto.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS proto )
list(APPEND _IMPORT_CHECK_FILES_FOR_proto "${_IMPORT_PREFIX}/lib/libproto.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
