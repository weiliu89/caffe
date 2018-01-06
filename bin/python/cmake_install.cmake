# Install script for directory: /Users/wangyi/GitHub/SSD/caffe/python

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/Users/wangyi/GitHub/SSD/caffe/bin/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/python" TYPE FILE FILES
    "/Users/wangyi/GitHub/SSD/caffe/python/classify.py"
    "/Users/wangyi/GitHub/SSD/caffe/python/detect.py"
    "/Users/wangyi/GitHub/SSD/caffe/python/draw_net.py"
    "/Users/wangyi/GitHub/SSD/caffe/python/requirements.txt"
    )
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/python" TYPE DIRECTORY FILES "/Users/wangyi/GitHub/SSD/caffe/python/caffe" FILES_MATCHING REGEX "/[^/]*\\.py$" REGEX "/ilsvrc\\_2012\\_mean\\.npy$" REGEX "/test$" EXCLUDE)
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/python/caffe" TYPE SHARED_LIBRARY FILES "/Users/wangyi/GitHub/SSD/caffe/bin/lib/_caffe.dylib")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/python/caffe/_caffe.dylib" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/python/caffe/_caffe.dylib")
    execute_process(COMMAND /opt/local/bin/install_name_tool
      -delete_rpath "/Users/wangyi/GitHub/SSD/caffe/bin/lib"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/python/caffe/_caffe.dylib")
    execute_process(COMMAND /opt/local/bin/install_name_tool
      -delete_rpath "/Developer/NVIDIA/CUDA-7.0/lib"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/python/caffe/_caffe.dylib")
    execute_process(COMMAND /opt/local/bin/install_name_tool
      -add_rpath "/Users/wangyi/GitHub/SSD/caffe/bin/install/lib"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/python/caffe/_caffe.dylib")
    execute_process(COMMAND /opt/local/bin/install_name_tool
      -add_rpath "/Developer/NVIDIA/CUDA-7.0/lib"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/python/caffe/_caffe.dylib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/python/caffe/_caffe.dylib")
    endif()
  endif()
endif()

