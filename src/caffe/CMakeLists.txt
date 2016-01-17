# generate protobuf sources
file(GLOB proto_files proto/*.proto)
caffe_protobuf_generate_cpp_py(${proto_gen_folder} proto_srcs proto_hdrs proto_python ${proto_files})

# include python files either to force generation



add_library(proto STATIC ${proto_hdrs} ${proto_srcs} ${proto_python})
IF(WIN32)
target_link_libraries(proto ${Caffe_LINKER_LIBS})
ENDIF(WIN32)
set(Caffe_LINKER_LIBS proto ${Caffe_LINKER_LIBS}) # note, crucial to prepend!
ADD_DEFINITIONS(-Dlibcaffe_EXPORTS)

caffe_default_properties(proto)

IF(WIN32)
#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /FIcaffe/common.hpp")
	SET_TARGET_PROPERTIES(proto PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} /FIcaffe/proto/caffe_pb.h /Dproto_EXPORTS")
ELSE(WIN32)
    SET_TARGET_PROPERTIES(proto PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -include caffe/common.hpp")
ENDIF(WIN32)
# --[ Caffe library

# creates 'test_srcs', 'srcs', 'test_cuda', 'cuda' lists
caffe_pickup_caffe_sources(${PROJECT_SOURCE_DIR})

if(HAVE_CUDA)
  caffe_cuda_compile(cuda_objs ${cuda})
  list(APPEND srcs ${cuda_objs} ${cuda})
endif()

IF(WIN32)
  #add_definitions(-D_CRT_SECURE_NO_WARNINGS)
  #add_definitions(-D_VARIADIC_MAX=10)
  #add_definitions(-DGTEST_HAS_TR1_TUPLE=0)
  LINK_DIRECTORIES(${Boost_LIBRARY_DIR_DEBUG})
  
  add_library(libcaffe_${LINK_TYPE} ${LINK_TYPE} ${srcs})
  
  target_link_libraries(libcaffe_${LINK_TYPE} proto ${Caffe_LINKER_LIBS})
  caffe_default_properties(libcaffe_${LINK_TYPE})

else(WIN32)
  add_library(caffe ${srcs})
  target_link_libraries(caffe proto ${Caffe_LINKER_LIBS})
  caffe_default_properties(caffe)
ENDIF(WIN32)



# ---[ Tests
 add_subdirectory(test)

# ---[ Install
install(DIRECTORY ${Caffe_INCLUDE_DIR}/caffe DESTINATION include)
install(FILES ${proto_hdrs} DESTINATION include/caffe/proto)
IF(WIN32)
	install(TARGETS libcaffe_${LINK_TYPE} proto EXPORT CaffeTargets DESTINATION lib)
ELSE(WIN32)
	install(TARGETS caffe proto EXPORT CaffeTargets DESTINATION lib)
ENDIF(WIN32)

file(WRITE ${PROJECT_BINARY_DIR}/__init__.py)
list(APPEND proto_python ${PROJECT_BINARY_DIR}/__init__.py)
install(PROGRAMS ${proto_python} DESTINATION python/caffe/proto)


