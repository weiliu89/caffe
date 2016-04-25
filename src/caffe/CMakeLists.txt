# generate protobuf sources
file(GLOB proto_files proto/*.proto)
caffe_protobuf_generate_cpp_py(${proto_gen_folder} proto_srcs proto_hdrs proto_python ${proto_files})
if(MSVC)
  set(caffe_lib_name libcaffe)
else(MSVC)
  set(caffe_lib_name caffe)
endif(MSVC)

# include python files either to force generation
add_library(proto STATIC ${proto_hdrs} ${proto_srcs} ${proto_python})
IF(MSVC)
  SET_TARGET_PROPERTIES(proto PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} /FIcaffe/proto/caffe_pb.h /Dproto_EXPORTS")
else(MSVC)
  SET_TARGET_PROPERTIES(proto PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -include caffe/common.hpp")
endif(MSVC)

set(Caffe_LINKER_LIBS proto ${Caffe_LINKER_LIBS}) # note, crucial to prepend!
caffe_default_properties(proto)

# --[ Caffe library

# creates 'test_srcs', 'srcs', 'test_cuda', 'cuda' lists
caffe_pickup_caffe_sources(${PROJECT_SOURCE_DIR})

if(HAVE_CUDA)
  caffe_cuda_compile(cuda_objs ${cuda})
  list(APPEND srcs ${cuda_objs} ${cuda})
endif()


add_library(${caffe_lib_name} ${srcs})
target_link_libraries(${caffe_lib_name} proto ${Caffe_LINKER_LIBS})
caffe_default_properties(${caffe_lib_name})
set_target_properties(${caffe_lib_name} PROPERTIES
  VERSION   ${CAFFE_TARGET_VERSION}
  SOVERSION ${CAFFE_TARGET_SOVERSION}
)

# ---[ Tests
 add_subdirectory(test)

# ---[ Install
install(DIRECTORY ${Caffe_INCLUDE_DIR}/caffe DESTINATION include)
install(FILES ${proto_hdrs} DESTINATION include/caffe/proto)


# install pdb files
IF(MSVC)
  install( FILES ${PROJECT_BINARY_DIR}/Bin/Debug/${caffe_lib_name}-d.pdb DESTINATION bin CONFIGURATIONS Debug)
  install( FILES ${PROJECT_BINARY_DIR}/Bin/Release/${caffe_lib_name}.pdb DESTINATION bin CONFIGURATIONS Release)
  install(TARGETS ${caffe_lib_name} proto 
    EXPORT CaffeTargets DESTINATION lib
	RUNTIME DESTINATION bin)
else()
  install(TARGETS ${caffe_lib_name} proto EXPORT CaffeTargets DESTINATION lib)

ENDIF(MSVC)

file(WRITE ${PROJECT_BINARY_DIR}/__init__.py)
list(APPEND proto_python ${PROJECT_BINARY_DIR}/__init__.py)
install(PROGRAMS ${proto_python} DESTINATION python/caffe/proto)


