#ifndef CAFFE_UTIL_HDF5_H_
#define CAFFE_UTIL_HDF5_H_

#include <string>

#include "hdf5.h"
#include "hdf5_hl.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
namespace caffe {

template <typename Dtype>
DLL_EXPORT void hdf5_load_nd_dataset_helper(
    hid_t file_id, const char* dataset_name_, int min_dim, int max_dim,
    Blob<Dtype>* blob);

template <typename Dtype>
DLL_EXPORT void hdf5_load_nd_dataset(
    hid_t file_id, const char* dataset_name_, int min_dim, int max_dim,
    Blob<Dtype>* blob);

template <typename Dtype>
DLL_EXPORT void hdf5_save_nd_dataset(
    const hid_t file_id, const string& dataset_name, const Blob<Dtype>& blob,
    bool write_diff = false);

DLL_EXPORT int hdf5_load_int(hid_t loc_id, const string& dataset_name);
DLL_EXPORT void hdf5_save_int(hid_t loc_id, const string& dataset_name, int i);
DLL_EXPORT string hdf5_load_string(hid_t loc_id, const string& dataset_name);
DLL_EXPORT void hdf5_save_string(hid_t loc_id, const string& dataset_name,
                      const string& s);

DLL_EXPORT int hdf5_get_num_links(hid_t loc_id);
DLL_EXPORT string hdf5_get_name_by_idx(hid_t loc_id, int idx);

}  // namespace caffe

#endif   // CAFFE_UTIL_HDF5_H_
