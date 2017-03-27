#!/bin/sh
# Script for installing Caffe with cuDNN support on Jetson TX1 Development Kitls
# 9-15-16 JetsonHacks.com
# MIT License
# Install and compile Caffe on NVIDIA Jetson TX1 Development Kit
# Prerequisites (which can be installed with JetPack 2):
# L4T 24.2 (Ubuntu 16.04)
# OpenCV4Tegra
# CUDA 8.0
# cuDNN v5.1
# Tested with last Github Caffe commit: 80f44100e19fd371ff55beb3ec2ad5919fb6ac43
sudo add-apt-repository universe
sudo apt-get update -y
/bin/echo -e "\e[1;32mLoading Caffe Dependencies.\e[0m"
sudo apt-get install cmake -y
# General Dependencies
sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev \
libhdf5-serial-dev protobuf-compiler -y
sudo apt-get install --no-install-recommends libboost-all-dev -y
# BLAS
sudo apt-get install libatlas-base-dev -y
# Remaining Dependencies
sudo apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev -y
sudo apt-get install python-dev python-numpy -y

sudo usermod -a -G video $USER
/bin/echo -e "\e[1;32mCloning Caffe into the home directory\e[0m"
cd caffe 
git checkout ssd
cp Makefile.config.example Makefile.config
# Enable cuDNN usage
sed -i 's/# USE_CUDNN/USE_CUDNN/g' Makefile.config
# Regen the makefile; On 16.04, aarch64 has issues with a static cuda runtime
mkdir build
cd build
cmake -DCUDA_USE_STATIC_CUDA_RUNTIME=OFF ..
cd ..
# Include the hdf5 directory for the includes; 16.04 has issues for some reason
echo "INCLUDE_DIRS += /usr/include/hdf5/serial/" >> Makefile.config
/bin/echo -e "\e[1;32mCompiling Caffe\e[0m"
make -j4 all
# Run the tests to make sure everything works
/bin/echo -e "\e[1;32mRunning Caffe Tests\e[0m"
make -j4 runtest
# The following is a quick timing test ...
tools/caffe time --model=models/bvlc_alexnet/deploy.prototxt --gpu=0
