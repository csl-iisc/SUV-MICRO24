#!/bin/bash

profile=$1
tx=$2

sed -i "s/#define NVML_PROFILER.*/#define NVML_PROFILER ${1}/g" penguin-sc.h
sed -i "s/#define NVML_TX.*/#define NVML_TX ${2}/g" penguin-sc.h
sed -i "s/#define NVML_PROFILER.*/#define NVML_PROFILER ${1}/g" penguin-suv.h
sed -i "s/#define NVML_TX.*/#define NVML_TX ${2}/g" penguin-suv.h
