#!/bin/bash

cd llvm
mkdir build
cmake -S llvm -B build -G Ninja -DLLVM_ENABLE_PROJECTS='clang' -DLLVM_ENABLE_BACKENDS='x86;nvptx' -DCMAKE_BUILD_TYPE='Debug'
