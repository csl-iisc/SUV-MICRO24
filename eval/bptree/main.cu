/*Copyright(c) 2020, The Regents of the University of California, Davis.            */
/*                                                                                  */
/*                                                                                  */
/*Redistribution and use in source and binary forms, with or without modification,  */
/*are permitted provided that the following conditions are met :                    */
/*                                                                                  */
/*1. Redistributions of source code must retain the above copyright notice, this    */
/*list of conditions and the following disclaimer.                                  */
/*2. Redistributions in binary form must reproduce the above copyright notice,      */
/*this list of conditions and the following disclaimer in the documentation         */
/*and / or other materials provided with the distribution.                          */
/*                                                                                  */
/*THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND   */
/*ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED     */
/*WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.*/
/*IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,  */
/*INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES(INCLUDING, BUT */
/*NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR*/
/*PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, */
/*WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT(INCLUDING NEGLIGENCE OR OTHERWISE) */
/*ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE        */
/*POSSIBILITY OF SUCH DAMAGE.                                                       */
/************************************************************************************/
/************************************************************************************/

#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <random>
#include <vector>

#define SKEWQUERY 0
#define SHFLINS 1
#define SC19 0

#define MiB 20447
#define RESERVATION ((MiB*1024UL*1024))

#include "penguin.h"
#include "GpuBTree.h"

int main(int argc, char* argv[]) {
  /* GpuBTree::GpuBTreeMap<uint32_t, uint32_t, uint32_t> btree; */

  int* reservation;
  /* cudaMalloc((void**) &reservation, RESERVATION); */
  GpuBTree::GpuBTreeMap<uint32_t, uint32_t, uint32_t> btree;
  // Input number of keys
  /* uint32_t numKeys = 307200000; */
  uint32_t numKeys = 250000000;
  /* if (argc > 1) */
    /* numKeys = std::atoi(argv[1]); */

  // RNG
  std::random_device rd;
  std::mt19937 g(rd());

  ///////////////////////////////////
  ///		 Build the tree    	  ///
  ///////////////////////////////////

  // Prepare the keys
  std::vector<uint32_t> keys;
  std::vector<uint32_t> values;
  keys.reserve(numKeys);
  values.reserve(numKeys);
  for (uint32_t iKey = 0; iKey < numKeys; iKey++) {
    keys.push_back(iKey);
  }

#if SHFLINS
  // shuffle the keys
  std::shuffle(keys.begin(), keys.end(), g);
#endif

  // assign the values
  for (uint32_t iKey = 0; iKey < numKeys; iKey++) {
    values.push_back(keys[iKey]);
  }

  // Move data to GPU
  GpuTimer build_timer;
  build_timer.timerStart();
  uint32_t *d_keys, *d_values;
  /* CHECK_ERROR(memoryUtil::deviceAlloc(d_keys, numKeys)); */
  /* CHECK_ERROR(memoryUtil::deviceAlloc(d_values, numKeys)); */
  /* cudaMallocManaged((void**)&d_keys, sizeof(uint32_t) * numKeys); */
  /* cudaMallocManaged((void**)&d_values, sizeof(uint32_t) * numKeys); */
  CHECK_ERROR(memoryUtil::hostAlloc(d_keys, numKeys));
  CHECK_ERROR(memoryUtil::hostAlloc(d_values, numKeys));
  CHECK_ERROR(memoryUtil::cpyToDevice(keys.data(), d_keys, numKeys));
  CHECK_ERROR(memoryUtil::cpyToDevice(values.data(), d_values, numKeys));

  // Build the tree
  /* GpuTimer build_timer; */
  /* build_timer.timerStart(); */
  btree.insertKeys(d_keys, d_values, numKeys, SourceT::DEVICE);
  build_timer.timerStop();
  printf("insert done\n");
  cudaError_t err = cudaGetLastError();
  if ( cudaSuccess != err) {
    fprintf(stderr, "Kernel execution failed: %s.\n",
        cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  ///////////////////////////////////
  ///		 Query the tree       ///
  ///////////////////////////////////

  btree.getAllocator()->bringToHostSide();
  cudaMalloc((void**) &reservation, RESERVATION);

  // Input number of queries
  uint32_t numQueries = 128 * 1024 * 1024;
  /* uint32_t numQueries = 64 * 1024 * 1024; */
  /* if (argc > 2) */
  /*   numQueries = std::atoi(argv[2]); */
  // Prepare the query keys
  std::vector<uint32_t> query_keys;
  std::vector<uint32_t> query_results;
  /* query_keys.reserve(numQueries * 2); */
  query_keys.reserve(numQueries );
  query_results.resize(numQueries);
  /* for (uint32_t iKey = 0; iKey < numQueries * 2; iKey++) { */
  /*   query_keys.push_back(iKey); */
  /* } */
  for (uint32_t iKey = 0; iKey < numKeys ; iKey++) {
#if SKEWQUERY
    if(iKey % 25 == 0){
      query_keys.push_back(iKey);
    } else {
      uint32_t fifth = numKeys / 25;
      uint32_t key = iKey % fifth;
      key = 24 * fifth + key;
      query_keys.push_back(key);
    }
#else
    query_keys.push_back(iKey);
#endif
  }

  // shuffle the queries
  std::shuffle(query_keys.begin(), query_keys.end(), g);

  // Move data to GPU
  GpuTimer query_timer;
  query_timer.timerStart();
  uint32_t *d_queries, *d_results;
  nvml_start();
  penguinStartStatCollection();
  /* CHECK_ERROR(memoryUtil::deviceAlloc(d_queries, numQueries)); */
  /* CHECK_ERROR(memoryUtil::deviceAlloc(d_results, numQueries)); */
  cudaMallocManaged((void**)&d_queries, sizeof(uint32_t) * numQueries);
  cudaMallocManaged((void**)&d_results, sizeof(uint32_t) * numQueries);
  printf("memory done\n");
  /* CHECK_ERROR(memoryUtil::hostAlloc(d_queries, numQueries)); */
  /* CHECK_ERROR(memoryUtil::hostAlloc(d_results, numQueries)); */
  CHECK_ERROR(memoryUtil::cpyToDevice(query_keys.data(), d_queries, numQueries));
  memset((void*)d_results, 0, sizeof(uint32_t) * numQueries);

#if SC19
        cudaMemAdvise(d_queries, sizeof(uint32_t) * numQueries, cudaMemAdviseSetAccessedBy, 0);
  penguinSetNoMigrateRegion(d_queries, sizeof(uint32_t) * numQueries, 0, true);
        /* cudaMemAdvise(d_results, sizeof(uint32_t) * numQueries, cudaMemAdviseSetAccessedBy, 0); */
  /* penguinSetNoMigrateRegion(d_results, sizeof(uint32_t) * numQueries, 0, true); */
        cudaMemAdvise(d_results, sizeof(uint32_t) * numQueries, cudaMemAdviseSetPreferredLocation, 0);
#endif


  /* GpuTimer query_timer; */
  /* query_timer.timerStart(); */
  btree.searchKeys(d_queries, d_results, numQueries, SourceT::DEVICE);
  nvml_stop();
  query_timer.timerStop();
  penguinStopStatCollection();


  // Copy results back
  CHECK_ERROR(memoryUtil::cpyToHost(d_results, query_results.data(), numQueries));

  // Validate
  uint32_t exist_count = 0;
  for (uint32_t iKey = 0; iKey < numQueries; iKey++) {
    if (query_keys[iKey] < numKeys) {
      exist_count++;
      if (query_results[iKey] != query_keys[iKey]) {
        printf("Error validating queries (Key = %i, Value = %i) found (Value = %i)\n",
               query_keys[iKey],
               query_keys[iKey],
               query_results[iKey]);
        exit(0);
      }
    } else {
      if (query_results[iKey] != 0) {
        printf(
            "Error validating queries (Key = %i, Value = NOT_FOUND) found (Value = %i)\n",
            query_keys[iKey],
            query_results[iKey]);
        exit(0);
      }
    }
  }

  // output
  printf("SUCCESS. ([%0.2f%%] queries exist in search.)\n",
         float(exist_count) / float(numQueries) * 100.0);

  printf("Build: %i pairs in %f ms (%0.2f MKeys/sec)\n",
         numKeys,
         build_timer.getMsElapsed(),
         float(numKeys) * 1e-6 / build_timer.getSElapsed());

  printf("Query: %i pairs in %f ms (%0.2f MKeys/sec)\n",
         numQueries,
         query_timer.getMsElapsed(),
         float(numQueries) * 1e-6 / query_timer.getSElapsed());
  printf("GPU.Parser.Time: %f\n", query_timer.getMsElapsed());

  printf("Tree size: %f GiBs.\n", float(btree.compute_usage()));
  // cleanup
  cudaFree(d_keys);
  cudaFree(d_values);
  cudaFree(d_queries);
  cudaFree(d_results);
  btree.free();
  return 0;
}
