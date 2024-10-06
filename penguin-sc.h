/* Library for calling the new ioctl for prioritizing pages on the GPU */
/* SC */

#ifndef PENGUIN
#define PENGUIN // the penguin library

#define PENGUIN_PRIORITIZED_GPU_IOCTL_NUM 75
#define PENGUIN_NO_MIGRATE_IOCTL_NUM 76
#define PENGUIN_START_STAT_COLLECTION_IOCTL_NUM 77
#define PENGUIN_STOP_STAT_COLLECTION_IOCTL_NUM 78
#define PENGUIN_QUICK_MIGRATE_IOCTL_NUM 79
#define PENGUIN_ACCESS_COUNTER_ENABLE 80

/* #define PENGUIN_MIN_PREFETCH (32*1024*1024) */
#define PENGUIN_MIN_PREFETCH (8*1024*1024)

#include <stdio.h>
#include <string.h>
#include <iostream>
#include <map>
#include <set>
#include <vector>
#include <algorithm>
#include <stdint.h>
#include <dirent.h>
#include <sys/ioctl.h>
#include <cuda_runtime.h>
#include <nvml.h>
#include <iterator>

#define NVML_PROFILER 1
#define NVML_TX 0
unsigned int nvml_running = 0;
pthread_t monitor;

#define PSF_DIR "/proc/self/fd"
#define NVIDIA_UVM_PATH "/dev/nvidia-uvm"

static int nvidia_uvm_fd = -1;

/* static volatile unsigned counter = 0; */

unsigned long long MBs = 5451ULL;
unsigned long long gpu_memory = 1 *  MBs * 1024ULL * 1024ULL;
unsigned long long available = gpu_memory;
unsigned long long pinned_memory = 0;
unsigned long long SCAvail = gpu_memory;
std::map<void*, unsigned long long> SCGPUResidentAllocs;
// add code to evict anything whose use is over
// add code to prioritize higher AD temporal region over lower AD temporal region

enum State {
    PENGUIN_STATE_UNKNOWN,
    PENGUIN_STATE_GPU,
    PENGUIN_STATE_GPU_PINNED,
    PENGUIN_STATE_GPU_PINNED_PART,
    PENGUIN_STATE_HOST,
    PENGUIN_STATE_AC,
    PENGUIN_STATE_MAX
};

enum Decision {
    PENGUIN_DEC_NONE,
    PENGUIN_DEC_HOST_PIN,
    PENGUIN_DEC_GPU_PIN,
    PENGUIN_DEC_GPU_HOST_PARTIAL_PIN,
    PENGUIN_DEC_MIGRATE_ON_DEMAND,
    PENGUIN_DEC_ITERATION_MIGRATION,
    PENGUIN_DEC_ITERATION_MIGRATION_PLUS_GPU_HOST_PIN,
    PENGUIN_DEC_ACCESS_COUNTER,
    PENGUIN_DEC_MAX
};

typedef enum {
    PENGUIN_OK,
    PENGUIN_ERR_PATH,
    PENGUIN_ERR_IOCTL,
    PENGUIN_ERR_NOT_IMPLEMENTED
} penguin_error_t;

typedef struct 
{
    void *base;
    size_t length;
    uint8_t uuid[16];
    int status;
} penguin_prioritized_ioctl_params;

typedef struct 
{
    uint8_t uuid[16];
    unsigned mimc_gran;
    unsigned momc_gran;
    unsigned mimc_use_limit;
    unsigned momc_use_limit;
    unsigned threshold;
    bool enable_mimc;
    bool enable_momc;
    int status;
} penguin_enable_access_counter_param;

typedef struct 
{
    void *base;
    size_t length;
    bool quick_migrate;
} penguin_quick_migrate_ioctl_params;

typedef struct 
{
    void *base;
    size_t length;
    bool ignore_notifications;
    int status;
} penguin_ignore_notif_ioctl_params;

typedef struct
{
    int status;
} penguin_start_stat_collection_params;

typedef struct
{
    int status;
}  penguin_stop_stat_collection_params;

std::map<void*, State> SCState;

std::map<void*, unsigned long long> allocation_size_map;
std::map<void*, unsigned long long> allocation_ac_map;
std::map<void*, unsigned long long> allocation_pd_bidx_map;
std::map<void*, unsigned long long> allocation_pd_bidy_map;
std::map<void*, unsigned long long> allocation_pd_phi_map;
std::map<void*, unsigned long long> allocation_wss_map;
std::vector<std::pair<void*, unsigned long long>> ad_vector;
std::vector<std::pair<void*, unsigned long long>> wss_vector;

std::map<unsigned, unsigned long long> aid_ac_map;
std::map<unsigned, void*> aid_allocation_map;
std::map<unsigned, unsigned long long> aid_wss_map_iterdep;
std::map<unsigned, unsigned long long> aid_wss_map;
std::map<unsigned, bool> aid_pchase_map;
std::map<unsigned, unsigned> aid_invocation_id_map;
std::map<unsigned, bool> aid_ac_incomp_map;

// Badly named
std::map<unsigned, unsigned long long> aid_ac_map_reuse;
std::map<unsigned, void*> aid_allocation_map_reuse;
std::map<unsigned, unsigned> aid_invocation_id_map_reuse;

std::map<unsigned, std::set<void*>> global_mmg_invid_alloc_list;
std::map<void*, std::map<unsigned, unsigned>> global_alloc_inv_resinv_map;
std::map<void*, unsigned long long> global_mmg_alloc_ac_map;
std::vector<std::pair<void*, unsigned long long>> global_mmg_alloc_ac_vector;
std::map<void*, unsigned> global_alloc_firstuse_map;
std::map<unsigned, std::vector<std::pair<void*, unsigned>>> global_invid_sorted_reuse_alloc_map;

extern "C"
void add_aid_ac_map_reuse(unsigned aid, unsigned long long ac) {
    /* std::cout << "adi aid ac map resue " << aid  << " " << ac << "\n"; */
    aid_ac_map_reuse[aid] = ac;
}

extern "C"
void add_aid_allocation_map_reuse(unsigned aid, void* allocation) {
    /* std::cout<< "add_aid_allocation_map reuse" << aid << " " << allocation << std::endl; */
    aid_allocation_map_reuse[aid] = allocation;
}

extern "C"
void add_aid_invocation_map_reuse(unsigned aid, unsigned invocation_id) {
    /* std::cout<< "add_aid_invocation_map reuse" << aid << " " << invocation_id << std::endl; */
    aid_invocation_id_map_reuse[aid] = invocation_id;
}

std::map<void*, unsigned> AllocationToItersPerBatchMap; 
std::map<void*, unsigned> AllocationToLengthMap; 

// working data structures
std::map<void*, std::map<unsigned, unsigned long long>> AllocationToInvocationIDtoADMap;
std::map<void*, std::map<unsigned, Decision>> AllocationToInvocationIDtoDecisionMap;

std::map<unsigned, std::map<void*, Decision>> InvocationIDtoAllocationToADMap;
std::map<unsigned, std::map<void*, Decision>> InvocationIDtoAllocationToDecisionMap;
std::map<unsigned, std::map<void*, unsigned long long>> InvocationIDtoAllocationToPartialSize;
std::map<void*, Decision> AllocationToCommonDecisionMap;
std::map<void*, unsigned long long> AllocationToPartialSizeMap;

std::map<void*, Decision> AllocationToDecisionMap;
std::map<void*, bool> AllocationToPrefetchBoolMap;
std::map<void*, unsigned long long> AllocationToPrefetchSizeMap;
std::map<void*, unsigned long long> AllocationToPrefetchItersPerBatchMap;
std::map<void*, unsigned long long> AllocationToPchaseMap;

static bool is_iterative = false;

// this bool helps helper function at kernel invocations to quickly decide if 
// there are any API calls to be made or it can just skip
std::map<unsigned, bool> InvocationIDtoDecisionBoolMap;
std::set<unsigned> InvocationIDs;

// state maps
std::map<void*, State> AllocationStateMap;
std::map<void*, unsigned long long> AllocationGPUResStart;
std::map<void*, unsigned long long> AllocationGPUResStop;
std::map<void*, bool> AllocationStateIterBoolMap;
std::map<void*, unsigned> AllocationStateIterStartMap;
std::map<void*, unsigned> AllocationStateIterStopMap;

unsigned max_invid = 0;

// Do we need "C" linkage?

void* round_down(void* addr) {
    unsigned long long ad = (unsigned long long) addr;
    return (void*) ((ad >> 16) << 16);
}

extern "C"
penguin_error_t penguinSetPrioritizedLocation(void *base, size_t length,
        unsigned proc_id) {

    DIR *d;
    struct dirent *dir;
    char psf_path[512];
    char *psf_realpath;
    penguin_prioritized_ioctl_params request;
    int status;


    /* std::cout << "set prioritized location " << base << std::endl; */

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    for(int i = 0; i < 16; i++) {
        request.uuid[i] = prop.uuid.bytes[i];
        /* printf("%u", prop.uuid.bytes[i]); */
    }

    request.base = base;
    request.length = length;

    d = opendir(PSF_DIR);
    if (d)
    {
        while ((dir = readdir(d)) != NULL)
        {
            if (dir->d_type == DT_LNK)
            {
                sprintf(psf_path, "%s/%s", PSF_DIR, dir->d_name);
                psf_realpath = realpath(psf_path, NULL);
                if (strcmp(psf_realpath, NVIDIA_UVM_PATH) == 0)
                    nvidia_uvm_fd = atoi(dir->d_name);
                free(psf_realpath);
                if (nvidia_uvm_fd >= 0)
                    break;
            }
        }
        closedir(d);
    }
    if (nvidia_uvm_fd < 0)
    {
        fprintf(stderr, "Cannot open %s\n", PSF_DIR);
        return PENGUIN_ERR_PATH;
    }
    if ((status = ioctl(nvidia_uvm_fd, PENGUIN_PRIORITIZED_GPU_IOCTL_NUM, &request)) != 0)
    {
        fprintf(stderr, "error: %d\n", status);
        return PENGUIN_ERR_IOCTL;
    }
    return PENGUIN_OK;
}

extern "C"
penguin_error_t penguinSetQuickMigrate(void *base, size_t length,
        bool quick_migrate) {

    DIR *d;
    struct dirent *dir;
    char psf_path[512];
    char *psf_realpath;
    penguin_quick_migrate_ioctl_params request;
    int status;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    request.base = base;
    request.length = length;

    d = opendir(PSF_DIR);
    if (d)
    {
        while ((dir = readdir(d)) != NULL)
        {
            if (dir->d_type == DT_LNK)
            {
                sprintf(psf_path, "%s/%s", PSF_DIR, dir->d_name);
                psf_realpath = realpath(psf_path, NULL);
                if (strcmp(psf_realpath, NVIDIA_UVM_PATH) == 0)
                    nvidia_uvm_fd = atoi(dir->d_name);
                free(psf_realpath);
                if (nvidia_uvm_fd >= 0)
                    break;
            }
        }
        closedir(d);
    }
    if (nvidia_uvm_fd < 0)
    {
        fprintf(stderr, "Cannot open %s\n", PSF_DIR);
        return PENGUIN_ERR_PATH;
    }
    if ((status = ioctl(nvidia_uvm_fd, PENGUIN_QUICK_MIGRATE_IOCTL_NUM, &request)) != 0)
    {
        fprintf(stderr, "error: %d\n", status);
        fprintf(stderr, "debuggy\n");
        return PENGUIN_ERR_IOCTL;
    }
    return PENGUIN_OK;
}

extern "C"
penguin_error_t penguinSetNoMigrateRegion(void *base, size_t length,
        unsigned proc_id, bool setNoMigrate) {

    DIR *d;
    struct dirent *dir;
    char psf_path[512];
    char *psf_realpath;
    penguin_ignore_notif_ioctl_params request;
    int status;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    request.base = base;
    request.length = length;
    request.ignore_notifications = setNoMigrate;

    d = opendir(PSF_DIR);
    if (d)
    {
        while ((dir = readdir(d)) != NULL)
        {
            if (dir->d_type == DT_LNK)
            {
                sprintf(psf_path, "%s/%s", PSF_DIR, dir->d_name);
                psf_realpath = realpath(psf_path, NULL);
                if (strcmp(psf_realpath, NVIDIA_UVM_PATH) == 0)
                    nvidia_uvm_fd = atoi(dir->d_name);
                free(psf_realpath);
                if (nvidia_uvm_fd >= 0)
                    break;
            }
        }
        closedir(d);
    }
    if (nvidia_uvm_fd < 0)
    {
        fprintf(stderr, "Cannot open %s\n", PSF_DIR);
        return PENGUIN_ERR_PATH;
    }
    if ((status = ioctl(nvidia_uvm_fd, PENGUIN_NO_MIGRATE_IOCTL_NUM, &request)) != 0)
    {
        fprintf(stderr, "error: %d\n", status);
        return PENGUIN_ERR_IOCTL;
    }
    return PENGUIN_OK;
}

extern "C"
void penguinSuperPrefetch(void *base, size_t length, unsigned iter, unsigned iterPerBatch, size_t max) {
    /* counter++; */
    if (length == 0) return;
    if ((iter % iterPerBatch) == 0) {
        int prefnum = iter / iterPerBatch;
        if ((prefnum+1) * length > max) {
            length = max - (prefnum) * length;
            /* return; // not return, use new value */
        }
        printf("base = %p prefnum = %d; iter = %d; length = %llu\n", base, prefnum, iter, length);
        // TODO: prefetch back, but only if there is memory pressure.
        auto pref_addr = prefnum*length;
        if(AllocationGPUResStart[base] <= pref_addr &&
                (pref_addr + length) < AllocationGPUResStop[base]){
            return;
        }
        /* std::cout << "pref_addr = " << pref_addr << std::endl; */
        /* std::cout << "alloc start on gpu = " << AllocationGPUResStart[base] << std::endl; */
        /* std::cout << "alloc stop on gpu = " << AllocationGPUResStop[base] << std::endl; */

        if(prefnum > 0 && available == 0) {
            /* std::cout << "revpref\n"; */
            cudaMemPrefetchAsync((char*)base + ((prefnum-1)*length), length, -1, 0 );
            AllocationGPUResStart[base] += length;
            AllocationGPUResStop[base] += length;
        }
        /* std::cout << "pre\n"; */
        cudaMemPrefetchAsync((char*)base + (prefnum*length), length, 0, 0 );
    }
    return;
}

extern "C"
void penguinSuperPrefetchWrapper(unsigned iter) {
    //get parameters from runtime maps and call penguinSuperPrefetch
    /* std::cout << "penguinSuperPrefetchWrapper " << iter << "\n"; */
    for(auto memalloc = AllocationToPrefetchBoolMap.begin();
            memalloc != AllocationToPrefetchBoolMap.end(); memalloc++) {
        auto base = memalloc->first;
        if(AllocationToPrefetchBoolMap[base] == true) {
            /* std::cout << "pspw says prefetch iter " << base << " " << iter << "\n"; */
            unsigned long long length = AllocationToPrefetchSizeMap[base];
            auto dsize = allocation_size_map[base];
            auto iterPerBatch = AllocationToPrefetchItersPerBatchMap[base];
            penguinSuperPrefetch(base, length, iter, iterPerBatch, dsize);
            /* penguinSuperPrefetch(base, 512*1024*1024, iter, 128, dsize); */
        }
    }
}

void* nvml_monitor(void* argp) {
    /* printf("ellloooo\n"); */
    auto status = nvmlInit();
    if(status != NVML_SUCCESS) {
        printf("unable to init nvml\n");
        return NULL;
    }
    nvmlDevice_t device_;
    unsigned throughput;
    unsigned long long total = 0;
    unsigned long long count = 0;
    status = nvmlDeviceGetHandleByIndex(0, &device_);
    while(nvml_running == 1) {
#if NVML_TX
        status = nvmlDeviceGetPcieThroughput(device_, NVML_PCIE_UTIL_TX_BYTES, &throughput);
#else
        status = nvmlDeviceGetPcieThroughput(device_, NVML_PCIE_UTIL_RX_BYTES, &throughput);
#endif
        /* printf("throughput = %u\n", throughput); */
        total += throughput;
        count ++;
    }
#if NVML_TX
    printf("total TX PCIe = %llu\n", total);
#else 
    printf("total RX PCIe = %llu\n", total);
#endif
    /* printf("count = %u\n", count); */
    return NULL;
}

void nvml_start() {
#if NVML_PROFILER
    nvml_running = 1;
    pthread_create(&monitor, NULL, nvml_monitor, NULL);
#endif
}

void nvml_stop() {
#if NVML_PROFILER
    nvml_running = 0;
    /* printf("nvml: running = 0\n"); */
    pthread_join(monitor, NULL);
#endif
}

bool ac_enabled = false;

extern "C"
penguin_error_t penguinEnableAccessCounters() {
    if(ac_enabled == true) {
        return PENGUIN_OK;
    }
    ac_enabled = true;
    DIR *d;
    struct dirent *dir;
    char psf_path[512];
    char *psf_realpath;
    penguin_enable_access_counter_param request;
    int status;

    request.enable_mimc = true;
    request.enable_momc = false;
    request.mimc_gran  = 1;
    request.momc_gran  = 1;
    request.mimc_use_limit  = 4;
    request.momc_use_limit  = 4;
    request.threshold  = 256;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    for(int i = 0; i < 16; i++) {
        request.uuid[i] = prop.uuid.bytes[i];
        printf("%u", prop.uuid.bytes[i]);
    }

    fprintf(stderr, "enable access counters\n");
    d = opendir(PSF_DIR);
    if (d)
    {
        while ((dir = readdir(d)) != NULL)
        {
            if (dir->d_type == DT_LNK)
            {
                sprintf(psf_path, "%s/%s", PSF_DIR, dir->d_name);
                psf_realpath = realpath(psf_path, NULL);
                if (strcmp(psf_realpath, NVIDIA_UVM_PATH) == 0)
                    nvidia_uvm_fd = atoi(dir->d_name);
                free(psf_realpath);
                if (nvidia_uvm_fd >= 0)
                    break;
            }
        }
        closedir(d);
    }
    if (nvidia_uvm_fd < 0)
    {
        fprintf(stderr, "Cannot open %s\n", PSF_DIR);
        return PENGUIN_ERR_PATH;
    }
    if ((status = ioctl(nvidia_uvm_fd, PENGUIN_ACCESS_COUNTER_ENABLE, &request)) != 0)
    {
        fprintf(stderr, "error: %d\n", status);
        fprintf(stderr, "debuggy\n");
        return PENGUIN_ERR_IOCTL;
    }
    return PENGUIN_OK;
}

extern "C"
penguin_error_t penguinStartStatCollection() {
    DIR *d;
    struct dirent *dir;
    char psf_path[512];
    char *psf_realpath;
    penguin_start_stat_collection_params request;
    int status;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    d = opendir(PSF_DIR);
    if (d)
    {
        while ((dir = readdir(d)) != NULL)
        {
            if (dir->d_type == DT_LNK)
            {
                sprintf(psf_path, "%s/%s", PSF_DIR, dir->d_name);
                psf_realpath = realpath(psf_path, NULL);
                if (strcmp(psf_realpath, NVIDIA_UVM_PATH) == 0)
                    nvidia_uvm_fd = atoi(dir->d_name);
                free(psf_realpath);
                if (nvidia_uvm_fd >= 0)
                    break;
            }
        }
        closedir(d);
    }
    if (nvidia_uvm_fd < 0)
    {
        fprintf(stderr, "Cannot open %s\n", PSF_DIR);
        return PENGUIN_ERR_PATH;
    }
    if ((status = ioctl(nvidia_uvm_fd, PENGUIN_START_STAT_COLLECTION_IOCTL_NUM, &request)) != 0)
    {
        fprintf(stderr, "error: %d\n", status);
        return PENGUIN_ERR_IOCTL;
    }
    return PENGUIN_OK;
}

extern "C"
penguin_error_t penguinStopStatCollection() {
    DIR *d;
    struct dirent *dir;
    char psf_path[512];
    char *psf_realpath;
    penguin_stop_stat_collection_params request;
    int status;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    d = opendir(PSF_DIR);
    if (d)
    {
        while ((dir = readdir(d)) != NULL)
        {
            if (dir->d_type == DT_LNK)
            {
                sprintf(psf_path, "%s/%s", PSF_DIR, dir->d_name);
                psf_realpath = realpath(psf_path, NULL);
                if (strcmp(psf_realpath, NVIDIA_UVM_PATH) == 0)
                    nvidia_uvm_fd = atoi(dir->d_name);
                free(psf_realpath);
                if (nvidia_uvm_fd >= 0)
                    break;
            }
        }
        closedir(d);
    }
    if (nvidia_uvm_fd < 0)
    {
        fprintf(stderr, "Cannot open %s\n", PSF_DIR);
        return PENGUIN_ERR_PATH;
    }
    if ((status = ioctl(nvidia_uvm_fd, PENGUIN_STOP_STAT_COLLECTION_IOCTL_NUM, &request)) != 0)
    {
        fprintf(stderr, "error: %d\n", status);
        return PENGUIN_ERR_IOCTL;
    }
    return PENGUIN_OK;
}


extern "C"
void add_invocation_id(unsigned invid) {
    InvocationIDs.insert(invid);
    return;
}

// TODO: fix this ASAP
extern "C"
void addIntoAllocationMap(void** ptr, unsigned long long size) {
    void* p = (void*) *ptr;
    /* std::cout << "added to allocation map, " << p << " " << size << "\n"; */
    allocation_size_map[p] = size;
    return;
}

extern "C"
void printAllocationMap() {
    /* std::cout << "size map\n"; */
    for (auto a = allocation_size_map.begin(); a != allocation_size_map.end(); a++) {
        /* std::cout << a->first << " " << a->second << "\n"; */
    }
}

extern "C"
unsigned long long getAllocationSize(void* ptr) {
    return allocation_size_map[ptr];
}

extern "C"
void addACToAllocation(void* ptr, unsigned long long count) {
    /* void* p = (void*) *ptr; */
    /* std::cout << ptr << std::endl; */
    allocation_ac_map[ptr] += count;
    return;
}

extern "C"
void printACToAllocationMap() {
    /* std::cout << "ac map\n"; */
    for (auto a = allocation_ac_map.begin(); a != allocation_ac_map.end(); a++) {
        /* std::cout << a->first << " " << a->second << "\n"; */
    }
}

extern "C"
float getAccessDensity(void* ptr) {
    return (float) allocation_ac_map[ptr] / (float) allocation_size_map[ptr];
}

// TODO : add a method for clearing the allocation_ac_map

extern "C"
unsigned long long accessCountForAllocation(void* ptr) {
    return allocation_ac_map[ptr];
}

extern "C"
void add_pd_bidx_to_allocation(void* ptr, unsigned pd_bidx) {
    if(allocation_pd_bidx_map[ptr] < pd_bidx) {
        allocation_pd_bidx_map[ptr] = pd_bidx;
    }
}

extern "C"
void add_pd_bidy_to_allocation(void* ptr, unsigned pd_bidy) {
    if(allocation_pd_bidy_map[ptr] < pd_bidy) {
        allocation_pd_bidy_map[ptr] = pd_bidy;
    }
}

extern "C"
void add_pd_phi_to_allocation(void* ptr, unsigned pd_phi) {
    if(allocation_pd_phi_map[ptr] < pd_phi) {
        allocation_pd_phi_map[ptr] = pd_phi;
    }
}

extern "C"
unsigned get_pd_bidx(void* ptr) {
    /* std::cout <<  "pd_bidx = " << allocation_pd_bidx_map[ptr] << "\n"; */
    return allocation_pd_bidx_map[ptr];
}

extern "C"
unsigned get_pd_bidy(void* ptr) {
    /* std::cout <<  "pd_bidy = " << allocation_pd_bidy_map[ptr] << "\n"; */
    return allocation_pd_bidy_map[ptr];
}

extern "C"
unsigned get_pd_phi(void* ptr) {
    /* std::cout <<  "pd_phi = " << allocation_pd_phi_map[ptr] << "\n"; */
    return allocation_pd_phi_map[ptr];
}

extern "C"
void print_pd_bidx_map() {
    /* std::cout << "bidx map\n"; */
    for (auto a = allocation_pd_bidx_map.begin(); a != allocation_pd_bidx_map.end(); a++) {
        /* std::cout << a->first << " " << a->second << "\n"; */
    }
}

extern "C"
void print_pd_bidy_map() {
    /* std::cout << "bidy map\n"; */
    for (auto a = allocation_pd_bidy_map.begin(); a != allocation_pd_bidy_map.end(); a++) {
        /* std::cout << a->first << " " << a->second << "\n"; */
    }
}

extern "C"
void print_pd_phi_map() {
    /* std::cout << "phi map\n"; */
    for (auto a = allocation_pd_phi_map.begin(); a != allocation_pd_phi_map.end(); a++) {
        /* std::cout << a->first << " " << a->second << "\n"; */
    }
}

extern "C"
void add_wss_to_map(void *ptr, unsigned long long wss, unsigned aid) {
    aid_wss_map[aid] = wss;
    if(allocation_wss_map[ptr] < wss) {
        allocation_wss_map[ptr] = wss;
    }
}

extern "C"
void print_wss_map() {
    /* std::cout << "wss map\n"; */
    for (auto a = allocation_wss_map.begin(); a != allocation_wss_map.end(); a++) {
        /* std::cout << a->first << " " << a->second << "\n"; */
    }
}

extern "C"
unsigned long long get_wss(void* ptr) {
    return allocation_wss_map[ptr];
}

extern "C"
void print_value_i32(uint64_t value) {
    /* std::cout << "value(i32) = " << value << std::endl; */
}

extern "C"
void print_value_i64(uint64_t value) {
    /* std::cout << "value(i64) = " << value << std::endl; */
}

extern "C"
void print_value_f32(float value) {
    /* std::cout << "value(f32) = " << value << std::endl; */
}

extern "C"
void print_value_f64(double value) {
    /* std::cout << "value(f64) = " << value << std::endl; */
}

extern "C"
float compute_access_density(void* ptr, unsigned numThreads, unsigned loopIters, unsigned long long size) {
    float ad = ((float)numThreads * (float)loopIters) / (float)size;
    /* std::cout << "ad is " << ad << "\n"; */
    return ad;
}

bool sortfunc(std::pair<void*, unsigned long long> &a,  std::pair<void*, unsigned long long> &b){
    return a.second > b.second;
}

bool sortfuncf(std::pair<void*, float> &a,  std::pair<void*, float> &b){
    return a.second > b.second;
}

bool sortfunc_l_v_u(std::pair<void*, unsigned > &a,  std::pair<void*, unsigned > &b){
    return a.second < b.second;
}



/* extern "C" */
/* void sort_wss() { */
/*   std::copy(allocation_wss_map.begin(), allocation_wss_map.end(),back_inserter<std::vector<std::pair<void*, unsigned long long> > >(wss_vector)); */
/*   std::sort(wss_vector.begin(), wss_vector.end(), sortfunc); */
  /* std::cout << "sorted wss\n"; */
/*   for(auto wss = wss_vector.begin(); wss != wss_vector.end(); wss++) { */
    /* std::cout << wss->first << " " << wss->second << "\n"; */
/*   } */
/* } */

extern "C"
unsigned long long estimate_working_set2(unsigned long long wss_per_tb, unsigned bdimx, unsigned bdimy) {
    /* std::cout << "wss 2 = " <<  wss_per_tb << std::endl; */
    unsigned long long numTBs = (1536)/(bdimx * bdimy);
    numTBs = numTBs * 82;
    /* std::cout << "wss 2 sum = " << wss_per_tb * numTBs << std::endl; */
    return wss_per_tb * numTBs;
}

extern "C"
unsigned estimate_working_set(unsigned long long pd_bidx, unsigned long long pd_bidy, unsigned long long pd_phi, unsigned loopiters, unsigned bdimx, unsigned bdimy, unsigned gdimx, unsigned gdimy) {
    unsigned long long max = 0;
    if((pd_phi * loopiters) > max) {
        max = pd_phi * loopiters;
    }
    unsigned long long numTBs = (1536)/(bdimx * bdimy);
    numTBs = numTBs * 82;
    /* std::cout << "bdimx,y,liters = " << bdimx << " " << bdimy << " " << loopiters << "\n"; */
    std::cout << "numTBs = " << numTBs << "\n";
    unsigned long long concTBx = (gdimx > numTBs) ? numTBs: gdimx;
    unsigned long long concTBy = (gdimx > numTBs) ? 1 : (numTBs/gdimx);
    if((pd_bidx * concTBx) > max) {
        max = pd_bidx * concTBx;
    }
    if((pd_bidy * concTBy) > max) {
        max = pd_bidy * concTBy;
    }
    std::cout << "params = " << pd_phi << " " << pd_bidx << " " << loopiters << " " << gdimx << " " << concTBx << " " << "\n";
    std::cout << "working set = " << max << "\n";
    return max;
}
/* float compute_access_pattern_dx(void* ptr, */ 


extern "C"
void* identify_memory_allocation(void* addr) {
    unsigned long long addr_ull = (unsigned long long) addr;
    for (auto a = allocation_size_map.begin(); a != allocation_size_map.end(); a++) {
        unsigned long long alloc_ull = (unsigned long long) a->first;
        unsigned long long size = a->second;
        if(alloc_ull < addr_ull && addr_ull < alloc_ull + size) {
            return a->first;
        }
    }
    return 0;
}

extern "C"
unsigned estimate_working_set_iteration(unsigned gdimx, unsigned gdimy, unsigned bdimx, unsigned bdimy) {
    return 42;
}

extern "C"
void add_aid_pchase_map(unsigned aid, void* addr, bool pchase) {
    /* std::cout << "added to pchase map " << aid << " " << addr << "\n"; */
    aid_pchase_map[aid] = pchase;
    aid_allocation_map[aid] = addr;
}

extern "C"
void add_aid_ac_incomp_map(unsigned aid, bool incomp) {
    aid_ac_incomp_map[aid] = incomp;
}

extern "C"
void add_aid_wss_map_iterdep(unsigned aid, unsigned long long wss) {
    /* std::cout << "added to iterdep map " << aid << " " << wss << "\n"; */
    aid_wss_map_iterdep[aid] = wss;
}

extern "C"
void add_aid_wss_map(unsigned aid, unsigned long long wss) {
    aid_wss_map[aid] = wss;
}

extern "C"
void add_aid_ac_map(unsigned aid, unsigned long long ac) {
    aid_ac_map[aid] = ac;
}

extern "C"
void add_aid_allocation_map(unsigned aid, void* allocation) {
    /* std::cout<< "add_aid_allocation_map " << aid << " " << allocation << std::endl; */
    aid_allocation_map[aid] = allocation;
}

extern "C"
void add_aid_invocation_map(unsigned aid, unsigned invocation_id) {
    aid_invocation_id_map[aid] = invocation_id;
}

extern "C"
bool is_iterdep_access(unsigned aid) {
    return (aid_wss_map_iterdep.find(aid) != aid_wss_map_iterdep.end());
}

extern "C"
void print_aid_wss_map_iterdep() {
    /* std::cout << "aid wss map (iterdep)\n"; */
    for (auto a = aid_wss_map_iterdep.begin(); a != aid_wss_map_iterdep.end(); a++) {
        /* std::cout << a->first << " " << a->second << "\n"; */
    }
}

extern "C"
void process_iterdep_access() {
    for (auto a = aid_wss_map_iterdep.begin(); a != aid_wss_map_iterdep.end(); a++) {
        /* std::cout << a->first << " " << a->second << "\n"; */
    }
}

extern "C"
void process_all_accesses() {
}

// this function is for all non-iterative kernels (and non iteration-dependent accesses within iterative kernels)
extern "C"
void perform_memory_management_global() {
    // many maps
    /* std::cout << "mm global \n"; */
    return;
}

extern "C"
void perform_memory_management_iterative() {
    std::cout << "mm iterative \n";
    is_iterative = true;
    /* std::cout << "data from reuse\n"; */
    for (auto a = aid_ac_map_reuse.begin(); a != aid_ac_map_reuse.end(); a++) {
        /* std::cout << a->first << "  " << a->second << " " << aid_allocation_map_reuse[a->first] << std::endl; */
        auto alloc = aid_allocation_map_reuse[a->first];
        auto invid = aid_invocation_id_map_reuse[a->first];
        global_mmg_invid_alloc_list[invid].insert(alloc);
    }
    /* std::cout << "invid to alloc list\n"; */
    /* unsigned max_invid = 0; */
    for(auto i = global_mmg_invid_alloc_list.begin(); i != global_mmg_invid_alloc_list.end(); i++) {
        /* std::cout << i->first << " :  "; */
        if(i->first > max_invid) {
            max_invid = i->first;
        }
        for(auto a = i->second.begin(); a != i->second.end(); a++) {
            /* std::cout << *a << " "; */
            if(global_alloc_firstuse_map.find(*a) == global_alloc_firstuse_map.end()) {
                global_alloc_firstuse_map[*a] = i->first;
            }
        }
        /* std::cout << std::endl; */
    }
    /* std::cout << "max invid = " << max_invid << std::endl; */
    /* for each allocation, for each invocation, compute the next reuse */
    for(auto alloc = allocation_size_map.begin(); alloc != allocation_size_map.end(); alloc++) {
        /* std::cout << alloc->first << std::endl; */
        for (auto c = 1; c <= max_invid; c++) {
            /* std::cout << c << std::endl; */
            unsigned nearest_reuse = 1000 ;
            for(auto i = global_mmg_invid_alloc_list.begin(); i != global_mmg_invid_alloc_list.end(); i++) {
                if(i->second.find(alloc->first) != i->second.end()) {
                    /* std::cout << "reuse at " << i->first << std::endl; */
                    if(i->first > c && i->first < nearest_reuse) {
                        nearest_reuse = i->first;
                    }
                }
            }
            /* std::cout << "nearest reuse is " << nearest_reuse << std::endl; */
            global_alloc_inv_resinv_map[alloc->first][c] = nearest_reuse;
        }
    }
    // for each invid, list all allocations, sorted by reuse
    /* std::cout << " each invid, list all allocations, sorted by reuse\n"; */
    for (auto i = global_mmg_invid_alloc_list.begin(); i != global_mmg_invid_alloc_list.end(); i++) {
        /* std::cout << i->first << " :  "; */
        for(auto a = i->second.begin(); a != i->second.end(); a++) {
            /* std::cout << *a << " "; */
            void* alloc = *a;
            unsigned reuse = global_alloc_inv_resinv_map[*a][i->first];
            /* std::cout << reuse << " "; */
            if(global_alloc_inv_resinv_map[*a][i->first] == 1000) {
                /* std::cout << global_alloc_firstuse_map[*a] << " "; */
                reuse = global_alloc_firstuse_map[*a] + max_invid;
            }
            global_invid_sorted_reuse_alloc_map[i->first].push_back(std::pair<void*, unsigned>(alloc, reuse));
            /* std::cout << std::endl; */
        }
        /* global_invid_sorted_reuse_alloc_map[i->first] = sorted_reuse_vector; */
        std::sort(global_invid_sorted_reuse_alloc_map[i->first].begin(),
                global_invid_sorted_reuse_alloc_map[i->first].end(), sortfunc_l_v_u);
        /* std::cout << "sorted by reuse, allocations\n"; */
        for(auto ra = global_invid_sorted_reuse_alloc_map[i->first].begin();
                ra != global_invid_sorted_reuse_alloc_map[i->first].end(); ra++) {
            /* std::cout << ra->first << " " ; */
        }
        /* std::cout << std::endl; */
    }
    //compute Global Locality
    /* std::map<void*, unsigned long long> mmg_alloc_ac_map; */
    for(auto a = aid_ac_map_reuse.begin(); a != aid_ac_map_reuse.end(); a++) {
        auto alloc = aid_allocation_map_reuse[a->first];
        global_mmg_alloc_ac_map[alloc] += a->second;
    }
    /* std::cout << "global locality\n"; */
    for(auto a = global_mmg_alloc_ac_map.begin(); a != global_mmg_alloc_ac_map.end(); a++) {
        /* std::cout << "GL " << a->first << " " << a->second << std::endl; */
        SCState[a->first] = PENGUIN_STATE_UNKNOWN;
        /* global_mmg_alloc_ac_vector.push_back( */
        /*         std::pair<void*, unsigned long long>(a->first, a->second)); */
    }
    // rearrange sorted reuse by GL
    /* std::cout << "rearrange sorted reuse by GL\n"; */
    for (auto i = global_mmg_invid_alloc_list.begin(); i != global_mmg_invid_alloc_list.end(); i++) {
        auto srav = global_invid_sorted_reuse_alloc_map[i->first];
        std::vector<std::pair<void*, unsigned>> new_sorted_reuse_map;
        std::vector<std::pair<void*, unsigned long long>> temp;
                temp.clear();
        unsigned current = 0;
        srav.push_back(std::pair<void*, unsigned>(0, 1002)); // sentinal
        for(auto ra = srav.begin(); ra != srav.end(); ra++) {
            /* std::cout << ra->first << " ( " << ra->second << " ) " << std::endl; */
            if(ra->second != current) {
                // batch is over, sort by GL 
                std::sort(temp.begin(), temp.end(), sortfunc);
                /* std::cout << "sorted alloc with same reuse\n"; */
                for(auto s = temp.begin(); s != temp.end(); s++) {
                    /* std::cout << s->first << " " << s->second << std::endl; */
                    new_sorted_reuse_map.push_back(
                            std::pair<void*, unsigned>(s->first, current));
                    global_invid_sorted_reuse_alloc_map[i->first] = 
                        new_sorted_reuse_map;
                }
                current = ra->second;
                temp.clear();
            } 
            auto gl = global_mmg_alloc_ac_map[ra->first];
            temp.push_back(std::pair<void*, unsigned long long>(ra->first, gl));
        }
    }
    /* std::cout << "post processed sorted reuse\n"; */
    for (auto i = global_mmg_invid_alloc_list.begin(); i != global_mmg_invid_alloc_list.end(); i++) {
        /* std::cout << "invid = " << i->first << std::endl; */
        for(auto ra = global_invid_sorted_reuse_alloc_map[i->first].begin();
                ra != global_invid_sorted_reuse_alloc_map[i->first].end(); ra++) {
            /* std::cout << ra->first << " ( " << ra->second << " ) " ; */
        }
        /* std::cout << std::endl; */
    }
}

// Take the memory size (can also get from this file, or by querying APIs),
// and the invocation ID;
// Then for each allocation used in the invocation, takes appropriate action
extern "C"
void perform_memory_management(unsigned long long memsize, unsigned invid) {
    /* std::cout << "perform mem mgmt\n"; */
    bool has_pchase = false;
    // iterate over all allocation
    /* std::cout << "invid = " << invid << std::endl; */
    if(global_mmg_invid_alloc_list.find(invid) != global_mmg_invid_alloc_list.end()) {
        std::set<void*> alloc_list = global_mmg_invid_alloc_list[invid];
        for(auto i = alloc_list.begin(); i != alloc_list.end(); i++) {
            /* std::cout << *i << " "; */
            /* std::cout << global_alloc_inv_resinv_map[*i][invid] << " "; */
            /* std::cout << global_mmg_alloc_ac_map[*i] << " "; */
            /* std::cout << std::endl; */
        }
    } else {
        /* std::cout << "PANIK!"; */
    }
    if(max_invid > 1) {
        auto sorted_reuse_vector = global_invid_sorted_reuse_alloc_map[invid];
        /* std::cout << "sorted reuse\n"; */
        available = gpu_memory;  // logical availability
        for (auto al = sorted_reuse_vector.begin();
                al != sorted_reuse_vector.end(); al++) {
            /* std::cout << al->first << " "; */
            unsigned long long dsize = allocation_size_map[al->first];
            /* std::cout << "\navail = " << available << std::endl; */
            /* std::cout << "SCavail = " << SCAvail << std::endl; */
            /* std::cout << "dize = " << dsize << std::endl; */
            if(available > dsize) {
                /* std::cout << "pin on GPU\n"; */
                if(SCState[al->first] == PENGUIN_STATE_GPU_PINNED) {
                    /* std::cout << "already on gpu\n"; */
                    available -= dsize;
                } else {
                    SCState[al->first] = PENGUIN_STATE_GPU_PINNED;
                    // migrate and account in pinned memory
                    if(SCAvail > dsize) {
                        /* std::cout << "migrate to GPU\n"; */
                        SCAvail -= dsize;
                        penguinSetPrioritizedLocation((char*) al->first, dsize, 0);
                        cudaMemPrefetchAsync((char*)al->first, dsize, 0, 0 );
                        /* std::cout << "just like that SABy\n"; */
                        cudaMemAdvise((char*) al->first, dsize, cudaMemAdviseSetAccessedBy, 0);
                        SCGPUResidentAllocs[al->first] = dsize;
                        available -= dsize;
                    } else {
                        /* std::cout << "see if Evict\n"; */
                        auto my_reuse = global_alloc_inv_resinv_map[al->first][invid];
                        unsigned long long free_mem = SCAvail;
                        unsigned long long req_mem = dsize;
                        for(auto ec = SCGPUResidentAllocs.begin();
                                ec != SCGPUResidentAllocs.end(); ) {
                            auto ec_reuse = global_alloc_inv_resinv_map[ec->first][invid];
                            /* std::cout << al->first << " " << ec->first << std::endl; */
                            /* std::cout << my_reuse << " " << ec_reuse << std::endl; */
                            if(my_reuse < ec_reuse ) {
                                /* std::cout << "evict\n"; */
                                cudaMemPrefetchAsync((char*)ec->first, ec->second, -1, 0 );
                                SCState[ec->first] = PENGUIN_STATE_HOST;
                                SCAvail += ec->second;
                                free_mem += ec->second;
                                available += ec->second;
                                SCGPUResidentAllocs.erase(ec++);
                                if(free_mem >= req_mem) {
                                    break;
                                }
                            } else {
                                ec++;
                            }
                        }
                        /* std::cout << "freereq mems " << free_mem << " " << req_mem << std::endl; */
                        if(free_mem >= req_mem) {
                            /* std::cout << "Migrate nad pin\n"; */
                            /* std::cout << req_mem << std::endl; */
                            penguinSetPrioritizedLocation((char*) al->first, req_mem, 0);
                            cudaMemPrefetchAsync((char*)al->first, req_mem, 0, 0 );
                            SCGPUResidentAllocs[al->first] = req_mem;
                            SCState[al->first] = PENGUIN_STATE_GPU_PINNED;
                            SCAvail -= req_mem;
                            available -= req_mem;
                            /* std::cout << "SABy\n"; */
                            cudaMemAdvise((char*) al->first, dsize, cudaMemAdviseSetAccessedBy, 0);
                        } else {
                            if(free_mem > 0) {
                                penguinSetPrioritizedLocation((char*) al->first, free_mem, 0);
                                cudaMemPrefetchAsync((char*)al->first, req_mem, 0, 0 );
                                SCGPUResidentAllocs[al->first] = free_mem;
                                SCState[al->first] = PENGUIN_STATE_GPU_PINNED_PART;
                                SCAvail -= free_mem;
                                available -= req_mem;
                                /* std::cout << "SABy\n"; */
                                cudaMemAdvise((char*) al->first, dsize, cudaMemAdviseSetAccessedBy, 0);
                            } else {
                                SCState[al->first] = PENGUIN_STATE_HOST;
                                cudaMemAdvise((char*) al->first, dsize, cudaMemAdviseSetAccessedBy, 0);
                            }
                        }
                    }
                }
                /* available -= dsize; */
            } else if(available > 0) {
                /* std::cout << "pin (partial) on GPU\n"; */
                if(SCState[al->first] == PENGUIN_STATE_GPU_PINNED_PART) {
                    /* std::cout << "already part pinned\n"; */
                    available -= available;
                } else {
                    SCState[al->first] = PENGUIN_STATE_GPU_PINNED_PART;
                    //migrate partial and pin rest
                    if(SCAvail > available) {
                        /* std::cout << "migrate (partial) to GPU\n"; */
                        SCGPUResidentAllocs[al->first] = available;
                        penguinSetPrioritizedLocation((char*) al->first, available, 0);
                        cudaMemPrefetchAsync((char*)al->first, available, 0, 0 );
                        SCAvail -= available;
                        available -= available;
                        // pin rest on host
                        /* std::cout << "SABy rest\n"; */
                        cudaMemAdvise((char*) al->first, dsize, cudaMemAdviseSetAccessedBy, 0);
                    } else {
                        /* std::cout << "see if Evict\n"; */
                        auto my_reuse = global_alloc_inv_resinv_map[al->first][invid];
                        unsigned long long free_mem = SCAvail;
                        unsigned long long req_mem = available;
                        for(auto ec = SCGPUResidentAllocs.begin();
                                ec != SCGPUResidentAllocs.end(); ) {
                            auto ec_reuse = global_alloc_inv_resinv_map[ec->first][invid];
                            /* std::cout << al->first << " " << ec->first << std::endl; */
                            /* std::cout << my_reuse << " " << ec_reuse << std::endl; */
                            if(my_reuse < ec_reuse ) {
                                /* std::cout << "evict\n"; */
                                cudaMemPrefetchAsync((char*)ec->first, ec->second, -1, 0 );
                                SCState[ec->first] = PENGUIN_STATE_HOST;
                                SCAvail += ec->second;
                                free_mem += ec->second;
                                available += ec->second;
                                SCGPUResidentAllocs.erase(ec++);
                                if(free_mem >= req_mem) {
                                    break;
                                }
                            } else {
                                ec++;
                            }
                        }
                        /* std::cout << "freereq mems " << free_mem << " " << req_mem << std::endl; */
                        if(free_mem >= req_mem) {
                            /* std::cout << "Migrate nad pin\n"; */
                            /* std::cout << req_mem << std::endl; */
                            penguinSetPrioritizedLocation((char*) al->first, req_mem, 0);
                            cudaMemPrefetchAsync((char*)al->first, req_mem, 0, 0 );
                            SCGPUResidentAllocs[al->first] = req_mem;
                            SCAvail -= req_mem;
                            available -= req_mem;
                            /* std::cout << "SABy\n"; */
                            cudaMemAdvise((char*) al->first, dsize, cudaMemAdviseSetAccessedBy, 0);
                        }
                        else {
                            SCState[al->first] = PENGUIN_STATE_HOST;
                            cudaMemAdvise((char*) al->first, dsize, cudaMemAdviseSetAccessedBy, 0);
                        }
                    }
                }
                /* available -= available; */
            } else {
                /* std::cout << "pin on host\n"; */
                if(SCState[al->first] == PENGUIN_STATE_HOST) {
                    /* std::cout << "already on host\n"; */
                } else {
                    SCState[al->first] = PENGUIN_STATE_HOST;
                    cudaMemAdvise((char*) al->first , dsize, cudaMemAdviseSetAccessedBy, 0);
                    // pin on host
                }
                /* available -= available; */
            }
        }
    } else { // maxinvid = 1
        available = gpu_memory;  // logical availability
        for(auto a = global_mmg_alloc_ac_map.begin();
                a != global_mmg_alloc_ac_map.end(); a++) {
            /* std::cout << "GL " << a->first << " " << a->second << std::endl; */
            SCState[a->first] = PENGUIN_STATE_UNKNOWN;
            global_mmg_alloc_ac_vector.push_back(
                    std::pair<void*, unsigned long long>(a->first, a->second));
        }
        std::sort(global_mmg_alloc_ac_vector.begin(), global_mmg_alloc_ac_vector.end(),
                sortfunc);
        for(auto al = global_mmg_alloc_ac_vector.begin();
                al != global_mmg_alloc_ac_vector.end(); al++) {
            unsigned long long dsize = allocation_size_map[al->first];
            if(available > dsize) {
                /* std::cout << "pin on GPU\n"; */
                penguinSetPrioritizedLocation((char*) al->first, dsize, 0);
                cudaMemPrefetchAsync((char*)al->first, dsize, 0, 0 );
                available -= dsize;
            } else if (available > 0) {
                /* std::cout << "partial pin\n"; */
                penguinSetPrioritizedLocation((char*) al->first, available, 0);
                cudaMemPrefetchAsync((char*)al->first, available, 0, 0 );
                available -= available;
                /* std::cout << "SABy rest\n"; */
                cudaMemAdvise((char*) al->first, dsize,
                        cudaMemAdviseSetAccessedBy, 0);
            } else {
                /* std::cout << "host pin\n"; */
                cudaMemAdvise((char*) al->first, dsize,
                        cudaMemAdviseSetAccessedBy, 0);
            }

        }
    }
    /* std::cout << "end reus\n"; */
    return;
}

extern "C"
void MemoryMgmtFirstInvocationNonIter() {
    /* std::cout <<"MemoryMgmtFirstInvocationNonIter\n"; */
    /* std::cout << "data from reuse\n"; */
    for (auto a = aid_ac_map_reuse.begin(); a != aid_ac_map_reuse.end(); a++) {
        /* std::cout << a->first << "  " << a->second << " " << aid_allocation_map_reuse[a->first] << std::endl; */
        auto alloc = aid_allocation_map_reuse[a->first];
        auto invid = aid_invocation_id_map_reuse[a->first];
        global_mmg_invid_alloc_list[invid].insert(alloc);
    }
    /* std::cout << "invid to alloc list\n"; */
    /* unsigned max_invid = 0; */
    for(auto i = global_mmg_invid_alloc_list.begin(); i != global_mmg_invid_alloc_list.end(); i++) {
        /* std::cout << i->first << " :  "; */
        if(i->first > max_invid) {
            max_invid = i->first;
        }
        for(auto a = i->second.begin(); a != i->second.end(); a++) {
            /* std::cout << *a << " "; */
        }
        /* std::cout << std::endl; */
    }
    /* std::cout << "max invid = " << max_invid << std::endl; */
    /* for each allocation, for each invocation, compute the next reuse */
    /* std::map<void*, std::map<unsigned, unsigned>> global_alloc_inv_resinv_map; */
    if(max_invid > 1) {
        for(auto alloc = allocation_size_map.begin(); alloc != allocation_size_map.end(); alloc++) {
            /* std::cout << alloc->first << std::endl; */
            for (auto c = 1; c <= max_invid; c++) {
                /* std::cout << c << std::endl; */
                unsigned nearest_reuse = 1000 ;
                for(auto i = global_mmg_invid_alloc_list.begin(); i != global_mmg_invid_alloc_list.end(); i++) {
                    if(i->second.find(alloc->first) != i->second.end()) {
                        /* std::cout << "reuse at " << i->first << std::endl; */
                        if(i->first > c && i->first < nearest_reuse) {
                            nearest_reuse = i->first;
                        }
                    }
                }
                /* std::cout << "nearest reuse is " << nearest_reuse << std::endl; */
                global_alloc_inv_resinv_map[alloc->first][c] = nearest_reuse;
            }
        }
        // for each invid, list all allocations, sorted by reuse
        /* std::cout << " each invid, list all allocations, sorted by reuse\n"; */
        for (auto i = global_mmg_invid_alloc_list.begin(); i != global_mmg_invid_alloc_list.end(); i++) {
            /* std::cout << i->first << " :  "; */
            for(auto a = i->second.begin(); a != i->second.end(); a++) {
                /* std::cout << *a << " "; */
                void* alloc = *a;
                unsigned reuse = global_alloc_inv_resinv_map[*a][i->first];
                /* std::cout << reuse << " "; */
                global_invid_sorted_reuse_alloc_map[i->first].push_back(std::pair<void*, unsigned>(alloc, reuse));
                if(global_alloc_inv_resinv_map[*a][i->first] == 1000) {
                    /* std::cout << global_alloc_firstuse_map[*a] << " "; */
                }
                /* std::cout << std::endl; */
            }
            /* global_invid_sorted_reuse_alloc_map[i->first] = sorted_reuse_vector; */
            std::sort(global_invid_sorted_reuse_alloc_map[i->first].begin(),
                    global_invid_sorted_reuse_alloc_map[i->first].end(), sortfunc_l_v_u);
            /* std::cout << "sorted by reuse, allocations\n"; */
            for(auto ra = global_invid_sorted_reuse_alloc_map[i->first].begin();
                    ra != global_invid_sorted_reuse_alloc_map[i->first].end(); ra++) {
                /* std::cout << ra->first << " ( " << ra->second << " ) " ; */
            }
            /* std::cout << std::endl; */
        }
        //compute Global Locality
        for(auto a = aid_ac_map_reuse.begin(); a != aid_ac_map_reuse.end(); a++) {
            auto alloc = aid_allocation_map_reuse[a->first];
            global_mmg_alloc_ac_map[alloc] += a->second;
        }
        /* std::cout << "global locality\n"; */
        for(auto a = global_mmg_alloc_ac_map.begin(); a != global_mmg_alloc_ac_map.end(); a++) {
            /* std::cout << "GL " << a->first << " " << a->second << std::endl; */
            SCState[a->first] = PENGUIN_STATE_UNKNOWN;
        }
        // rearrange sorted reuse by GL
        /* std::cout << "rearrange sorted reuse by GL\n"; */
        for (auto i = global_mmg_invid_alloc_list.begin(); i != global_mmg_invid_alloc_list.end(); i++) {
            auto srav = global_invid_sorted_reuse_alloc_map[i->first];
            std::vector<std::pair<void*, unsigned>> new_sorted_reuse_map;
            std::vector<std::pair<void*, unsigned long long>> temp;
            temp.clear();
            unsigned current = 0;
            srav.push_back(std::pair<void*, unsigned>(0, 1002)); // sentinal
            for(auto ra = srav.begin(); ra != srav.end(); ra++) {
                /* std::cout << ra->first << " ( " << ra->second << " ) " << std::endl; */
                if(ra->second != current) {
                    // batch is over, sort by GL 
                    std::sort(temp.begin(), temp.end(), sortfunc);
                    /* std::cout << "sorted alloc with same reuse\n"; */
                    for(auto s = temp.begin(); s != temp.end(); s++) {
                        /* std::cout << s->first << " " << s->second << std::endl; */
                        new_sorted_reuse_map.push_back(
                                std::pair<void*, unsigned>(s->first, current));
                        global_invid_sorted_reuse_alloc_map[i->first] = 
                            new_sorted_reuse_map;
                    }
                    current = ra->second;
                    temp.clear();
                } 
                auto gl = global_mmg_alloc_ac_map[ra->first];
                temp.push_back(std::pair<void*, unsigned long long>(ra->first, gl));
            }
        }
        /* std::cout << "post processed sorted reuse\n"; */
        for (auto i = global_mmg_invid_alloc_list.begin(); i != global_mmg_invid_alloc_list.end(); i++) {
            /* std::cout << "invid = " << i->first << std::endl; */
            for(auto ra = global_invid_sorted_reuse_alloc_map[i->first].begin();
                    ra != global_invid_sorted_reuse_alloc_map[i->first].end(); ra++) {
                /* std::cout << ra->first << " ( " << ra->second << " ) " ; */
            }
            /* std::cout << std::endl; */
        }
    } else {
        for(auto a = aid_ac_map_reuse.begin(); a != aid_ac_map_reuse.end(); a++) {
            auto alloc = aid_allocation_map_reuse[a->first];
            global_mmg_alloc_ac_map[alloc] += a->second;
        }
        /* std::cout << "global locality\n"; */
        for(auto a = global_mmg_alloc_ac_map.begin(); a != global_mmg_alloc_ac_map.end(); a++) {
            /* std::cout << "GL " << a->first << " " << a->second << std::endl; */
            SCState[a->first] = PENGUIN_STATE_UNKNOWN;
        }
    }
    /* std::cout << "end MemoryMgmtFirstInvocationNonIter\n"; */
}

/* std::pair<double, double> compute_intersection(double m1, double c1, double m2, double c2) { */
/*     double x = (c2 - c1) / (m1 - m2); */
/*     double y = m1 * x + c1; */
/*     return std::make_pair<x,y>; */
/* } */

/* // ax + by < c */
/* // x_min, y_min are zero */
/* extern "C" */
/* void compute_area_polyhedron (int x_max, int y_max, int a, int b, int c) */ 
/* { */

/* } */

extern "C"
unsigned larger_of_two (unsigned a, unsigned b) {
    if (a > b) {
        return a;
    } else {
        return b;
    }
}

extern "C"
unsigned smaller_of_two (unsigned a, unsigned b) {
    if (a < b) {
        return a;
    } else {
        return b;
    }
}

#endif // PENGUIN
