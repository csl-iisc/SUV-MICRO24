/* Library for calling the new ioctl for prioritizing pages on the GPU */
/* SUV */

#ifndef PENGUIN
#define PENGUIN // the penguin library

#define PENGUIN_PRIORITIZED_GPU_IOCTL_NUM 75
#define PENGUIN_NO_MIGRATE_IOCTL_NUM 76
#define PENGUIN_START_STAT_COLLECTION_IOCTL_NUM 77
#define PENGUIN_STOP_STAT_COLLECTION_IOCTL_NUM 78
#define PENGUIN_QUICK_MIGRATE_IOCTL_NUM 79
#define PENGUIN_ACCESS_COUNTER_ENABLE 80
#define PENGUIN_IS_ALLOCATED 81

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

unsigned long long MBs = 2259ULL;
unsigned long long gpu_memory = 1 *  MBs * 1024ULL * 1024ULL;
unsigned long long available = gpu_memory;
unsigned long long pinned_memory = 0;
// add code to evict anything whose use is over
// add code to prioritize higher AD temporal region over lower AD temporal region

enum State {
    PENGUIN_STATE_UNKNOWN,
    PENGUIN_STATE_GPU,
    PENGUIN_STATE_GPU_PINNED,
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
    /* size_t length; */
    bool is_allocated;
    int status;
} penguin_pin_host_params;

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
        printf("%u", prop.uuid.bytes[i]);
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
penguin_error_t penguinPinHost(void *base, size_t length) {

    DIR *d;
    struct dirent *dir;
    char psf_path[512];
    char *psf_realpath;
    penguin_pin_host_params request;
    int status;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    request.base = base;
    /* request.length = length; */

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
    if ((status = ioctl(nvidia_uvm_fd, PENGUIN_IS_ALLOCATED, &request)) != 0)
    {
        fprintf(stderr, "error: %d\n", status);
        return PENGUIN_ERR_IOCTL;
    }
    if(request.is_allocated) {
    std::cout << "is alloocated\n";
    } else {
    std::cout << "not alloocated\n";
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


/* extern "C" */
/* void sort_wss() { */
/*   std::copy(allocation_wss_map.begin(), allocation_wss_map.end(),back_inserter<std::vector<std::pair<void*, unsigned long long> > >(wss_vector)); */
/*   std::sort(wss_vector.begin(), wss_vector.end(), sortfunc); */
  /* std::cout << "sorted wss\n"; */
/*   for(auto wss = wss_vector.begin(); wss != wss_vector.end(); wss++) { */
/*     std::cout << wss->first << " " << wss->second << "\n"; */
/*   } */
/* } */

#define PG_SIZE (2*1024ULL*1024ULL)

unsigned long long roundup(unsigned long long value) {
    return (value + PG_SIZE -1) & ~(PG_SIZE-1);
}
    
extern "C"
unsigned long long estimate_working_set2(unsigned long long wss_per_tb, unsigned bdimx, unsigned bdimy) {
    /* std::cout << "wss 2 = " <<  wss_per_tb << std::endl; */
    wss_per_tb = roundup(wss_per_tb * 4); // 4 bytes per element in all our workloads
    
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
    /* std::cout << "numTBs = " << numTBs << "\n"; */
    unsigned long long concTBx = (gdimx > numTBs) ? numTBs: gdimx;
    unsigned long long concTBy = (gdimx > numTBs) ? 1 : (numTBs/gdimx);
    if((pd_bidx * concTBx) > max) {
        max = pd_bidx * concTBx;
    }
    if((pd_bidy * concTBy) > max) {
        max = pd_bidy * concTBy;
    }
    /* std::cout << "params = " << pd_phi << " " << pd_bidx << " " << loopiters << " " << gdimx << " " << concTBx << " " << "\n"; */
    /* std::cout << "working set = " << max << "\n"; */
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
    std::map<void*, unsigned long long> mmg_alloc_ac_map_iteronly;
    std::map<void*, unsigned long long> mmg_alloc_span_map_iteronly;
    std::map<void*, unsigned long long> mmg_alloc_ad_map_iteronly;
    std::map<void*, unsigned long long> mmg_alloc_wss_map;
    std::vector<std::pair<void*, unsigned long long>> mmg_alloc_ad_vector_iteronly;

    std::map<unsigned, std::map<void*, unsigned long long>> mmg_alloc_ac_map_invid;
    std::map<unsigned, std::map<void*, float>> mmg_alloc_ad_map_invid;
    /* std::cout << "all aid along with ac, allocation, size, invid\n"; */
    for (auto a = aid_allocation_map.begin(); a != aid_allocation_map.end(); a++) {
        /* std::cout << a->first << " " << aid_ac_map[a->first] << " " << a->second << " "; */
        // find the allocation
        if(allocation_size_map[aid_allocation_map[a->first]] == 0) {
            /* std::cout << "[inside] "; */
            void * insideallocation = identify_memory_allocation(aid_allocation_map[a->first]);
            /* std::cout << insideallocation << "\n"; */
            allocation_size_map[aid_allocation_map[a->first]] = allocation_size_map[insideallocation];
        }
        /* std::cout << allocation_size_map[aid_allocation_map[a->first]] << " "; */
        /* std::cout << aid_invocation_id_map[a->first] << " "; */
        if(aid_wss_map_iterdep.find(a->first) != aid_wss_map_iterdep.end()) {
            /* std::cout << " iterdep "; */
            /* std::cout << aid_wss_map_iterdep[a->first]; */
        }
        if(aid_pchase_map.find(a->first) != aid_pchase_map.end()) {
            /* std::cout << " pchase "; */
        }
        /* std::cout << "\n"; */
    }
    /* std::cout <<"computing access density\n"; */
    for (auto a = aid_allocation_map.begin(); a != aid_allocation_map.end(); a++) {
        auto invid = aid_invocation_id_map[a->first];
        if(aid_wss_map_iterdep.find(a->first) != aid_wss_map_iterdep.end()) {
            // check if only a fractiof of the data structure is being accesses in this access
            auto span = aid_wss_map_iterdep[a->first];
            auto allocation = aid_allocation_map[a->first];
            auto dsize = allocation_size_map[allocation];
            /* std::cout << "hi " << span << "  " << dsize << "\n"; */
            float span_to_size = (float) span / (float) dsize;
            if(span_to_size < 0.05 && span != 0) {
                /* std::cout << "span is smallr than dsize significantly\n"; */
                mmg_alloc_ac_map_iteronly[a->second] += aid_ac_map[a->first];
                if(mmg_alloc_span_map_iteronly[a->second] < span) {
                    mmg_alloc_span_map_iteronly[a->second] = span;
                }
            } else {
                mmg_alloc_ac_map_invid[invid][a->second] += aid_ac_map[a->first];
            }

        } else {
            /* std::cout << aid_ac_map[a->first]; */
            mmg_alloc_ac_map_invid[invid][a->second] += aid_ac_map[a->first];
            // not an iteration dependent access
        }
    }
    /* std::cout << "allocation to ac map iteronly\n"; */
    for(auto a = mmg_alloc_ac_map_iteronly.begin(); a != mmg_alloc_ac_map_iteronly.end(); a++) {
        auto span = mmg_alloc_span_map_iteronly[a->first];
        /* std::cout << a->first << " " << a->second << " " << span << " "; */
        auto ad = (float) a->second / (float) span;
        /* std::cout << ad  << "\n"; */
        mmg_alloc_ad_map_iteronly[a->first] = ad;
    }
    /* std::cout << "allocation to ac to ad map\n"; */
    float max_ad_among_noniter = 0;
    for (auto mmg_alloc_ac_map_iter = mmg_alloc_ac_map_invid.begin();
            mmg_alloc_ac_map_iter != mmg_alloc_ac_map_invid.end();
            mmg_alloc_ac_map_iter++) {
        auto invid  = mmg_alloc_ac_map_iter->first;
        /* std::cout << "invid = " << mmg_alloc_ac_map_iter->first << "\n"; */
        auto mmg_alloc_ac_map = mmg_alloc_ac_map_iter->second;
        for(auto a = mmg_alloc_ac_map.begin(); a != mmg_alloc_ac_map.end(); a++) {
            /* std::cout << a->first << " " << a->second << " "; */
            auto dsize = allocation_size_map[a->first];
            auto ad = (float) a->second / (float) dsize;
            if(ad > max_ad_among_noniter) {
                max_ad_among_noniter = ad;
            }
            /* std::cout << ad  << "\n"; */
            mmg_alloc_ad_map_invid[invid][a->first] = mmg_alloc_ac_map_invid[invid][a->first] / (float) dsize;
        }
    }
    /* std::cout << "working set size map\n"; */
    for (auto a = aid_wss_map.begin(); a != aid_wss_map.end(); a++) {
        /* std::cout << a->first << " " << a->second << "\n"; */
        if(mmg_alloc_wss_map[aid_allocation_map[a->first]] < a->second) {
            mmg_alloc_wss_map[aid_allocation_map[a->first]] = a->second;
        }
    }
    /* std::cout << "max ad among non iter = " << max_ad_among_noniter << "\n"; */
    /* std::cout << "phase 2, decisions for iteronly\n"; */
    std::copy(mmg_alloc_ad_map_iteronly.begin(), 
            mmg_alloc_ad_map_iteronly.end(),
            back_inserter<std::vector<std::pair<void*, unsigned long long> > >(mmg_alloc_ad_vector_iteronly));
    std::sort(mmg_alloc_ad_vector_iteronly.begin(),
            mmg_alloc_ad_vector_iteronly.end(), sortfunc);
    /* std::cout << "sorted ad iteronly \n"; */
    for(auto a = mmg_alloc_ad_vector_iteronly.begin();
            a != mmg_alloc_ad_vector_iteronly.end(); a++) {
        /* std::cout << a->first << "  " << a->second << "\n"; */
        if(a->second > max_ad_among_noniter) {
            /* std::cout << "will be considered; "; */
            auto span = mmg_alloc_span_map_iteronly[a->first];
            if(span == 0) {
                /* std::cout << "skipped\n"; */
                continue;
            }
            /* auto span = 1024*1024; */
            auto dsize = allocation_size_map[a->first];
            /* std::cout << "span = " << span << "  "; */
            auto prefetch_size = span;
            if(prefetch_size < PENGUIN_MIN_PREFETCH) {
                prefetch_size = PENGUIN_MIN_PREFETCH;
            }
            if(prefetch_size >= dsize) {
                prefetch_size = dsize;
            }
            /* std::cout << "prefetch = " << prefetch_size << "\n"; */
            /* auto prefetch_iters_per_batch = dsize/prefetch_size; */
            auto prefetch_iters_per_batch = prefetch_size/span;
            /* std::cout << "prefetch iters per batch= " << prefetch_iters_per_batch << "\n"; */
            // insert into data structures for penguinSuperPrefetch to read from 
            AllocationToPrefetchBoolMap[a->first] = true;
            AllocationToPrefetchSizeMap[a->first] = prefetch_size * 4;
            AllocationToPrefetchItersPerBatchMap[a->first] = prefetch_iters_per_batch;
            AllocationToDecisionMap[a->first] = PENGUIN_DEC_ITERATION_MIGRATION;
            available -= prefetch_size * 4;
            /* std::cout << "available = " << available << "\n"; */ 
        } else {
            /* std::cout << "will NOT be considered\n"; */
        }
    } // iteronly ends here
    /* std::cout << "phase 2.5, decision for non-iter\n"; */
    std::vector<std::pair<void*, float>> mmg_alloc_ad_vector_invid;
    for (auto invid = mmg_alloc_ad_map_invid.begin();
            invid != mmg_alloc_ad_map_invid.end(); invid++) {
        /* std::cout << "processing invid = " << invid->first  << "\n"; */
        auto mmg_alloc_ad_map = invid->second;

        for(auto alloc = mmg_alloc_ad_map.begin();
                alloc != mmg_alloc_ad_map.end(); alloc++) {
            /* std::cout << alloc->first << "  " << alloc->second << std::endl; */
            // add a check to insert only the relevant 
            mmg_alloc_ad_vector_invid.push_back(std::pair<void*, float>(alloc->first, alloc->second));
        }
    }
    std::sort(mmg_alloc_ad_vector_invid.begin(),
            mmg_alloc_ad_vector_invid.end(), sortfuncf);
    /* std::cout << "sorted \n"; */
    unsigned locally_available = available;
    for(auto a = mmg_alloc_ad_vector_invid.begin();
            a != mmg_alloc_ad_vector_invid.end(); a++) {
        /* std::cout << a->first << " " << a->second << "\n"; */
    }

    /* std::cout << "wss for allocation\n"; */
    for(auto a = mmg_alloc_wss_map.begin();
            a != mmg_alloc_wss_map.end(); a++) {
        /* std::cout << a->first << " " << a->second << std::endl; */
    }

    // Actual decision
    /* std::cout << "actual decision\n"; */
    if(available > 0) {
        for(auto a = mmg_alloc_ad_vector_invid.begin();
                a != mmg_alloc_ad_vector_invid.end(); a++) {
            /* if */
            if(mmg_alloc_wss_map.find(a->first) != mmg_alloc_wss_map.end()){
                auto awss = mmg_alloc_wss_map.find(a->first);
                auto dsize = allocation_size_map[a->first];
                /* std::cout << a->first << " " << awss->second << std::endl; */
                if(awss->second < dsize) {
                    /* std::cout << "temporal\n"; */
                    available -= awss->second;
                } else {
                    if(available > dsize) {
                        available -= dsize;
            /* penguinSetPrioritizedLocation((char*) a->first, dsize, 0); */
                    } else {
                        available = 0;
            /* penguinSetPrioritizedLocation((char*) a->first, available, 0); */
                    }
                }
            }
        }
    }

    // LAST STEP 
    // check if memory is available; if so increase prefetch size.
    if(available > 0) {
        unsigned numPrefetchedAllocs = 0;
        unsigned long long PrefetchTotal = 0;
        for(auto memalloc = AllocationToPrefetchBoolMap.begin();
                memalloc != AllocationToPrefetchBoolMap.end(); memalloc++) {
            numPrefetchedAllocs++;
            PrefetchTotal += AllocationToPrefetchSizeMap[memalloc->first];
        }
        /* std::cout << "numPrefetchedAllocs = " << numPrefetchedAllocs << std::endl; */
        /* std::cout << "prefetchTotal = " << PrefetchTotal << std::endl; */
        auto available_now = available;
        for(auto memalloc = AllocationToPrefetchBoolMap.begin();
                memalloc != AllocationToPrefetchBoolMap.end(); memalloc++) {
            unsigned long long newAllocation = 
                (AllocationToPrefetchSizeMap[memalloc->first] * available_now) / PrefetchTotal;
            available -= newAllocation;
            AllocationToPrefetchSizeMap[memalloc->first] = newAllocation;
            auto span = mmg_alloc_span_map_iteronly[memalloc->first];
            span = span * 4;
            AllocationToPrefetchItersPerBatchMap[memalloc->first] =
                newAllocation / span;
            /* std::cout << memalloc->first << " " << newAllocation << " " */ 
                /* << newAllocation/span << std::endl; */
        }
    }
    /* std::cout << "available = " << available << std::endl; */
}

extern "C"
void perform_memory_management_iterative() {
    /* std::cout << "mm iterative \n"; */
    is_iterative = true;
    /* std::cout << "available = " << available << std::endl; */

    /* std::cout << "data from reuse\n"; */
    for (auto a = aid_ac_map_reuse.begin(); a != aid_ac_map_reuse.end(); a++) {
        /* std::cout << a->first << "  " << a->second << std::endl; */
    }
    std::map<void*, unsigned long long> mmg_alloc_ac_map_iteronly;
    std::map<void*, unsigned long long> mmg_alloc_ad_map_iteronly;
    std::map<void*, unsigned long long> mmg_alloc_span_map_iteronly;
    std::map<unsigned, std::map<void*, float>> mmg_alloc_ac_map_invid;
    std::map<unsigned, std::map<void*, float>> mmg_alloc_ad_map_invid;
     std::map<void*, float> mmg_alloc_ad_map_global;
     std::vector<std::pair<void*, float>> mmg_alloc_ad_vector_global;
    std::vector<std::pair<void*, unsigned long long>> mmg_alloc_ad_vector_iteronly;
    for (auto a = aid_allocation_map_reuse.begin(); a != aid_allocation_map_reuse.end(); a++) {
        auto invid = aid_invocation_id_map[a->first];
        if(aid_wss_map_iterdep.find(a->first) != aid_wss_map_iterdep.end()) {
            // check if only a fractiof of the data structure is being accesses in this access
            auto span = aid_wss_map_iterdep[a->first];
            auto allocation = aid_allocation_map_reuse[a->first];
            auto dsize = allocation_size_map[allocation];
            /* std::cout << "hi " << span << "  " << dsize << "\n"; */
            float span_to_size = (float) span / (float) dsize;
            if(span_to_size < 0.05 && span != 0) {
                /* std::cout << "span is smallr than dsize significantly\n"; */
                mmg_alloc_ac_map_iteronly[a->second] += aid_ac_map[a->first];
                if(mmg_alloc_span_map_iteronly[a->second] < span) {
                    mmg_alloc_span_map_iteronly[a->second] = span;
                }
            } else {
                mmg_alloc_ac_map_invid[invid][a->second] += aid_ac_map[a->first];
            }

        } else {
            /* std::cout << aid_ac_map[a->first]; */
            mmg_alloc_ac_map_invid[invid][a->second] += aid_ac_map[a->first];
            // not an iteration dependent access
        }
    }
    /* std::cout << "allocation to ac map iteronly\n"; */
    for(auto a = mmg_alloc_ac_map_iteronly.begin(); a != mmg_alloc_ac_map_iteronly.end(); a++) {
        auto span = mmg_alloc_span_map_iteronly[a->first];
        /* std::cout << a->first << " " << a->second << " " << span << " "; */
        auto ad = (float) a->second / (float) span;
        /* std::cout << ad  << "\n"; */
        mmg_alloc_ad_map_iteronly[a->first] = ad;
    }
    /* std::cout << "allocation to ac to ad map\n"; */
    float max_ad_among_noniter = 0;
    for (auto mmg_alloc_ac_map_iter = mmg_alloc_ac_map_invid.begin();
            mmg_alloc_ac_map_iter != mmg_alloc_ac_map_invid.end();
            mmg_alloc_ac_map_iter++) {
        auto invid  = mmg_alloc_ac_map_iter->first;
        /* std::cout << "invid = " << mmg_alloc_ac_map_iter->first << "\n"; */
        auto mmg_alloc_ac_map = mmg_alloc_ac_map_iter->second;
        for(auto a = mmg_alloc_ac_map.begin(); a != mmg_alloc_ac_map.end(); a++) {
            /* std::cout << a->first << " " << a->second << " "; */
            auto dsize = allocation_size_map[a->first];
            auto ad = (float) a->second / (float) dsize;
            if(ad > max_ad_among_noniter) {
                max_ad_among_noniter = ad;
            }
            /* std::cout << ad  << "\n"; */
            mmg_alloc_ad_map_invid[invid][a->first] = mmg_alloc_ac_map_invid[invid][a->first] / (float) dsize;
            mmg_alloc_ad_map_global[a->first] += mmg_alloc_ad_map_invid[invid][a->first];
        }
    }
    /* std::cout << "max ad among non iter = " << max_ad_among_noniter << "\n"; */
    /* std::cout << "phase 2, decisions for iteronly\n"; */
    std::copy(mmg_alloc_ad_map_iteronly.begin(), 
            mmg_alloc_ad_map_iteronly.end(),
            back_inserter<std::vector<std::pair<void*, unsigned long long> > >(mmg_alloc_ad_vector_iteronly));
    std::sort(mmg_alloc_ad_vector_iteronly.begin(),
            mmg_alloc_ad_vector_iteronly.end(), sortfunc);
    /* std::cout << "sorted ad iteronly \n"; */
    for(auto a = mmg_alloc_ad_vector_iteronly.begin();
            a != mmg_alloc_ad_vector_iteronly.end(); a++) {
        /* std::cout << a->first << "  " << a->second << "\n"; */
        if(a->second > max_ad_among_noniter) {
            /* std::cout << "will be considered; "; */
            auto span = mmg_alloc_span_map_iteronly[a->first];
            if(span == 0) {
                /* std::cout << "skipped\n"; */
                continue;
            }
            /* auto span = 1024*1024; */
            auto dsize = allocation_size_map[a->first];
            /* std::cout << "span = " << span << "  "; */
            auto prefetch_size = span;
            if(prefetch_size < PENGUIN_MIN_PREFETCH) {
                prefetch_size = PENGUIN_MIN_PREFETCH;
            }
            if(prefetch_size >= dsize) {
                prefetch_size = dsize;
            }
            /* std::cout << "prefetch = " << prefetch_size << "\n"; */
            /* auto prefetch_iters_per_batch = dsize/prefetch_size; */
            auto prefetch_iters_per_batch = prefetch_size/span;
            /* std::cout << "prefetch iters per batch= " << prefetch_iters_per_batch << "\n"; */
            // insert into data structures for penguinSuperPrefetch to read from 
            AllocationToPrefetchBoolMap[a->first] = true;
            AllocationToPrefetchSizeMap[a->first] = prefetch_size * 4;
            AllocationToPrefetchItersPerBatchMap[a->first] = prefetch_iters_per_batch;
            AllocationToDecisionMap[a->first] = PENGUIN_DEC_ITERATION_MIGRATION;
            available -= prefetch_size * 4;
            /* std::cout << "available = " << available << "\n"; */ 
        } else {
            /* std::cout << "will NOT be considered\n"; */
        }
    } // iteronly ends here
      // phase 3: decisions for non iterative
    /* std::cout << "global AD\n"; */
    for (auto a = mmg_alloc_ad_map_global.begin();
            a != mmg_alloc_ad_map_global.end(); a++) {
        /* std::cout << a->first << " " << a->second << std::endl; */
        mmg_alloc_ad_vector_global.push_back(std::pair<void*, float>(a->first, a->second));
    }
    std::sort(mmg_alloc_ad_vector_global.begin(),
            mmg_alloc_ad_vector_global.end(), sortfuncf);
    /* std::cout << "sorted ad global \n"; */
    for(auto a = mmg_alloc_ad_vector_global.begin();
            a != mmg_alloc_ad_vector_global.end(); a++) {
        /* std::cout << a->first << "  " << a->second << "\n"; */
    }
    // consider high AD ones first, then move on to low AD
    // If AD < 5, consider only for pinning at the same place across iterations
    /* available = gpu_memory; */
    /* std::cout << "available " << available << std::endl; */
    for(auto a = mmg_alloc_ad_vector_global.begin();
            a != mmg_alloc_ad_vector_global.end(); a++) {
        /* std::cout << a->first << "  " << a->second << "\n"; */
        auto dsize = allocation_size_map[a->first];
        if(available) {
            if(available > dsize) {
                if(AllocationStateMap[a->first] == PENGUIN_STATE_GPU_PINNED) {
                }  else {
                    /* std::cout << "gpu pin A\n"; */
                    available -= dsize;
                    /* std::cout << available <<  std::endl; */
                    AllocationStateMap[a->first] = PENGUIN_STATE_GPU_PINNED;
                    AllocationGPUResStart[a->first] = 0;
                    AllocationGPUResStop[a->first] = dsize;
                    penguinSetPrioritizedLocation((char*) a->first, dsize, 0);
                    cudaMemPrefetchAsync((char*)a->first, dsize, 0, 0 );
                }
            } else {
                /* std::cout << "gpu pin B\n"; */
                    AllocationGPUResStart[a->first] = 0;
                    AllocationGPUResStop[a->first] = available;
                /* std::cout << available <<  std::endl; */
                AllocationStateMap[a->first] = PENGUIN_STATE_GPU_PINNED;
                penguinSetPrioritizedLocation((char*) a->first, available, 0);
                cudaMemPrefetchAsync((char*)a->first, available, 0, 0 );
                available = 0;
                /* std::cout << "cpu pin rest B\n"; */
                cudaMemAdvise((char*) a->first + available, dsize - available, cudaMemAdviseSetAccessedBy, 0);
            }
        } else {
                AllocationStateMap[a->first] = PENGUIN_STATE_HOST;
                /* std::cout << "cpu pin rest B\n"; */
                /* std::cout << available <<  std::endl; */
                cudaMemAdvise((char*) a->first + available, dsize - available, cudaMemAdviseSetAccessedBy, 0);
        }
    }

    // use remaning memory
    if(available > 0) {
    }


}

/* extern "C" */
/* void perform_memory_management_global_og() { */
/*     /1* return; // debuggy: quick test *1/ */
/*     for(auto InvId = InvocationIDs.begin(); InvId != InvocationIDs.end(); InvId++){ */
/*         InvocationIDtoDecisionBoolMap[*InvId] = false; */
/*     } */
/*     std::cout << "printing all aid along with allocation and ac\n"; */
/*     for (auto a = aid_allocation_map.begin(); a != aid_allocation_map.end(); a++) { */
/*         std::cout << a->first << " " << aid_ac_map[a->first] << " " << a->second << " "; */
/*         std::cout << aid_ac_map[a->first] << " "; */
/*         std::cout << allocation_size_map[aid_allocation_map[a->first]] << " "; */
/*         std::cout << aid_invocation_id_map[a->first] << " "; */
/*         if(aid_wss_map_iterdep.find(a->first) != aid_wss_map_iterdep.end()) { */
/*             std::cout << " iterdep "; */
/*             std::cout << aid_wss_map_iterdep[a->first]; */
/*         } */
/*         std::cout << "\n"; */
/*     } */

/*     std::map<void*, unsigned long long> mmg_alloc_ac_map_iteronly; */
/*     std::map<void*, unsigned long long> mmg_alloc_span_map_iteronly; */
/*     std::map<void*, unsigned long long> mmg_alloc_ad_map_iteronly; */
/*     std::vector<std::pair<void*, unsigned long long>> mmg_alloc_ad_vector_iteronly; */

/*     std::map<unsigned, std::map<void*, unsigned long long>> mmg_alloc_ac_map_invid; */
/*     std::map<unsigned, std::map<void*, float>> mmg_alloc_ad_map_invid; */
/*     std::map<unsigned, std::map<void*, unsigned long long>> mmg_alloc_pd_x_map_invid; */
/*     std::map<unsigned, std::map<void*, unsigned long long>> mmg_alloc_pd_y_map_invid; */
/*     std::map<unsigned, std::map<void*, unsigned long long>> mmg_alloc_pd_i_map_invid; */

/*     for (auto a = aid_allocation_map.begin(); a != aid_allocation_map.end(); a++) { */
/*         auto invid = aid_invocation_id_map[a->first]; */
/*         if(aid_wss_map_iterdep.find(a->first) != aid_wss_map_iterdep.end()) { */
/*             // check if only a fractiof of the data structure is being accesses in this access */
/*             auto span = aid_wss_map_iterdep[a->first]; */
/*             auto allocation = aid_allocation_map[a->first]; */
/*             auto dsize = allocation_size_map[allocation]; */
/*             /1* std::cout << "hi " << span << "  " << dsize << "\n"; *1/ */
/*             float span_to_size = (float) span / (float) dsize; */
/*             if(span_to_size < 0.05) { */
/*                 /1* std::cout << "span is smallr than dsize significantly\n"; *1/ */
/*                 mmg_alloc_ac_map_iteronly[a->second] += aid_ac_map[a->first]; */
/*                 if(mmg_alloc_span_map_iteronly[a->second] < span) { */
/*                     mmg_alloc_span_map_iteronly[a->second] = span; */
/*                 } */
/*             } else { */
/*                 mmg_alloc_ac_map_invid[invid][a->second] += aid_ac_map[a->first]; */
/*             } */

/*         } else { */
/*             /1* std::cout << aid_ac_map[a->first]; *1/ */
/*             mmg_alloc_ac_map_invid[invid][a->second] += aid_ac_map[a->first]; */
/*             // not an iteration dependent access */
/*         } */
/*     } */
/*     std::cout << "allocation to ac map iteronly\n"; */
/*     for(auto a = mmg_alloc_ac_map_iteronly.begin(); a != mmg_alloc_ac_map_iteronly.end(); a++) { */
/*         auto span = mmg_alloc_span_map_iteronly[a->first]; */
/*         std::cout << a->first << " " << a->second << " " << span << " "; */
/*         auto ad = (float) a->second / (float) span; */
/*         std::cout << ad  << "\n"; */
/*         mmg_alloc_ad_map_iteronly[a->first] = ad; */
/*     } */
/*     std::cout << "allocation to ac map\n"; */
/*     float max_ad_among_noniter = 0; */
/*     for (auto mmg_alloc_ac_map_iter = mmg_alloc_ac_map_invid.begin(); */
/*             mmg_alloc_ac_map_iter != mmg_alloc_ac_map_invid.end(); */
/*             mmg_alloc_ac_map_iter++) { */
/*         auto invid  = mmg_alloc_ac_map_iter->first; */
/*         std::cout << "invid = " << mmg_alloc_ac_map_iter->first << "\n"; */
/*         auto mmg_alloc_ac_map = mmg_alloc_ac_map_iter->second; */
/*         for(auto a = mmg_alloc_ac_map.begin(); a != mmg_alloc_ac_map.end(); a++) { */
/*             std::cout << a->first << " " << a->second << " "; */
/*             auto dsize = allocation_size_map[a->first]; */
/*             auto ad = (float) a->second / (float) dsize; */
/*             if(ad > max_ad_among_noniter) { */
/*                 max_ad_among_noniter = ad; */
/*             } */
/*             std::cout << ad  << "\n"; */
/*             mmg_alloc_ad_map_invid[invid][a->first] = mmg_alloc_ac_map_invid[invid][a->first] / (float) dsize; */
/*             mmg_alloc_pd_x_map_invid[invid][a->first] = allocation_pd_bidx_map[a->first]; */
/*             mmg_alloc_pd_y_map_invid[invid][a->first] = allocation_pd_bidy_map[a->first]; */
/*             mmg_alloc_pd_i_map_invid[invid][a->first] = allocation_pd_phi_map[a->first]; */
/*         } */
/*     } */
/*     std::cout << "phase 1.5, pchase"; */
/*     for(auto a = aid_pchase_map.begin(); a != aid_pchase_map.end(); a++) { */
/*         if(a->second == true) { */
/*             AllocationToPchaseMap[aid_allocation_map[a->first]] = true; */
/*         } */
/*     } */
/*     std::cout << "max ad among non iter = " << max_ad_among_noniter << "\n"; */
/*     std::cout << "phase 2, decisions for iteronly\n"; */
/*     std::copy(mmg_alloc_ad_map_iteronly.begin(), */ 
/*             mmg_alloc_ad_map_iteronly.end(), */
/*             back_inserter<std::vector<std::pair<void*, unsigned long long> > >(mmg_alloc_ad_vector_iteronly)); */
/*     std::sort(mmg_alloc_ad_vector_iteronly.begin(), */
/*             mmg_alloc_ad_vector_iteronly.end(), sortfunc); */
/*     std::cout << "sorted ad iteronly \n"; */
/*     for(auto a = mmg_alloc_ad_vector_iteronly.begin(); */
/*             a != mmg_alloc_ad_vector_iteronly.end(); a++) { */
/*         std::cout << a->first << "  " << a->second << "\n"; */
/*         if(a->second > max_ad_among_noniter) { */
/*             std::cout << "will be considered; "; */
/*             auto span = mmg_alloc_span_map_iteronly[a->first]; */
/*             /1* auto span = 1024*1024; *1/ */
/*             auto dsize = allocation_size_map[a->first]; */
/*             std::cout << "span = " << span << "  "; */
/*             auto prefetch_size = span; */
/*             if(prefetch_size < PENGUIN_MIN_PREFETCH) { */
/*                 prefetch_size = PENGUIN_MIN_PREFETCH; */
/*             } */
/*             if(prefetch_size >= dsize) { */
/*                 prefetch_size = dsize; */
/*             } */
/*             std::cout << "prefetch = " << prefetch_size << "\n"; */
/*             /1* auto prefetch_iters_per_batch = dsize/prefetch_size; *1/ */
/*             auto prefetch_iters_per_batch = prefetch_size/span; */
/*             std::cout << "prefetch iters per batch= " << prefetch_iters_per_batch << "\n"; */
/*             // insert into data structures for penguinSuperPrefetch to read from */ 
/*             AllocationToPrefetchBoolMap[a->first] = true; */
/*             AllocationToPrefetchSizeMap[a->first] = prefetch_size * 4; */
/*             AllocationToPrefetchItersPerBatchMap[a->first] = prefetch_iters_per_batch; */
/*             AllocationToDecisionMap[a->first] = PENGUIN_DEC_ITERATION_MIGRATION; */
/*             available -= prefetch_size * 4; */
/*             std::cout << "available = " << available << "\n"; */ 
/*         } else { */
/*             std::cout << "will NOT be considered\n"; */
/*         } */
/*     } // iteronly ends here */
/*     std::cout << "phase 3, decisions on per invocation basis, excluding iteronly\n"; */
/*     // first phcase */
/*     // */
/*     unsigned long long total_size = 0; */
/*     for (auto a = allocation_size_map.begin(); a != allocation_size_map.end(); a++) { */
/*         total_size += a->second; */
/*     } */
/*     for (auto a = AllocationToPchaseMap.begin(); a != AllocationToPchaseMap.end(); a++) { */
/*         /1* auto invid = aid_invocation_id_map[a->first]; *1/ */
/*         for(auto invid = InvocationIDs.begin(); invid != InvocationIDs.end(); invid++){ */
/*             InvocationIDtoAllocationToDecisionMap[*invid][a->first] = PENGUIN_DEC_ACCESS_COUNTER; */
/*             unsigned long long psize = allocation_size_map[a->first]; */
/*             psize = (gpu_memory * psize) / total_size ; */
/*             available -= psize; */
/*         } */
/*     } */
/*     for (auto invid = mmg_alloc_ad_map_invid.begin(); */
/*             invid != mmg_alloc_ad_map_invid.end(); invid++) { */
/*         std::cout << "processing invid = " << invid->first  << "\n"; */
/*         auto mmg_alloc_ad_map = invid->second; */
/*         std::vector<std::pair<void*, float>> mmg_alloc_ad_vector_invid; */
/*         for(auto alloc = mmg_alloc_ad_map.begin(); */
/*                 alloc != mmg_alloc_ad_map.end(); alloc++) { */
/*             auto pd_x = mmg_alloc_pd_x_map_invid[invid->first][alloc->first]; */
/*             auto pd_y = mmg_alloc_pd_y_map_invid[invid->first][alloc->first]; */
/*             auto pd_i = mmg_alloc_pd_i_map_invid[invid->first][alloc->first]; */
/*             std::cout << alloc->first << " " << alloc->second << " " << pd_x << " " << pd_y << " " << pd_i << "\n"; */
/*             // be default, host pin */
/*             InvocationIDtoAllocationToDecisionMap[invid->first][alloc->first] = PENGUIN_DEC_NONE; */
/*             /1* if(alloc->second < 0.25) { *1/ */
/*             if(alloc->second <= 1) { */
/*                 std::cout << "AD too low: ignoring or host pinning (default)\n"; */
/*                 InvocationIDtoAllocationToDecisionMap[invid->first][alloc->first] = PENGUIN_DEC_HOST_PIN; */
/*             } */
/*             auto dsize = allocation_size_map[alloc->first]; */
/*             if((alloc->second > 5) && (allocation_wss_map[alloc->first] < 0.5 * (dsize))) { */
/*                 std::cout << "migrate on demand " << allocation_wss_map[alloc->first]  << " " << (0.5 * dsize) << "\n"; */
/*                 InvocationIDtoAllocationToDecisionMap[invid->first][alloc->first] = PENGUIN_DEC_MIGRATE_ON_DEMAND; */
/*                 InvocationIDtoAllocationToPartialSize[invid->first][alloc->first] = allocation_wss_map[alloc->first]; */
/*             } */
/*             if(alloc->second < 5 && (allocation_wss_map[alloc->first] < (0.5 * dsize))){ */
/*                 InvocationIDtoAllocationToDecisionMap[invid->first][alloc->first] = PENGUIN_DEC_HOST_PIN; */
/*             } */
/*             if(InvocationIDtoAllocationToDecisionMap[invid->first][alloc->first] != PENGUIN_DEC_MIGRATE_ON_DEMAND) { */
/*                 mmg_alloc_ad_vector_invid.push_back(std::pair<void*, float>(alloc->first, alloc->second)); */
/*             } */
/*             /1* } *1/ */
/*             std::sort(mmg_alloc_ad_vector_invid.begin(), */
/*                     mmg_alloc_ad_vector_invid.end(), sortfuncf); */
/*             std::cout << "sorted \n"; */
/*             unsigned locally_available = available; */
/*             for(auto a = mmg_alloc_ad_vector_invid.begin(); */
/*                     a != mmg_alloc_ad_vector_invid.end(); a++) { */
/*                 std::cout << a->first << " " << a->second << "\n"; */
/*                 if(InvocationIDtoAllocationToDecisionMap[invid->first][a->first] == PENGUIN_DEC_HOST_PIN) { */
/*                     std::cout << "skipping already host alloced\n"; */
/*                     continue; */
/*                 } */
/*                 if(locally_available) { */
/*                     auto dsize = allocation_size_map[a->first]; */
/*                     if(locally_available >= dsize) { */
/*                         InvocationIDtoAllocationToDecisionMap[invid->first][a->first] = PENGUIN_DEC_GPU_PIN; */
/*                         std::cout << "available before= " << locally_available << "\n"; */ 
/*                         locally_available -= dsize; */
/*                         std::cout << "gpu pin\n"; */
/*                         std::cout << "locally available = " << locally_available << "\n"; */ 
/*                     } else if(locally_available > 0) { */
/*                         auto amount = locally_available; */
/*                         InvocationIDtoAllocationToDecisionMap[invid->first][a->first] = PENGUIN_DEC_GPU_HOST_PARTIAL_PIN; */
/*                         InvocationIDtoAllocationToPartialSize[invid->first][a->first] = amount; */
/*                         std::cout << "available before= " << locally_available << "\n"; */ 
/*                         locally_available = locally_available - amount; */
/*                         std::cout << "gpu host partial pin\n"; */
/*                         std::cout << "locally available after = " << locally_available << "\n"; */ 
/*                     } else { */
/*                         InvocationIDtoAllocationToDecisionMap[invid->first][a->first] = PENGUIN_DEC_HOST_PIN; */
/*                         std::cout << "host pin\n"; */
/*                         std::cout << "locally available = " << locally_available << "\n"; */ 
/*                     } */
/*                 } */
/*             } */
/*         } */
/*     } */

/*     std::cout << "print decisions\n"; */
/*     for(auto alloc = allocation_size_map.begin(); alloc != allocation_size_map.end(); alloc++) { */
/*         std::cout << alloc->first << "\n"; */
/*         auto common_decision = PENGUIN_DEC_NONE; */
/*         for(auto invid = InvocationIDtoAllocationToDecisionMap.begin(); */ 
/*                 invid != InvocationIDtoAllocationToDecisionMap.end(); invid++) { */
/*             std::cout << invid->first << "  " << InvocationIDtoAllocationToDecisionMap[invid->first][alloc->first] << "\n"; */
/*         } */
/*         std::cout << "\n"; */
/*     } */

/*     // identify common decisions */
/*     // by going over each allocation, and looking at each of the decisions (nc2) */
/*     std::cout << "identify common decisions\n"; */
/*     for(auto alloc = allocation_size_map.begin(); alloc != allocation_size_map.end(); alloc++) { */
/*         std::cout << alloc->first << "\n"; */
/*         auto common_decision = PENGUIN_DEC_NONE; */
/*         for(auto invid = InvocationIDtoAllocationToDecisionMap.begin(); */ 
/*                 invid != InvocationIDtoAllocationToDecisionMap.end(); invid++) { */
/*             if(AllocationToDecisionMap[alloc->first]  == PENGUIN_DEC_ACCESS_COUNTER ) { */
/*                 AllocationToCommonDecisionMap[alloc->first] = PENGUIN_DEC_ACCESS_COUNTER; */
/*             } */
/*         } */
/*         for(auto invid = InvocationIDtoAllocationToDecisionMap.begin(); */ 
/*                 invid != InvocationIDtoAllocationToDecisionMap.end(); invid++) { */
/*             auto dec = InvocationIDtoAllocationToDecisionMap[invid->first][alloc->first]; */
/*             if(AllocationToDecisionMap[alloc->first]  == PENGUIN_DEC_ITERATION_MIGRATION ) { */
/*                 if(dec == PENGUIN_DEC_GPU_HOST_PARTIAL_PIN) { */
/*                     common_decision = PENGUIN_DEC_ITERATION_MIGRATION_PLUS_GPU_HOST_PIN; */
/*                     AllocationToPartialSizeMap[alloc->first] */
/*                         = InvocationIDtoAllocationToPartialSize[invid->first][alloc->first] ; */
/*                 } */
/*             } */
/*             if(dec == PENGUIN_DEC_HOST_PIN && common_decision == PENGUIN_DEC_NONE) { */
/*                 common_decision = PENGUIN_DEC_HOST_PIN; */
/*             } else if(dec == PENGUIN_DEC_GPU_HOST_PARTIAL_PIN && common_decision == PENGUIN_DEC_NONE) { */
/*                 common_decision = PENGUIN_DEC_GPU_HOST_PARTIAL_PIN; */
/*             } else if(dec == PENGUIN_DEC_GPU_HOST_PARTIAL_PIN && common_decision == PENGUIN_DEC_HOST_PIN) { */
/*                 common_decision = PENGUIN_DEC_GPU_HOST_PARTIAL_PIN; */
/*                 auto parsize = AllocationToPartialSizeMap[alloc->first]; */
/*                 if(parsize < InvocationIDtoAllocationToPartialSize[invid->first][alloc->first]) { */
/*                     AllocationToPartialSizeMap[alloc->first] */
/*                         = InvocationIDtoAllocationToPartialSize[invid->first][alloc->first] ; */
/*                 } */
/*             } */
/*         } */
/*         std::cout << "common decision = " << common_decision << "\n"; */
/*         for(auto invid = InvocationIDtoAllocationToDecisionMap.begin(); */ 
/*                 invid != InvocationIDtoAllocationToDecisionMap.end(); invid++) { */
/*             if(common_decision == PENGUIN_DEC_NONE) { */
/*                 if(AllocationToPrefetchBoolMap[alloc->first] == false) { */
/*                     InvocationIDtoDecisionBoolMap[invid->first] = true; */
/*                     std::cout << alloc->first << " "; */
/*                     std::cout << "need to take per allocation, per invocation decision\n"; */
/*                     AllocationStateMap[alloc->first] = PENGUIN_STATE_UNKNOWN; */
/*                 } */
/*             } */
/*             else { */
/*                 AllocationToCommonDecisionMap[alloc->first] = common_decision; */
/*             } */
/*         } */
/*     } */

/*     // if a data structure is being iteration prefetched and GPU host pinned */

/*     // perform default decision */
/*     // */
/*     std::cout << "performing allocation common decision here\n"; */
/*     // first handle pointer chase */
/*     bool pchase_enabled = false; */
/*     for(auto alloc = AllocationToCommonDecisionMap.begin(); */
/*             alloc != AllocationToCommonDecisionMap.end(); alloc++) { */
/*         if(alloc->second == PENGUIN_DEC_ACCESS_COUNTER) { */
/*             pchase_enabled = true; */
/*             penguinEnableAccessCounters(); */
/*         } */
/*     } */
/*     for(auto alloc = AllocationToCommonDecisionMap.begin(); */
/*             alloc != AllocationToCommonDecisionMap.end(); alloc++) { */
/*         std::cout << alloc->first << " " << alloc->second << "\n"; */
/*         if(alloc->second == PENGUIN_DEC_MIGRATE_ON_DEMAND) { */
/*             continue; */
/*         } */
/*         if(alloc->second == PENGUIN_DEC_GPU_PIN) { */
/*             auto dsize = allocation_size_map[alloc->first]; */
/*             std::cout << "gpu pin " << dsize << "\n"; */
/*             char* ptr = (char*) alloc->first; */
/*             /1* cudaMemPrefetchAsync(ptr, dsize, 0, 0 ); *1/ */
/*             /1* penguinSetPrioritizedLocation((char*) alloc->first, dsize, 0); *1/ */
/*             continue; */
/*         } */
/*         if(alloc->second == PENGUIN_DEC_ITERATION_MIGRATION_PLUS_GPU_HOST_PIN) { */
/*             // asssuming partial size is larger than prefetch size */
/*             // TODO: check for the above */
/*             unsigned long long partial_size = AllocationToPartialSizeMap[alloc->first]; */
/*             auto span = mmg_alloc_span_map_iteronly[alloc->first]; */
/*             unsigned long long prefetch_iters_per_batch = partial_size/span; */
/*             unsigned long long prefetch_size = AllocationToPrefetchSizeMap[alloc->first]; */
/*             std::cout << "prefetching and partial pinning together\n"; */
/*             std::cout << prefetch_size << " " << partial_size << "  " << span << " " << prefetch_iters_per_batch << "\n"; */
/*             AllocationToPrefetchBoolMap[alloc->first] = true; */
/*             AllocationToPrefetchSizeMap[alloc->first] = prefetch_size + partial_size; */
/*             AllocationToPrefetchItersPerBatchMap[alloc->first] = prefetch_iters_per_batch; */
/*             char* ptr = (char*) alloc->first; */
/*             auto dsize = allocation_size_map[alloc->first]; */
/*             cudaMemAdvise(ptr, dsize, cudaMemAdviseSetAccessedBy, 0); */
/*         } */
/*         if(alloc->second == PENGUIN_DEC_GPU_HOST_PARTIAL_PIN) { */
/*             auto dsize = allocation_size_map[alloc->first]; */
/*             auto partial_size = AllocationToPartialSizeMap[alloc->first]; */
/*             std::cout << "partial pin " << partial_size << "\n"; */
/*             if(AllocationToPrefetchBoolMap[alloc->first] == true) { */
/*                 std::cout << "prefetching and partial pinning together\n"; */
/*                 // asssuming partial size is larger than prefetch size */
/*                 // TODO: check for the above */
/*                 AllocationToPrefetchSizeMap[alloc->first] = partial_size; */
/*                 auto span = mmg_alloc_span_map_iteronly[alloc->first]; */
/*                 auto prefetch_iters_per_batch = partial_size/span; */
/*                 AllocationToPrefetchItersPerBatchMap[alloc->first] = prefetch_iters_per_batch; */
/*             } else { */
/*                 char* ptr = (char*) alloc->first; */
/*                 cudaMemPrefetchAsync(ptr, partial_size, 0, 0 ); */
/*                 penguinSetPrioritizedLocation((char*) alloc->first, partial_size, 0); */
/*                 cudaMemAdvise(ptr + partial_size, dsize - partial_size, cudaMemAdviseSetAccessedBy, 0); */
/*             } */
/*             continue; */
/*         } */
/*         if(alloc->second == PENGUIN_DEC_HOST_PIN) { */
/*             auto dsize = allocation_size_map[alloc->first]; */
/*             std::cout << "host pin " << dsize << "\n"; */
/*             char* ptr = (char*) alloc->first; */
/*             cudaMemAdvise(ptr , dsize, cudaMemAdviseSetAccessedBy, 0); */
/*             continue; */
/*         } */
/*     } */
/*     return; */
/* } */

// Take the memory size (can also get from this file, or by querying APIs),
// and the invocation ID;
// Then for each allocation used in the invocation, takes appropriate action
extern "C"
void perform_memory_management(unsigned long long memsize, unsigned invid) {
    /* std::cout << "perform mem mgmt\n"; */
    bool has_pchase = false;
    bool has_unknown = false;
    if(!is_iterative) {
        /* std::cout << "performing local memory mgmt for invid " << invid << std::endl; */
        std::map<void*, unsigned long long> mmg_alloc_wss_map;
        std::map<void*, unsigned long long> mmg_alloc_ac_map;
        std::map<void*, bool> mmg_alloc_pchase_map;
        std::map<void*, double> mmg_alloc_ad_map;
        for (auto a = aid_allocation_map.begin(); a != aid_allocation_map.end(); a++) {
            /* std::cout << a->first << " " << aid_ac_map[a->first] << " " << a->second << " "; */
            // find the allocation
            if(allocation_size_map[aid_allocation_map[a->first]] == 0) {
                /* std::cout << "[inside] "; */
                void * insideallocation = identify_memory_allocation(aid_allocation_map[a->first]);
                /* std::cout << insideallocation << "\n"; */
                allocation_size_map[aid_allocation_map[a->first]] = allocation_size_map[insideallocation];
                // force onto the og allocation
                aid_allocation_map[a->first] = insideallocation;
            }
        }
        // compute reuse distance at the first invocation
        if(invid == 1) {
        }
        /* std::cout << "working set size map\n"; */
        for (auto a = aid_wss_map.begin(); a != aid_wss_map.end(); a++) {
            if(aid_invocation_id_map[a->first] == invid) {
                /* std::cout << a->first << " " << a->second << "\n"; */
                if(mmg_alloc_wss_map[aid_allocation_map[a->first]] < a->second) {
                    mmg_alloc_wss_map[aid_allocation_map[a->first]] = a->second;
                }
            }
        }
        for (auto a = aid_pchase_map.begin(); a != aid_pchase_map.end(); a++) {
            /* std::cout << a->first << " pchase\n"; */
            has_pchase = true;
            mmg_alloc_pchase_map[aid_allocation_map[a->first]] = true;
        }
        for (auto a = aid_ac_incomp_map.begin(); a != aid_ac_incomp_map.end(); a++) {
            has_unknown = true;
        }
        /* std::cout << "access count map\n"; */
        for (auto a = aid_ac_map.begin(); a != aid_ac_map.end(); a++) {
            if(aid_invocation_id_map[a->first] == invid) {
                /* std::cout << a->first << " " << a->second << "\n"; */
                mmg_alloc_ac_map[aid_allocation_map[a->first]] += a->second;
            }
        }
        /* std::cout << "access density map\n"; */
        std::vector<std::pair<void*, float>> mmg_alloc_ad_vector_invid;
        /* std::vector<std::pair<void*, State>> mmg_alloc_decision_vector_invid; */
        std::map<void*, State> mmg_alloc_decision_map_invid; // decision for upcoming iterations
        std::map<void*, unsigned long long> mmg_alloc_length_map_invid; // decision for upcoming iterations
        for (auto a = mmg_alloc_ac_map.begin(); a != mmg_alloc_ac_map.end(); a++) {
            mmg_alloc_ad_map[a->first] = (double) a->second / allocation_size_map[a->first];
            /* std::cout << a->first << " " << mmg_alloc_ad_map[a->first] << std::endl; */
            mmg_alloc_ad_vector_invid.push_back(std::pair<void*, float>(a->first, mmg_alloc_ad_map[a->first]));
        }
        std::sort(mmg_alloc_ad_vector_invid.begin(),
                mmg_alloc_ad_vector_invid.end(), sortfuncf);
        /* std::cout << "sorted \n"; */
        unsigned locally_available = available;
        for(auto a = mmg_alloc_ad_vector_invid.begin();
                a != mmg_alloc_ad_vector_invid.end(); a++) {
            /* std::cout << a->first << " " << a->second << "\n"; */
        }

        unsigned long long total_memory_used = 0;
        for(auto a = allocation_size_map.begin(); a != allocation_size_map.end(); a++) {
            total_memory_used += a->second;
        }

        // Actual decision
        available = gpu_memory;
        /* std::cout << available <<  std::endl; */
        unsigned long long total_available = gpu_memory;

        /* std::cout << "actual decision\n"; */
        if(available > 0) {
            if(has_pchase || has_unknown) {
                /* std::cout << "hi enable access counters\n"; */
                penguinEnableAccessCounters();
                /* return; */
            }
            for(auto a = mmg_alloc_pchase_map.begin(); a != mmg_alloc_pchase_map.end(); a++) {
                auto dsize = allocation_size_map[a->first];
                /* std::cout << std::endl << a->first << "size = " << dsize << std::endl; */
                /* std::cout << "av = " << available << std::endl; */
                if(AllocationStateMap[a->first] == PENGUIN_STATE_AC) {
                } else {
                    AllocationStateMap[a->first] = PENGUIN_STATE_AC;
                    /* std::cout << a->first << " " << available << std::endl; */
                        unsigned long long size = (dsize *total_available)/ total_memory_used;
                    if(available > 0) {
                        if(available > size) {
                            /* std::cout << "case A\n"; */
                            /* size = (dsize * 2) / 3; */
                            available -= size;
                        } else {
                            /* std::cout << "case B. ought not to come here\n"; */
                            size = available - 10 * 1024ULL*1024ULL;
                            available -= size;
                        }
                    } else {
                            /* std::cout << "case C\n"; */
                        size = 0;
                    }
                    /* std::cout << "qeeping " << size << "for pointer chase\n"; */
                    /* cudaMemPrefetchAsync((char*)a->first, size, 0, 0 ); */

                }
            }
            for(auto a = mmg_alloc_ad_vector_invid.begin();
                    a != mmg_alloc_ad_vector_invid.end(); a++) {
                /* if */
                                        void *addr = a->first;
                                        addr = round_down(addr);
                auto dsize = allocation_size_map[addr];
                /* std::cout << std::endl << addr << "size = " << dsize << std::endl; */
                /* std::cout << "av = " << available << std::endl; */
                if(mmg_alloc_pchase_map.find(a->first) != mmg_alloc_pchase_map.end()) {
                    /* std::cout << "dominated by pchase\n"; */
                    continue;
                }
                if(mmg_alloc_wss_map.find(a->first) != mmg_alloc_wss_map.end()){
                    auto awss = mmg_alloc_wss_map.find(a->first);
                    auto dsize = allocation_size_map[a->first];
                    /* std::cout << a->first << " " << awss->second << std::endl; */
                    auto ad = mmg_alloc_ad_map[a->first];
                    if(available < 2*1024*1024 && has_pchase) { // hard to place small regions
                        /* std::cout << "leave to pchase\n"; */
                        continue;
                    }
                    if(awss->second < dsize && dsize > 2* 1024*1024 && ad > 5) { //TODO:keep low ad temporal on CPU, unless these is left over memory even after reserving enough for the entire temporal
                        /* std::cout << "temporal\n"; */
                        available -= awss->second;
                        /* std::cout << available <<  std::endl; */
                        /* cudaMemPrefetchAsync((char*)a->first, dsize, 0, 0 ); */
                        mmg_alloc_decision_map_invid[a->first] = PENGUIN_STATE_GPU;
                    } else {
                        if(mmg_alloc_ad_map[a->first] > 5.0) {
                            if(available > dsize) {
                                if(AllocationStateMap[addr] == PENGUIN_STATE_GPU_PINNED) {
                                    AllocationStateMap[addr] = PENGUIN_STATE_GPU_PINNED;
                                }  else {
                                    /* std::cout << "gpu pin A\n"; */
                                    AllocationStateMap[addr] = PENGUIN_STATE_GPU_PINNED;
                                    penguinSetPrioritizedLocation((char*) a->first, dsize, 0);
                                    cudaMemPrefetchAsync((char*)a->first, dsize, 0, 0 );
                                available -= dsize;
                        /* std::cout << available <<  std::endl; */
                                    pinned_memory += dsize;
                        /* std::cout << "pinned = " << pinned_memory << std::endl; */
                                }
                            } else {
                                if(AllocationStateMap[addr] == PENGUIN_STATE_GPU_PINNED) {
                                }  else {
                                    /* std::cout << "gpu pin B\n"; */
                                    AllocationStateMap[addr] = PENGUIN_STATE_GPU_PINNED;
                                    penguinSetPrioritizedLocation((char*) a->first, available, 0);
                                    cudaMemPrefetchAsync((char*)a->first, available, 0, 0 );
                                    /* std::cout << "cpu pin rest B\n"; */
                                    cudaMemAdvise((char*) a->first + available, dsize - available, cudaMemAdviseSetAccessedBy, 0);
                                    pinned_memory += available;
                        /* std::cout << "pinned = " << pinned_memory << std::endl; */
                                    available = 0;
                        /* std::cout << available <<  std::endl; */
                                }
                            }
                        } else {
                            if(!has_pchase) {
                                if(AllocationStateMap[a->first] == PENGUIN_STATE_GPU_PINNED) {
                                }  else {
                                    if(available > dsize) {
                                        /* std::cout << "gpu pin c.1\n"; */
                                        AllocationStateMap[addr] = PENGUIN_STATE_GPU_PINNED;
                                        penguinSetPrioritizedLocation((char*) a->first, dsize, 0);
                                        cudaMemPrefetchAsync((char*)a->first, dsize, 0, 0 );
                                        available -= dsize;
                                        /* std::cout << available <<  std::endl; */
                                    pinned_memory += dsize;
                        /* std::cout << "pinned = " << pinned_memory << std::endl; */
                                    } else {
                                        /* std::cout << "gpu pin c.2\n"; */
                                        /* std::cout << "cpu pin rest c.2\n"; */
                                        AllocationStateMap[addr] = PENGUIN_STATE_GPU_PINNED;
                                        cudaMemPrefetchAsync((char*)a->first, available, 0, 0 );
                                        cudaMemAdvise((char*) a->first +available, dsize -available, cudaMemAdviseSetAccessedBy, 0);
                                    pinned_memory += available;
                        /* std::cout << "pinned = " << pinned_memory << std::endl; */
                                        available = 0;
                                        /* std::cout << available <<  std::endl; */
                                    }
                                }
                            } else {
                                /* std::cout << "cpu pin rest D\n"; */
                                cudaMemAdvise((char*) a->first , dsize, cudaMemAdviseSetAccessedBy, 0);
                                penguinSetNoMigrateRegion((char*) a->first, dsize, 0, true);
                            }
                        }
                    }
                    /* std::cout << "\navailable = " << available << std::endl; */
                }
            }
        }

        // if there is still free memory
        if(available > 0) {
        }
    }
    return;
}

extern "C"
void MemoryMgmtFirstInvocationNonIter() {
    /* std::cout <<"MemoryMgmtFirstInvocationNonIter\n"; */
    /* std::cout << "data from reuse\n"; */
    std::map<unsigned, std::set<void*>> mmg_invid_alloc_list;
    for (auto a = aid_ac_map_reuse.begin(); a != aid_ac_map_reuse.end(); a++) {
        /* std::cout << a->first << "  " << a->second << " " << aid_allocation_map_reuse[a->first] << std::endl; */
        auto alloc = aid_allocation_map_reuse[a->first];
        auto invid = aid_invocation_id_map_reuse[a->first];
        mmg_invid_alloc_list[invid].insert(alloc);
    }
    /* std::cout << "invid to alloc list\n"; */
    unsigned max_invid = 0;
    for(auto i = mmg_invid_alloc_list.begin(); i != mmg_invid_alloc_list.end(); i++) {
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
    std::map<void*, std::map<unsigned, unsigned>> alloc_inv_resinv_map;
    for(auto alloc = allocation_size_map.begin(); alloc != allocation_size_map.end(); alloc++) {
        /* std::cout << alloc->first << std::endl; */
        for (auto c = 1; c <= max_invid; c++) {
            /* std::cout << c << std::endl; */
            unsigned nearest_reuse = 1000 ;
            for(auto i = mmg_invid_alloc_list.begin(); i != mmg_invid_alloc_list.end(); i++) {
                if(i->second.find(alloc->first) != i->second.end()) {
                    /* std::cout << "reuse at " << i->first << std::endl; */
                    if(i->first > c && i->first < nearest_reuse) {
                        nearest_reuse = i->first;
                    }
                }
            }
            /* std::cout << "nearest reuse is " << nearest_reuse << std::endl; */
            alloc_inv_resinv_map[alloc->first][c] = nearest_reuse;
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
