# Installing CUDA Toolkit 525.60.13

If you want to run SUV on a system with an existing CUDA installation, we
recommend downloading libraries in the home directory.

## CUDA Toolkit and Runtime Components

- Driver - in memory, not in disk.
- CUDA Runtime - `libcudart.so`
- NVIDIA Management Library - `libnvidia-ml.so`
- Firmware - `/lib/firmware/nvidia/525.60.13`

## Download the CUDA Toolkit

- Download the correct version (525.60.13) from
[https://developer.nvidia.com/cuda-12-0-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=runfile_local]
(https://developer.nvidia.com/cuda-12-0-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=runfile_local)

```sh
wget https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda_12.0.0_525.60.13_linux.run
```

To inspect the run script, use `less(1)`. It does not load the entire file in memory.  
You can also use the `--help` message.

## Local Installation

Do **NOT** run any of this as root. Instead, use environment variables
to use the correct libraries/headers.

### The CUDA Installer

There are two installers (run-files). Both are self-extracting archives.
- `cuda-12.0.*.run` contains the toolkit (userspace components, documentation,
  etc) and the kernel installer, `NVIDIA-Linux-x86_64-575.51.03.run`.
- The kernel installer (`NVIDIA-Linux-x86_64-575.51.03.run`) is what you get
  when you download from the NVIDIA Drivers page. It contains the kernel source
  (open and proprietary flavours) and some other libraries: `libnvidia-ml.so`,
  `libcuda.so`, which unfortulately resides in `/usr/lib/x86_64-linux-gnu/`.
- More importantly, the second runfile contains `firmware`.

### The Problem

By default, some libraries are written to `/usr/local/cuda/lib64`, and some
others to `/usr/lib/x86_64-linux-gnu/`.  
The latter is shared by all CUDA versions, and will cause trouble when one tries
to run a cherry-picked CUDA version. Compilation uses the desired `nvcc` version
and succeeds, but program execution fails.   

**Identifying Problems**: Run the executable with `strace`, and search for the
libraries listed above. See which version/copy of each library is used:

```c
openat(AT_FDCWD, "/usr/lib/x86_64-linux-gnu/libcuda.so.1", O_RDONLY|O_CLOEXEC) = 3
```

**The Solution**: For the non-default CUDA version, copy these libraries to
`/installation_path/lib64/`. Export `LD_LIBRARY_PATH`.

**Which Libraries:** There are two sets of libraries: one in the `cuda_*.run`
runfile, and the second in `NVIDIA-Linux-x86_64-*.run`. Extract both and copy
the _second_ set of libraries.  
The first set is handled by the installer, given the right CLI flags.  

----

## Steps

Install the runfile.

```sh
mkdir cuda-525
install_path=`realpath cuda-525` # backticks, not single quotes.

# can take tens of minutes
sh cuda_12*run                          \
    --no-man-page                       \
    --kernel-output-path=$install_path  \
    --toolkit                           \
    --installpath=$install_path         \
    -m=kernel-open                      \
    --silent
```

Then, extract the contents of the _second_ `run` file, inside the first,
to get `libnvidia-ml.so`.

```sh
mkdir cuda-extract
extract_path=`realpath cuda-extract`
sh cuda_12*run                          \
    --extract=$extract_path
cd $extract_path
sh NVIDIA-Linux-x86_64-525.60.13.run --extract-only

# check!
ls NVIDIA-Linux-x86_64-525.60.13/libnvidia-ml*
cp NVIDIA-Linux-x86_64-525.60.13/libnvidia-ml* $install_path/lib64 -P
```

The linker expects specific suffixes. Make a lot of symlinks to be safe, for
`nvml`, `libcudart`, and `libstdc++`.

```sh
cd $install_path/lib64
ln -s libnvidia-ml.so.525.60.13 libnvidia-ml.so
ln -s libnvidia-ml.so.525.60.13 libnvidia-ml.so.1
ln -s libnvidia-ml.so.525.60.13 libnvidia-ml.so.12
ln -s libnvidia-ml.so.525.60.13 libnvidia-ml.so.12.0

# The C++ library isn't supposed to be here, but this is sometimes needed.
# The linker isn't very good with -lstdc++.
ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6
ln -s libstdc++.so.6 libstdc++.so
ln -s libstdc++.so.6 libstdc++.so.1
```

Now, `$install_path/lib64` should have the required libraries.  

### Environment Variables
Finally, add the new installation to `startup.sh`. (Replace `install_path`
with the actual path.) To be safe, add export all variables you can think of.  

```sh
export LD_LIBRARY_PATH=$install_path/lib64:$LD_LIBRARY_PATH
export CPATH="$install_path/include:/usr/include/c++/11/:/usr/include/x86_64-linux-gnu/c++/11:$CPATH"
export PATH="$install_path/bin:$PATH"

# PyTorch, CMake, UVMBench
export CUDA_PATH=$install_path
export CUDA_INSTALL_DIR=$install_path
export CUDA_HOME=$install_path
export CUDA_DIR=$install_path
```

Watch out for human and artificial stupidity (and hardcoding): for instance, this `Makefile` line:
```
CUDA_DIR = /usr/local/cuda-10.2/
```
should be replaced by
```
CUDA_DIR ?= /usr/local/cuda-10.2/
```
or better yet, omitted altogether. This will break an idle system in six months of updates.

## Copy Firmware

```sh
cd $runfile_path

mkdir /lib/firmware/nvidia/525.60.13
sudo cp NVIDIA-Linux-x86_64-525.60.13/firmware/* /lib/firmware/nvidia/525.60.13
```

The driver can be configured to _not_ use the firmware files.
This may have a performance penalty.
```sh
# Details in $runfile_path/NVIDIA-Linux-x86_64-525.60.13/html/gsp.html

sudo insmod nvidia.ko                       \
        NVreg_EnableGpuFirmware=0           \
        NVreg_OpenRmEnableUnsupportedGpus=1
```

## Loading the Driver

Manually load and unload the driver using `insmod`, `rmmod`, and `modprobe`.  
This does _not_ modify the global or local CUDA installations - the driver runs
entirely in memory.

Doing a `sudo make modules_install`, as the instructions in the driver's `README.md`
suggests, might change the installed driver version, and break the system.

To check the currently loaded driver version:
```sh
cat /proc/driver/nvidia/version
```

## Bugs/Errors

Please email me at `pranjal.singh4910@gmail.com`.  
Include these:
- cleaned-up `dmesg` output. (Run `sudo dmesg -C`, re-`insmod` the kernel, run
  the binary, repeat.)
- The `strace` log of the userspace binary executable which failed to run.
  (`strace -o tmp.strace ./a.out`)
