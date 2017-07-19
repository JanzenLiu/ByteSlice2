
### Build ###

You need CMake to generate build scripts. By default, we use makefile.

To generate debug build:

    mkdir debug
    cd debug
    cmake -DCMAKE_BUILD_TYPE=debug ..
    make

To generate release build:

    mkdir release
    cd release
    cmake -DCMAKE_BUILD_TYPE=release ..
    make

### Multithreading ###
All multithreading is controlled by setting environment variable,
per session or per execution.

Per session:

    export OMP_NUM_THREADS=4    # set the max num of threads
    export GOMP_CPU_AFFINITY=0,1,2,3    # set the thread binding (to cores)

Per execution: (see next section for why you need sudo)

    sudo OMP_NUM_THREADS=4 experiments/simple   # set the env var in command line
    sudo -E experiments/simple  # Or: inherit the env var from current shell

### Running Experiments in Linux ###
All experiments use Intel PCM to profile hardware events.
And everything that uses PCM requires admin right.

IMPORTANT: You need proper system configuration before you can run the programs correctly.

Preparation:
1. Stop the NMI watchdog:

    echo 0 | sudo tee /proc/sys/kernel/nmi_watchdog 

2. Load the MSR module in kernel:

    sudo modprobe msr

Execution:
1. You need admin privilege to run all experiments as PCM needs that. E.g.,

    sudo experiments/simple

Reset Intel PCM:
In case the PCM output doesn't look right, try to reset the PMU counters (at your own risk)
by running the command line utility provide by Intel:

    sudo util/intel-pcm/pcm.x -r


### Usage Example ###

See how to use the library in the example program provided in "example/".

For example,

    cd release
    ./example/example1 -h

Multithreading is controlled by OpenMP:

    OMP_NUM_THREADS=2 ./example/example1

### File structure ###

./include
Header files.

./src
Implementation.

./tests
Test cases in GoogleTest.

./gtest-1.7.0
GoogleTest library.

./util
Utilities. Third-party libraries and wrapper classes.

./example
Minimal exapmles to demonstrate how to use the library.

