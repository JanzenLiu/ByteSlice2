#include    <iostream>
#include    <cstdlib>
#include    "util/intel-pcm/cpucounters.h"


int main(int argc, char* argv[]){
    constexpr size_t sz = 100*1024*1024;    // 100M
    int foo[sz];
    
    PCM* pcm = PCM::getInstance();
    PCM::ErrorCode status = pcm->program();
    
    switch(status){
        case PCM::Success:
            std::cerr << "PCM Init successful." << std::endl;
            break;
        case PCM::MSRAccessDenied:
            std::cerr << "Access to Intel PCM denied." << std::endl;
            std::cerr << "You need to run this program with admin privilege." << std::endl;
            exit(2);
            break;
        case PCM::PMUBusy:
            std::cerr << "PMU is occupied by other application." << std::endl;
            std::cerr << "Force reseting PMU configuration ...";
            pcm->resetPMU();
            pcm->cleanup();
            std::cerr << "... Done" << std::endl;
            std::cerr << "Please rerun the program." << std::endl;
            exit(2);
            break;
        default:
            std::cerr << "Unknown PCM error. Exit." << std::endl;
            exit(2);
            break;
    }
    
    SystemCounterState before, after;
    std::cout << "Begin ..." << std::endl;
    before = getSystemCounterState();
    
    for(size_t i=0; i < sz; i++){
        int j = i % sz;
        int tmp;
        tmp = foo[i]; foo[i] = foo[j]; foo[j] = tmp;
    }
    
    std::cout << "Done." << std::endl;
    after = getSystemCounterState();
    
    std::cout << "IPC, RefCycle, L2Miss, L2Hit, L2HR, Mem(R)" << std::endl;
    std::cout << getIPC(before, after) << ", "
              << getRefCycles(before, after) << ", "
              << getL2CacheMisses(before, after) << ", "
              << getL2CacheHits(before, after) << ", "
              << getL2CacheHitRatio(before, after) << ", "
              << getBytesReadFromMC(before, after)
              << std::endl;
    
    std::cout << "Trying to clean up ... " << std::endl;
    pcm->cleanup();
    std::cout << "Cleanup done." << std::endl;
}