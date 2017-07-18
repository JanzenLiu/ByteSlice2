#include    "system_perf_monitor.h"
#include    <iostream>

void SystemPerfMonitor::Init(){
    pcm_ = PCM::getInstance();
    PCM::ErrorCode status;
    status = pcm_->program();

    switch(status){
        case PCM::Success:
            break;
        case PCM::MSRAccessDenied:
            std::cerr << "Access to Intel PCM denied." << std::endl;
            std::cerr << "You need to run this program with admin privilege." << std::endl;
            exit(2);
            break;
        case PCM::PMUBusy:
            std::cerr << "PMU is occupied by other application." << std::endl;
            std::cerr << "Force reseting PMU configuration ...";
            pcm_->resetPMU();
            pcm_->cleanup();
            std::cerr << "... Done" << std::endl;
            std::cerr << "Please rerun the program." << std::endl;
            exit(2);
            break;
        default:
            std::cerr << "Unknown PCM error. Exit." << std::endl;
            exit(2);
            break;
    }
}

void SystemPerfMonitor::Destroy(){
    std::cout << "Clean up PCM ... " << std::endl;
    pcm_->cleanup();
}

void SystemPerfMonitor::Start(){
    before_state_ = getSystemCounterState();
}

void SystemPerfMonitor::Stop(){
    after_state_ = getSystemCounterState();
}

uint64 SystemPerfMonitor::GetRefCycles(){
    return getRefCycles(before_state_, after_state_);
}

uint64 SystemPerfMonitor::GetCycles(){
    return getCycles(before_state_, after_state_);
}


uint64 SystemPerfMonitor::GetL2CacheMisses(){
    return getL2CacheMisses(before_state_, after_state_);
}

double SystemPerfMonitor::GetL2CacheHitRatio(){
    return getL2CacheHitRatio(before_state_, after_state_);
}

uint64 SystemPerfMonitor::GetL3CacheMisses(){
    return getL3CacheMisses(before_state_, after_state_);
}

double SystemPerfMonitor::GetL3CacheHitRatio(){
    return getL3CacheHitRatio(before_state_, after_state_);
}

double SystemPerfMonitor::GetIPC(){
    return getIPC(before_state_, after_state_);
}

uint64 SystemPerfMonitor::GetRetiredInstructions(){
    return getInstructionsRetired(before_state_, after_state_);
}

uint64 SystemPerfMonitor::GetBytesReadFromMC(){
    return getBytesReadFromMC(before_state_, after_state_);
}

uint64 SystemPerfMonitor::GetBytesWrittenToMC(){
    return getBytesWrittenToMC(before_state_, after_state_);
}

double SystemPerfMonitor::GetL2LostCycles(){
    return getCyclesLostDueL2CacheMisses(before_state_, after_state_);
}

double SystemPerfMonitor::GetL3LostCycles(){
    return getCyclesLostDueL3CacheMisses(before_state_, after_state_);
}

