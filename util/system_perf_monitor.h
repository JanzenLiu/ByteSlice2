#ifndef SYSTEM_PERF_MONITOR_H
#define SYSTEM_PERF_MONITOR_H

#include    "intel-pcm/cpucounters.h"

/**

  A system-wide performance monitor that utilizes Intel's PCM API
  pcm-lib needed to be linked in when compiling
*/

class SystemPerfMonitor{
public:
    void Init();
    void Destroy();
    void Start();
    void Stop();
    
    uint64 GetRefCycles();
    uint64 GetCycles();
    
    uint64 GetL2CacheMisses();
    double GetL2CacheHitRatio();

    uint64 GetL3CacheMisses();
    double GetL3CacheHitRatio();

    uint64 GetBytesReadFromMC();
    uint64 GetBytesWrittenToMC();

    double GetL2LostCycles();
    double GetL3LostCycles();
    
    double GetIPC();
    
    uint64 GetRetiredInstructions();

private:
    PCM *pcm_;
    SystemCounterState before_state_;
    SystemCounterState after_state_;
};

#endif  //SYSTEM_PERF_MONITOR_H
