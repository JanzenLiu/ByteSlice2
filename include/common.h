#ifndef COMMON_H
#define COMMON_H

#include    <cstdlib>
#include    <cstdint>
#include    <cassert>
#include    <x86intrin.h>

#ifndef NUMTHREADS
#define NUMTHREADS (1)
#endif

namespace byteslice{

//hardware parameter: cache size in bytes
constexpr size_t kL1dCacheSize = 32*1024;
constexpr size_t kL2CacheSize = 256*1024;
constexpr size_t kL3CacheSize = 8*1024*1024;
constexpr size_t kL1TLBEntries = 64;

constexpr size_t kNumTuplesPerBlock = 1024*1024;    //each block contains 1M tuples
constexpr size_t kNumWordBits = 64;
constexpr size_t kNumAvxBits = 256;

constexpr size_t kNumRadixBits = 5;
constexpr size_t kNumPartitions = (1 << kNumRadixBits);

#define CEIL(X,Y) (((X)-1) / (Y) + 1)

#define POPCNT64(X) (_mm_popcnt_u64(X))

}   //namespace

#endif  //COMMON_H
