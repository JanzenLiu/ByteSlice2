#ifndef TYPES_H
#define TYPES_H

#include    <x86intrin.h>
#include    <cstdint>
#include    <iostream>
#include    <array>
#include    "common.h"

namespace byteslice{

typedef uint64_t WordUnit;
typedef uint8_t ByteUnit;
typedef uint16_t DualByteUnit;
typedef __m256i AvxUnit;
typedef uint32_t TupleId;   //assume 32 bits are sufficient for tuple ID

typedef std::array<size_t, kNumPartitions> Hist;

struct JoinIndex{
    JoinIndex(TupleId le, TupleId ri):
        left(le),
        right(ri){
    }

    TupleId left;
    TupleId right;
};

enum class ColumnType{
    kNumpsAvx,
    kNaive,
    kNaiveAvx,
    kBitSlice,
    kVbp,
    kHbp,
    kByteSlicePadRight,
    kByteSlicePadLeft,
    kHybridSlice,
    kSmartHybridSlice,
    kAvx2Scan,
    kDualByteSlicePadRight,
    kDualByteSlicePadLeft,
	kSuperscalar2ByteSlicePadRight,
	kSuperscalar2ByteSlicePadLeft,
	kSuperscalar4ByteSlicePadRight,
	kSuperscalar4ByteSlicePadLeft
};


enum class Bitwise{
    kSet,
    kAnd,
    kOr
};

enum class Comparator{
    kEqual,
    kInequal,
    kLess,
    kGreater,
    kLessEqual,
    kGreaterEqual
};

enum class Direction{
    kLeft,
    kRight
};


//for debug use
std::ostream& operator<< (std::ostream &out, ColumnType type);
std::ostream& operator<< (std::ostream &out, Comparator comp);

}   //namespace

#endif //TYPES_H
