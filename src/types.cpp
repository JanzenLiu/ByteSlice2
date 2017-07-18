#include    "include/types.h"
#include <column.h>

namespace byteslice{

//for debug purpose
std::ostream& operator<< (std::ostream &out, ColumnType type){
    switch(type){
        case ColumnType::kNaive:
            out << "Naive";
            break;
        case ColumnType::kNaiveAvx:
            out << "NaiveAvx";
            break;
        case ColumnType::kHbp:
            out << "Hbp";
            break;
        case ColumnType::kVbp:
            out << "Vbp";
            break;
        case ColumnType::kBitSlice:
            out << "BitSlice";
            break;
        case ColumnType::kByteSlicePadRight:
            out << "ByteSlicePadRight";
            break;
        case ColumnType::kByteSlicePadLeft:
            out << "ByteSlicePadLeft";
            break;
        case ColumnType::kSuperscalar2ByteSlicePadRight:
            out << "kSuperscalar2ByteSlice";
            break;
        case ColumnType::kSuperscalar4ByteSlicePadRight:
            out << "kSuperscalar4ByteSlice";
            break;
        case ColumnType::kHybridSlice:
            out << "HybridSlice";
            break;
        case ColumnType::kNumpsAvx:
            out << "NumpsAvx";
            break;
        case ColumnType::kSmartHybridSlice:
            out << "SmartHybridSlice";
            break;
        case ColumnType::kAvx2Scan:
            out << "Avx2Scan";
            break;
        case ColumnType::kDualByteSlicePadLeft:
            out << "DualByteSlicePadLeft";
            break;
        case ColumnType::kDualByteSlicePadRight:
            out << "DualByteSlicePadRigt";
            break;
    }
    return out;
}

//for debug purpose
std::ostream& operator<< (std::ostream &out, Comparator comp){
    switch(comp){
        case Comparator::kEqual:
            out << "Equal";
            break;
        case Comparator::kInequal:
            out << "Inequal";
            break;
        case Comparator::kLess:
            out << "Less";
            break;
        case Comparator::kGreater:
            out << "Greater";
            break;
        case Comparator::kLessEqual:
            out << "LessEqual";
            break;
        case Comparator::kGreaterEqual:
            out << "GreaterEqual";
            break;
    }
    return out;
}


}   //namespace
