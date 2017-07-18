#ifndef DUALBYTESLICE_COLUMN_BLOCK_H
#define DUALBYTESLICE_COLUMN_BLOCK_H

#include    "column_block.h"
#include    "avx-utility.h"

namespace byteslice{

/**
Warning:
    DualBytes are FLIPPED in internal storage to preserve order.
*/
template <size_t BIT_WIDTH, Direction PDIRECTION = Direction::kRight>
class DualByteSliceColumnBlock: public ColumnBlock{
public:
    DualByteSliceColumnBlock(size_t num=kNumTuplesPerBlock);
    virtual ~DualByteSliceColumnBlock();

    WordUnit GetTuple(size_t pos) const override;
    void SetTuple(size_t pos, WordUnit value) override;

    void Scan(Comparator comparator, WordUnit literal, BitVectorBlock* bvblock,
            Bitwise bit_opt = Bitwise::kSet) const override;
    void Scan(Comparator comparator, const ColumnBlock* other_block,
            BitVectorBlock* bvblock, Bitwise bit_opt = Bitwise::kSet) const override;


    void BulkLoadArray(const WordUnit* codes, size_t num, size_t start_pos = 0) override;

    void SerToFile(SequentialWriteBinaryFile &file) const override;
    void DeserFromFile(const SequentialReadBinaryFile &file) override;
    bool Resize(size_t size) override;

    Direction GetPadDirection();
    
private:
    //Scan Helper: literal
    template <Comparator CMP>
    void ScanHelper1(WordUnit literal, BitVectorBlock* bvblock, Bitwise bit_opt) const;
    template <Comparator CMP, Bitwise OPT>
    void ScanHelper2(WordUnit literal, BitVectorBlock* bvblock) const;

    //Scan Helper: other block
    template <Comparator CMP>
    void ScanHelper1(const DualByteSliceColumnBlock<BIT_WIDTH, PDIRECTION>* other_block,
                            BitVectorBlock* bvblock, Bitwise bit_opt) const;
    template <Comparator CMP, Bitwise OPT>
    void ScanHelper2(const DualByteSliceColumnBlock<BIT_WIDTH, PDIRECTION>* other_block,
                            BitVectorBlock* bvblock) const;

    //Scan Kernel
    template <Comparator CMP>
    inline void ScanKernel(const AvxUnit &byteslice1, const AvxUnit &byteslice2,
            AvxUnit &mask_less, AvxUnit &mask_greater, AvxUnit &mask_equal) const;
    template <Comparator CMP, size_t DUALBYTE_ID>
    inline void ScanKernel2(const AvxUnit &byteslice1, const AvxUnit &byteslice2,
            AvxUnit &mask_less, AvxUnit &mask_greater, AvxUnit &mask_equal) const;

    static constexpr size_t kNumDualBytesPerCode = CEIL(BIT_WIDTH, 16);
    static constexpr size_t kNumPaddingBits = kNumDualBytesPerCode * 16 - BIT_WIDTH;
    static constexpr Direction kPadDirection = PDIRECTION;
    static constexpr WordUnit kCodeMask = (1ULL << BIT_WIDTH) - 1;

    static constexpr size_t kMemSizePerDualByteSlice = 
        sizeof(DualByteUnit)*CEIL(kNumTuplesPerBlock, sizeof(AvxUnit)/sizeof(DualByteUnit))*(sizeof(AvxUnit)/sizeof(DualByteUnit));

    DualByteUnit* data_[4];

};

template <size_t BIT_WIDTH, Direction PDIRECTION>
inline WordUnit DualByteSliceColumnBlock<BIT_WIDTH, PDIRECTION>::GetTuple(size_t pos) const{
    assert(kNumDualBytesPerCode <= 2);
    WordUnit ret = 0ULL;
    switch(kNumDualBytesPerCode){
//         case 4:
//             ret = (static_cast<WordUnit>(FLIP(data_[0][pos])) << 24) |
//                     (static_cast<WordUnit>(FLIP(data_[1][pos])) << 16) |
//                     (static_cast<WordUnit>(FLIP(data_[2][pos])) << 8) |
//                     static_cast<WordUnit>(FLIP(data_[3][pos]));
//             break;
//         case 3:
//             ret = (static_cast<WordUnit>(FLIP(data_[0][pos])) << 16) |
//                     (static_cast<WordUnit>(FLIP(data_[1][pos])) << 8) |
//                     static_cast<WordUnit>(FLIP(data_[2][pos]));
//             break;
        case 2:
            ret = (static_cast<WordUnit>(FLIP(data_[0][pos])) << 16) |
                    static_cast<WordUnit>(FLIP(data_[1][pos]));
            break;
        case 1:
            ret = static_cast<WordUnit>(FLIP(data_[0][pos]));
            break;
    }
    switch(PDIRECTION){
        case Direction::kRight:
            ret >>= kNumPaddingBits;
            break;
        case Direction::kLeft:
            break;
    }
    return ret;
}

template <size_t BIT_WIDTH, Direction PDIRECTION>
inline void DualByteSliceColumnBlock<BIT_WIDTH, PDIRECTION>::SetTuple(size_t pos, WordUnit value){
    switch(PDIRECTION){
        case Direction::kRight:
            value <<= kNumPaddingBits;
            break;
        case Direction::kLeft:
            break;
    }

    switch(kNumDualBytesPerCode){
//         case 4:
//             data_[0][pos] = FLIP(static_cast<ByteUnit>(value >> 24));
//             data_[1][pos] = FLIP(static_cast<ByteUnit>(value >> 16));
//             data_[2][pos] = FLIP(static_cast<ByteUnit>(value >> 8));
//             data_[3][pos] = FLIP(static_cast<ByteUnit>(value));
//             break;
//         case 3:
//             data_[0][pos] = FLIP(static_cast<ByteUnit>(value >> 16));
//             data_[1][pos] = FLIP(static_cast<ByteUnit>(value >> 8));
//             data_[2][pos] = FLIP(static_cast<ByteUnit>(value));
//             break;
        case 2:
            data_[0][pos] = FLIP(static_cast<DualByteUnit>(value >> 16));
            data_[1][pos] = FLIP(static_cast<DualByteUnit>(value));
            break;
        case 1:
            data_[0][pos] = FLIP(static_cast<DualByteUnit>(value));
            break;
    }
}

}   //namespace
#endif
