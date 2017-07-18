#ifndef HYBRIDSLICE_COLUMN_BLOCK_H
#define HYBRIDSLICE_COLUMN_BLOCK_H

#include    "column_block.h"
#include    "avx-utility.h"

namespace byteslice{

static constexpr size_t kHybridSliceThreshold = 7;
static constexpr size_t kNumBitSlicesPerGroup = 4;

/**
  * @warning Bytes are *flipped* in its highest bit. Bit-slice components do not change.
  * @warning Pay strong attention to the sequence of tuples in ByteSlice and BitSlice format.
  */

template <size_t BIT_WIDTH>
class HybridSliceColumnBlock: public ColumnBlock{
public:
    HybridSliceColumnBlock(size_t num = kNumTuplesPerBlock);
    virtual ~HybridSliceColumnBlock();

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

    size_t GetNumByteSlices() const;
    size_t GetNumBitSlices() const;

private:
    //Scan Helper: literal
    template <Comparator CMP>
    void ScanHelper1(WordUnit literal, BitVectorBlock* bvblock, Bitwise bit_opt) const;
    template <Comparator CMP, Bitwise OPT>
    void ScanHelper2(WordUnit literal, BitVectorBlock* bvblock) const;

    //Scan Helper: other block
    template <Comparator CMP>
    void ScanHelper1(const HybridSliceColumnBlock<BIT_WIDTH>* other_block,
                            BitVectorBlock* bvblock, Bitwise bit_opt) const;
    template <Comparator CMP, Bitwise OPT>
    void ScanHelper2(const HybridSliceColumnBlock<BIT_WIDTH>* other_block,
                            BitVectorBlock* bvblock) const;
    
    //Scan Kernels
    template <Comparator CMP, size_t BYTE_ID>
    inline void ScanKernelByteSlice(const AvxUnit &byteslice1, const AvxUnit &byteslice2,
            AvxUnit &mask_less, AvxUnit &mask_greater, AvxUnit &mask_equal) const;
    template <Comparator CMP>
    inline void ScanKernelBitSlice(const AvxUnit &bitslice1, const AvxUnit &bitslice2,
            AvxUnit &m_lt, AvxUnit &m_gt, AvxUnit &m_eq) const;

    //constant parameters
    //Note: either one of kNumBitSlices or kNumPaddingBits must be zero.
    static constexpr size_t kNumResidualBits = BIT_WIDTH % 8;
    static constexpr size_t kNumBitSlices 
        = (kNumResidualBits > 0 && kNumResidualBits <= kHybridSliceThreshold)? kNumResidualBits:0;
    static constexpr size_t kNumByteSlices
        = (kNumBitSlices > 0)? (BIT_WIDTH / 8) : CEIL(BIT_WIDTH, 8);
    static constexpr size_t kNumPaddingBits
        = (8 * kNumByteSlices + kNumBitSlices) - BIT_WIDTH;
    

    ByteUnit* data_[4];
    WordUnit* bit_data_[8];

};

template <size_t BIT_WIDTH>
inline size_t HybridSliceColumnBlock<BIT_WIDTH>::GetNumByteSlices() const{
    return kNumByteSlices;
}

template <size_t BIT_WIDTH>
inline size_t HybridSliceColumnBlock<BIT_WIDTH>::GetNumBitSlices() const{
    return kNumBitSlices;
}

template <size_t BIT_WIDTH>
inline WordUnit HybridSliceColumnBlock<BIT_WIDTH>::GetTuple(size_t pos) const{
    assert(pos <= num_tuples_);
    WordUnit ret = 0ULL;
    switch(kNumByteSlices){
        case 4:
            ret = (static_cast<WordUnit>(FLIP(data_[0][pos])) << 24) |
                    (static_cast<WordUnit>(FLIP(data_[1][pos])) << 16) |
                    (static_cast<WordUnit>(FLIP(data_[2][pos])) << 8) |
                    static_cast<WordUnit>(FLIP(data_[3][pos]));
            break;
        case 3:
            ret = (static_cast<WordUnit>(FLIP(data_[0][pos])) << 16) |
                    (static_cast<WordUnit>(FLIP(data_[1][pos])) << 8) |
                    static_cast<WordUnit>(FLIP(data_[2][pos]));
            break;
        case 2:
            ret = (static_cast<WordUnit>(FLIP(data_[0][pos])) << 8) |
                    static_cast<WordUnit>(FLIP(data_[1][pos]));
            break;
        case 1:
            ret = static_cast<WordUnit>(FLIP(data_[0][pos]));
            break;
        case 0:
            break;
    }
    if(kNumPaddingBits > 0){
        ret >>= kNumPaddingBits;
    }
    if(kNumBitSlices > 0){
        size_t word_id = pos / kNumWordBits;
        size_t offset = pos % kNumWordBits;
        WordUnit mask = WordUnit(1) << offset;
        for(size_t i=0; i < kNumBitSlices; i++){
            ret <<= 1;
            ret |= ((mask & bit_data_[i][word_id]) >> offset);
        }
    }

    return ret;
}

template <size_t BIT_WIDTH>
inline void HybridSliceColumnBlock<BIT_WIDTH>::SetTuple(size_t pos, WordUnit value){
    assert(pos <= num_tuples_);
    //Fill in bit slice first
    if(kNumBitSlices > 0){
        size_t word_id = pos / kNumWordBits;
        size_t offset = pos % kNumWordBits;
        WordUnit mask = WordUnit(1) << offset;
        for(size_t i = kNumBitSlices - 1; i < kNumBitSlices; i--){
            WordUnit bit = (1ULL & value) << offset;
            bit_data_[i][word_id] &= ~mask;
            bit_data_[i][word_id] |= bit;
            value >>= 1;
        }
    }
    
    if(kNumPaddingBits > 0){
        value <<= kNumPaddingBits;
    }

    switch(kNumByteSlices){
        case 4:
            data_[0][pos] = FLIP(static_cast<ByteUnit>(value >> 24));
            data_[1][pos] = FLIP(static_cast<ByteUnit>(value >> 16));
            data_[2][pos] = FLIP(static_cast<ByteUnit>(value >> 8));
            data_[3][pos] = FLIP(static_cast<ByteUnit>(value));
            break;
        case 3:
            data_[0][pos] = FLIP(static_cast<ByteUnit>(value >> 16));
            data_[1][pos] = FLIP(static_cast<ByteUnit>(value >> 8));
            data_[2][pos] = FLIP(static_cast<ByteUnit>(value));
            break;
        case 2:
            data_[0][pos] = FLIP(static_cast<ByteUnit>(value >> 8));
            data_[1][pos] = FLIP(static_cast<ByteUnit>(value));
            break;
        case 1:
            data_[0][pos] = FLIP(static_cast<ByteUnit>(value));
            break;
        case 0:
            break;
    }
}

}   //namespace

#endif
