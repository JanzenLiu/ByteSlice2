#ifndef BITSLICE_COLUMN_BLOCK_H
#define BITSLICE_COLUMN_BLOCK_H

#include    "column_block.h"

namespace byteslice{

static const size_t kMemSizePerBitSlice 
    = sizeof(AvxUnit)*CEIL(kNumTuplesPerBlock, kNumAvxBits);

/**
  Cache-optimized SIMD-based bit-sliced format.
  Pay attention to the bit sequence, it should
  align with bit vector.
*/
class BitSliceColumnBlock: public ColumnBlock{
public:
    BitSliceColumnBlock(size_t bit_width):
        BitSliceColumnBlock(kNumTuplesPerBlock, bit_width){}
    BitSliceColumnBlock(size_t num, size_t bit_width);
    virtual ~BitSliceColumnBlock();

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

private:
    static const size_t kNumBitsPerGroup = 4;
    WordUnit* data_[64];
    const size_t num_bit_groups_;

    //Scan helper for literal
    template <Comparator CMP>
    void ScanHelper1(WordUnit literal, BitVectorBlock* bvblock, Bitwise bit_opt) const;
    template <Comparator CMP, Bitwise OPT>
    void ScanHelper2(WordUnit literal, BitVectorBlock* bvblock) const;

    //Scan helper for other block
    template <Comparator CMP>
    void ScanHelper1(const BitSliceColumnBlock* other_block,
            BitVectorBlock* bvblock, Bitwise bit_opt) const;
    template <Comparator CMP, Bitwise OPT>
    void ScanHelper2(const BitSliceColumnBlock* other_block, BitVectorBlock* bvblock) const;


    //Scan core: update intermidiate masks based on two data
    template <Comparator CMP>
    inline void ScanKernel(const AvxUnit &data, const AvxUnit &other, 
            AvxUnit &m_lt, AvxUnit &m_gt, AvxUnit &m_eq) const;
};

inline WordUnit BitSliceColumnBlock::GetTuple(size_t pos) const{
    assert(pos <= num_tuples_);
    WordUnit ret = 0ULL;
    size_t word_id = pos / kNumWordBits;
    size_t offset = pos % kNumWordBits;
    WordUnit mask = WordUnit(1) << offset;
    for(size_t i=0; i < bit_width_; i++){
        //__builtin_prefetch(data_[i+4] + word_id);
        ret <<= 1;
        ret |= ((mask & data_[i][word_id]) >> offset);
    }
    return ret;
}

inline void BitSliceColumnBlock::SetTuple(size_t pos, WordUnit value) {
    assert(pos <= num_tuples_);
    size_t word_id = pos / kNumWordBits;
    size_t offset = pos % kNumWordBits;
    WordUnit mask = WordUnit(1) << offset;
    //from LSB to MSB
    for(ssize_t i=bit_width_-1; i >= 0; i--){
        WordUnit bit = (WordUnit(1) & value) << offset;
        data_[i][word_id] &= ~mask;
        data_[i][word_id] |= bit;
        value >>= 1;
    }
}

}   //namespace
#endif  //BITSLICE_COLUMN_BLOCK_H
