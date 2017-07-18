#ifndef VBP_COLUMN_BLOCK
#define VBP_COLUMN_BLOCK

#include    "column_block.h"

namespace byteslice{


template <size_t BIT_WIDTH>
class VbpColumnBlock: public ColumnBlock{
public:
    VbpColumnBlock(size_t num = 0);
    ~VbpColumnBlock();
    
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
    static constexpr size_t kNumBitsPerGroup = 4;
    static constexpr size_t kNumBitGroups = CEIL(BIT_WIDTH, kNumBitsPerGroup);

    WordUnit* data_[kNumBitGroups];
    size_t bitgroup_helper_[kNumBitGroups];

    //Scan Helper for literal
    template <Comparator CMP>
    void ScanHelper1(WordUnit literal, BitVectorBlock* bvblock, Bitwise bit_opt) const;
    template <Comparator CMP, Bitwise OPT>
    void ScanHelper2(WordUnit literal, BitVectorBlock* bvblock) const;

    //Scan helper for other block
    template <Comparator CMP>
    void ScanHelper1(const VbpColumnBlock<BIT_WIDTH>* other_block,
            BitVectorBlock* bvblock, Bitwise bit_opt) const;
    template <Comparator CMP, Bitwise OPT>
    void ScanHelper2(const VbpColumnBlock<BIT_WIDTH>* other_block, BitVectorBlock* bvblock) const;

    //Scan Kernel
    template <Comparator CMP>
    inline void ScanKernel(const AvxUnit &data, const AvxUnit &other,
                           AvxUnit &m_lt, AvxUnit &m_gt, AvxUnit &m_eq) const;

};

template <size_t BIT_WIDTH>
inline WordUnit VbpColumnBlock<BIT_WIDTH>::GetTuple(size_t pos) const{
    assert(pos <= num_tuples_);

    constexpr size_t stride = sizeof(AvxUnit) / sizeof(WordUnit);
    WordUnit ret = 0ULL;
    size_t segment_id = pos / kNumAvxBits;
    size_t word_in_segment = (pos % kNumAvxBits) / kNumWordBits;
    size_t offset_in_word = (pos % kNumAvxBits) % kNumWordBits;
    WordUnit mask = WordUnit(1) << offset_in_word;

    for(size_t gid=0; gid < kNumBitGroups; gid++){
        size_t base_word_id = segment_id * bitgroup_helper_[gid] * stride;
        for(size_t j = 0; j < bitgroup_helper_[gid]; j++){
            ret <<= 1;
            ret |= 
                (mask & 
                 data_[gid][base_word_id + word_in_segment + j * stride]) 
                >> offset_in_word;
        }
    }

    return ret;
}

template <size_t BIT_WIDTH>
inline void VbpColumnBlock<BIT_WIDTH>::SetTuple(size_t pos, WordUnit value){
    assert(pos <= num_tuples_);

    constexpr size_t stride = sizeof(AvxUnit) / sizeof(WordUnit);

    size_t segment_id = pos / kNumAvxBits;
    size_t word_in_segment = (pos % kNumAvxBits) / kNumWordBits;
    size_t offset_in_word = (pos % kNumAvxBits) % kNumWordBits;
    WordUnit mask = WordUnit(1) << offset_in_word;

    for(size_t gid=0; gid < kNumBitGroups; gid++){
        size_t base_word_id = segment_id * bitgroup_helper_[gid] * stride;
        for(size_t j = 0; j < bitgroup_helper_[gid]; j++){
            size_t bit_id = gid * kNumBitsPerGroup + j;
            WordUnit bit = ((value >> (BIT_WIDTH -1 - bit_id)) & 1ULL) << offset_in_word;
            WordUnit &target =
                 data_[gid][base_word_id + word_in_segment + j * stride];
            target &= ~mask;
            target |= bit;
        }
    }
}

}   //namespace
#endif
