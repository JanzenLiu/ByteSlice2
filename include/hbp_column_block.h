#ifndef HBP_COLUMN_BLOCK_H
#define HBP_COLUMN_BLOCK_H

#include    "column_block.h"

namespace byteslice{

/**
  Horizontal Bit Packing (HBP)
  format proposed by Y. Li and J. Patel,
  but utilize 256-bit AVX
Warning:
  Pay attention to output bit sequence.
  It should align with bit vector.
*/

template <size_t BIT_WIDTH>
class HbpColumnBlock: public ColumnBlock{
public:
    HbpColumnBlock(size_t num=0);
    virtual ~HbpColumnBlock();

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
    static const WordUnit kCodeMask = (1ULL << BIT_WIDTH) - 1;
    static const WordUnit kSectionMask = (1ULL << (BIT_WIDTH+1)) - 1;
    static const size_t kNumCodesPerWord = kNumWordBits / (BIT_WIDTH+1);
    static const size_t kNumPaddingBits = kNumWordBits - kNumCodesPerWord * (BIT_WIDTH+1);
    static const size_t kNumWordsPerSegment = BIT_WIDTH + 1;
    static const size_t kNumCodesPerSegment = kNumCodesPerWord * kNumWordsPerSegment;
    //how much memory to allocate for each word id
    //need to be 256-bit aligned
    static const size_t kMemSizePerWordId = 
        sizeof(AvxUnit)*CEIL(kNumTuplesPerBlock, kNumCodesPerSegment*(kNumAvxBits/kNumWordBits));

    //Scan Helper for literal
    template <Comparator CMP>
    void ScanHelper1(WordUnit literal, BitVectorBlock* bvblock, Bitwise bit_opt) const;
    template <Comparator CMP, Bitwise OPT>
    void ScanHelper2(WordUnit literal, BitVectorBlock* bvblock) const;

    //Scan helper for other block
    template <Comparator CMP>
    void ScanHelper1(const HbpColumnBlock<BIT_WIDTH>* other_block,
            BitVectorBlock* bvblock, Bitwise bit_opt) const;
    template <Comparator CMP, Bitwise OPT>
    void ScanHelper2(const HbpColumnBlock<BIT_WIDTH>* other_block, BitVectorBlock* bvblock) const;

    //Scan Kernel
    template <Comparator CMP>
    inline void ScanKernel(AvxUnit &result, AvxUnit &data, AvxUnit &other_data,
            AvxUnit &m_base, AvxUnit &m_delimiter, AvxUnit &m_complement) const;

    //Append a bit vector to the bv block
    //The number of effective bits in the bit vector is kNumCodesPerSegment
    template <Bitwise OPT>
    inline void AppendBitVector(WordUnit &working_unit, WordUnit &source, 
            BitVectorBlock* bvblock, size_t &cursor_word_id, size_t &cursor_offset) const;
    template <Bitwise OPT>
    inline void FlushBitVector(WordUnit &working_unit, BitVectorBlock* bvblock, 
                                size_t &cursor_word_id, size_t &cursor_offset) const;

    //This helper stores the prcomputed data
    //needed to retrieve a code from a segment
    //given a position in segment
    struct SeekHelper{
        size_t word_id_in_segment;
        size_t shift_in_word;
    };

    SeekHelper seek_helpers_[kNumCodesPerSegment];
    WordUnit* data_[kNumWordsPerSegment];
    size_t num_segments_;

};

template <size_t BIT_WIDTH>
inline WordUnit HbpColumnBlock<BIT_WIDTH>::GetTuple(size_t pos) const{
    size_t segment_id = pos / kNumCodesPerSegment;
    size_t id_in_segment = pos % kNumCodesPerSegment;
    WordUnit word = data_[seek_helpers_[id_in_segment].word_id_in_segment][segment_id];
    return (word >> (seek_helpers_[id_in_segment].shift_in_word)) & kCodeMask;
}

template <size_t BIT_WIDTH>
inline void HbpColumnBlock<BIT_WIDTH>::SetTuple(size_t pos, WordUnit value){
    value &= kCodeMask;
    size_t segment_id = pos / kNumCodesPerSegment;
    size_t id_in_segment = pos % kNumCodesPerSegment;
    //mask 1 on the corresponding position where the
    //code is to be set
    WordUnit mask = kSectionMask << seek_helpers_[id_in_segment].shift_in_word;
    WordUnit &word = data_[seek_helpers_[id_in_segment].word_id_in_segment][segment_id];
    word = (word & ~mask) | (value << seek_helpers_[id_in_segment].shift_in_word);
}


}   //namespace
#endif  //HBP_COLUMN_BLOCK_H
