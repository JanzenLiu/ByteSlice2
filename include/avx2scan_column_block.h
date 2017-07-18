#ifndef AVX2SCAN_COLUMN_BLOCK_H
#define AVX2SCAN_COLUMN_BLOCK_H

#include    "column_block.h"
#include    "avx-utility.h"

namespace byteslice{
/**
  Store data in bit-packed format.
  Unpack and Scan with AVX2 instructions.
  Pay attention to the bit sequence and byte sequence:
  Always from least significant to most significant
  Warning: as AVX2 currently only support signed integer comparison,
  values are flipped before precate evaluation to preserve order.
*/

template <size_t BIT_WIDTH>
class Avx2ScanColumnBlock: public ColumnBlock{
public:
    Avx2ScanColumnBlock(size_t num=0);
    virtual ~Avx2ScanColumnBlock();

    WordUnit GetTuple(size_t pos) const override;
    void SetTuple(size_t pos, WordUnit value) override;

    void Scan(Comparator comparator, WordUnit literal, BitVectorBlock* bv_block,
            Bitwise bit_opt=Bitwise::kSet) const override;
    void Scan(Comparator comparator, const ColumnBlock* column_block,
            BitVectorBlock* bv_block, Bitwise bit_opt=Bitwise::kSet) const override;
    void BulkLoadArray(const WordUnit* codes, size_t num, size_t start_pos=0) override;

    void SerToFile(SequentialWriteBinaryFile &file) const override;
    void DeserFromFile(const SequentialReadBinaryFile &file) override;
    bool Resize(size_t size) override;

private:
    ByteUnit* data_;

    //Scan Helper: literal
    template <Comparator CMP>
    void ScanHelper1(WordUnit literal, BitVectorBlock* bvblock, Bitwise bit_opt) const;
    template <Comparator CMP, Bitwise OPT>
    void ScanHelper2(WordUnit literal, BitVectorBlock* bvblock) const;

    //Scan Helper: other block
    template <Comparator CMP>
    void ScanHelper1(const Avx2ScanColumnBlock<BIT_WIDTH>* other_block,
                            BitVectorBlock* bvblock, Bitwise bit_opt) const;
    template <Comparator CMP, Bitwise OPT>
    void ScanHelper2(const Avx2ScanColumnBlock<BIT_WIDTH>* other_block,
                            BitVectorBlock* bvblock) const;

    //Scan Kernel
    template <Comparator CMP>
    inline AvxUnit ScanKernel(AvxUnit a, AvxUnit b) const;

    //Unpack 8 value starting with aligned boundary
    inline AvxUnit Unpack8(size_t index, const AvxUnit& shiftv, 
                           const AvxUnit& shufflev, const AvxUnit& code_maskv) const;
    //SetUp the masks needed for unpacking
    void SetUpMasks(AvxUnit& shiftv, AvxUnit& shufflev, AvxUnit& code_maskv) const;

    //Gather and Scatter bytes as WordUnit (hide endian issue)
    //Warning: the word contains at most ONE code (higher bits are empty)
    WordUnit GatherBytesAsWord(size_t byte_pos) const;
    void ScatterWordToBytes(size_t byte_pos, WordUnit word);


    //parameters
    static constexpr size_t kNumTuplePerSegment = 8;
    static constexpr size_t kNumBytesPerSegment = BIT_WIDTH;
    static constexpr WordUnit kCodeMask = (1ULL << BIT_WIDTH) - 1;
    //the largest number of bytes that a single code can spans
    //Should be at most 5 for code no wider than 32 bits
    static constexpr size_t kNumMostSpannedBytesPerCode = CEIL(BIT_WIDTH-1, 8) + 1;

    //Precompute the starting word id and shift 
    struct Helper{
        size_t byte_id;
        size_t shift;
    };
    Helper helpers_[kNumTuplePerSegment];
};

template <size_t BIT_WIDTH>
inline WordUnit Avx2ScanColumnBlock<BIT_WIDTH>::GatherBytesAsWord(size_t byte_pos) const{
    WordUnit ret = 0ULL;
    size_t& index = byte_pos;
    switch(kNumMostSpannedBytesPerCode){
        case 5:
            ret |= static_cast<WordUnit>(data_[index + 4]) << 32;
        case 4:
            ret |= static_cast<WordUnit>(data_[index + 3]) << 24;
        case 3:
            ret |= static_cast<WordUnit>(data_[index + 2]) << 16;
        case 2:
            ret |= static_cast<WordUnit>(data_[index + 1]) << 8;
        case 1:
            ret |= static_cast<WordUnit>(data_[index]);
    }
    
    return ret;
}

template <size_t BIT_WIDTH>
inline void Avx2ScanColumnBlock<BIT_WIDTH>::ScatterWordToBytes(size_t byte_pos, WordUnit word){
    size_t& index = byte_pos;
    switch(kNumMostSpannedBytesPerCode){
        case 5:
            data_[index + 4] = static_cast<ByteUnit>(word >> 32);
        case 4:
            data_[index + 3] = static_cast<ByteUnit>(word >> 24);
        case 3:
            data_[index + 2] = static_cast<ByteUnit>(word >> 16);
        case 2:
            data_[index + 1] = static_cast<ByteUnit>(word >> 8);
        case 1:
            data_[index + 0] = static_cast<ByteUnit>(word);
    }

}

template <size_t BIT_WIDTH>
inline WordUnit Avx2ScanColumnBlock<BIT_WIDTH>::GetTuple(size_t pos) const{
    size_t segment_id = pos / kNumTuplePerSegment;
    size_t pos_in_segment = pos % kNumTuplePerSegment;
    size_t index = (kNumBytesPerSegment * segment_id) + 
                            helpers_[pos_in_segment].byte_id;

    WordUnit ret;
    ret = GatherBytesAsWord(index);

    ret >>= helpers_[pos_in_segment].shift;
    ret &= kCodeMask;

    return ret;

}

template <size_t BIT_WIDTH>
inline void Avx2ScanColumnBlock<BIT_WIDTH>::SetTuple(size_t pos, WordUnit value){
    size_t segment_id = pos / kNumTuplePerSegment;
    size_t pos_in_segment = pos % kNumTuplePerSegment;
    size_t shift = helpers_[pos_in_segment].shift;
    size_t index = (kNumBytesPerSegment * segment_id) + 
                            helpers_[pos_in_segment].byte_id;

    WordUnit w = GatherBytesAsWord(index);
    WordUnit mask = kCodeMask << shift;
    w = (w & ~mask) | (value << shift);

    ScatterWordToBytes(index, w);

}

}   //namespace

#endif
