#ifndef NAIVE_AVX_COLUMN_BLOCK_H
#define NAIVE_AVX_COLUMN_BLOCK_H

#include    "column_block.h"
#include    "avx-utility.h"

namespace byteslice{

/**
  Store data in 32-byte aligned plain array.
  Scan with AVX instructions
  Warning: as AVX2 currently only support signed integer comparison,
  values are flipped in internal storage to preserve order.
*/

template <typename DTYPE>
class NaiveAvxColumnBlock: public ColumnBlock{
public:
    NaiveAvxColumnBlock(size_t num=0);
    virtual ~NaiveAvxColumnBlock();

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
    DTYPE* data_;

    //scan helper: against a given literal
    template <Comparator CMP>
    void ScanHelper1(WordUnit literal, BitVectorBlock* bv_block, Bitwise bit_opt) const;
    template <Comparator CMP, Bitwise OPT>
    void ScanHelper2(WordUnit literal, BitVectorBlock* bv_block) const;
    template <Comparator CMP>
    inline void ScanIteration(AvxUnit &result, const size_t &offset, const AvxUnit &lit256) const;

    //scan helper: against another column_block
    template <Comparator CMP>
    void ScanHelper1(const ColumnBlock* colblock, BitVectorBlock* bvblock, Bitwise bit_opt) const;
    template <Comparator CMP, Bitwise OPT>
    void ScanHelper2(const ColumnBlock* colblock, BitVectorBlock* bvblock) const;
    template <Comparator CMP>
    inline void ScanIteration(AvxUnit &result, const size_t &offset, 
            const ColumnBlock* other_block) const;

    template <Comparator CMP>
    inline void ScanKernel(const AvxUnit &data, const AvxUnit &other, WordUnit (&buffer)[4], 
            const size_t &index) const;


    friend class Avx32JoinableBlock;
    friend class Avx32BlockBNLJoin;
};

template <typename DTYPE>
inline WordUnit NaiveAvxColumnBlock<DTYPE>::GetTuple(size_t pos) const{
    return static_cast<WordUnit>(FLIP<DTYPE>(data_[pos]));
}

template <typename DTYPE>
inline void NaiveAvxColumnBlock<DTYPE>::SetTuple(size_t pos, WordUnit value){
    data_[pos] = FLIP<DTYPE>(static_cast<DTYPE>(value));
}


}   //namespace

#endif  //NAIVE_AVX_COLUMN_BLOCK_H
