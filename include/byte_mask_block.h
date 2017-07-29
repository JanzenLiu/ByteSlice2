#ifndef _BYTE_MASK_BLOCK_H_
#define _BYTE_MASK_BLOCK_H_

#include    "common.h"
#include    "types.h"
#include    "bitvector_block.h"

namespace byteslice{

/**
  * @brief Host byte-masks that are generated as a result of
  * SIMD comparisons. A ByteMaskBlock corresponds to a column block
  * and can be condensed into a BitVectorBlock with every 8-bit
  * mask compressed as 1 bit.
  */
class ByteMaskBlock{
public:
    ByteMaskBlock(size_t num);
    ~ByteMaskBlock();
    
    void SetAllTrue();
    void SetAllFalse();
    void ClearTail();
    void And(const ByteMaskBlock* block);
    void Or(const ByteMaskBlock* block);
    void Set(const ByteMaskBlock* block);
    
    void Condense(BitVectorBlock* bvblk, Bitwise opt = Bitwise::kSet) const;

    //mutator and accessor
    void SetAvxUnit(size_t offset, AvxUnit src);
    AvxUnit GetAvxUnit(size_t offset) const;
    size_t num() const;
    bool GetByteMask(size_t offset) const;
    void SetByteMask(size_t offset, bool src);
    ByteUnit GetByte(size_t offset) const;

private:
    ByteUnit* data_;
    size_t num_;

    template <Bitwise OPT>
    void CondenseHelper(BitVectorBlock* bvblk) const;
};

inline void ByteMaskBlock::SetAvxUnit(size_t offset, AvxUnit src){
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(data_+offset), src);
}

inline AvxUnit ByteMaskBlock::GetAvxUnit(size_t offset) const{
    return _mm256_lddqu_si256(reinterpret_cast<__m256i*>(data_+offset));
}

inline size_t ByteMaskBlock::num() const{
    return num_;
}

inline bool ByteMaskBlock::GetByteMask(size_t offset) const{
    return data_[offset];
}

inline void ByteMaskBlock::SetByteMask(size_t offset, bool src){
    data_[offset] = src?(static_cast<ByteUnit>(-1)):0;
}

inline ByteUnit ByteMaskBlock::GetByte(size_t offset) const{
    return data_[offset];
}


}   //namespace
#endif
