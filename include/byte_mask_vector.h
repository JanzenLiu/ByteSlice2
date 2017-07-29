#ifndef BYTEMASKVECTOR_H
#define BYTEMASKVECTOR_H

#include    <vector>
#include    "types.h"
#include    "column.h"
#include    "bitvector.h"
#include    "byte_mask_block.h"

namespace byteslice{

/**
    Notice: ByteMaskVector is created based on a column. We don't resize ByteMaskVectors.
*/

class Column;

/**
  * @brief Host byte-masks that are generated as a result of
  * SIMD comparisons. A ByteMask corresponds to a column and
  * can be condensed into a BitVector with every 8-bit
  * mask compressed as 1 bit.
  */
class ByteMaskVector{
public:
    ByteMaskVector(const Column* column);
    ByteMaskVector(size_t num);
    ~ByteMaskVector();

    //entirety opeartion
    void SetAllTrue();
    void SetAllFalse();

    //bitwise operation
    void And(const ByteMaskVector* bmvector);
    void Or(const ByteMaskVector* bmvector);
    void Set(const ByteMaskVector* bmvector);

    void Condense(BitVector* bitvector, Bitwise opt = Bitwise::kSet) const;

    //single byte operation
    bool GetByteMask(size_t pos) const;
    void SetByteMask(size_t pos, bool src);

    //accessors
    size_t num() const;
    size_t GetNumBlocks() const;
    ByteMaskBlock* GetBMBlock(size_t id) const;

private:
    std::vector<ByteMaskBlock*> blocks_;
    const size_t num_;

};

inline size_t ByteMaskVector::num() const{
    return num_;
}

inline size_t ByteMaskVector::GetNumBlocks() const{
    return blocks_.size();
}


inline ByteMaskBlock* ByteMaskVector::GetBMBlock(size_t id) const{
    return blocks_[id];
}

};	//namespace

#endif  //BYTEMASKVECTOR_H
