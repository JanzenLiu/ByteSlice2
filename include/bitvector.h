#ifndef BITVECTOR_H
#define BITVECTOR_H

#include    <vector>
#include    "types.h"
#include    "column.h"
#include    "bitvector_block.h"

namespace byteslice{

/**
    Notice: BitVector is created based on a column. We don't resize BitVectors.
*/

class Column;

class BitVector{
/*
   The bit vector (blocks) are guaranteed to be 32-byte aligned
   so that it can also be used with 256-bit AVX instruction
*/
public:
    BitVector(const Column* column);
    BitVector(size_t num);
    ~BitVector();

    void SetOnes();
    void SetZeros();
    size_t CountOnes() const;

    //bitwise combination
    void And(const BitVector* bitvector);
    void Or(const BitVector* bitvector);
    void Not();

    //bit manipulation
    bool GetBit(size_t pos);
    void SetBit(size_t pos);
    void UnsetBit(size_t pos);

    //accessors
    size_t num() const;
    size_t GetNumBlocks() const;
    BitVectorBlock* GetBVBlock(size_t id) const;

private:
    std::vector<BitVectorBlock*> blocks_;
    const size_t num_;

};

inline size_t BitVector::num() const{
    return num_;
}

inline size_t BitVector::GetNumBlocks() const{
    return blocks_.size();
}


inline BitVectorBlock* BitVector::GetBVBlock(size_t id) const{
    return blocks_[id];
}

}   //namespace

#endif  //BITVECTOR_H
