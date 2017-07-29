#include    "include/byte_mask_block.h"
#include    <cstring>
#include    <cstdlib>

namespace byteslice{

static constexpr size_t kAllocSize
    = sizeof(ByteUnit)*CEIL(kNumTuplesPerBlock, sizeof(AvxUnit))*sizeof(AvxUnit);

ByteMaskBlock::ByteMaskBlock(size_t num):
    num_(num){
    assert(num_ <= kNumTuplesPerBlock);
    int ret 
        = posix_memalign((void**)&data_, 32, kAllocSize);
    ret = ret;
    SetAllTrue();
}

ByteMaskBlock::~ByteMaskBlock(){
    free(data_);
}

void ByteMaskBlock::SetAllTrue(){
    memset(data_, static_cast<ByteUnit>(-1ULL), kAllocSize);
    ClearTail();
}

void ByteMaskBlock::SetAllFalse(){
    memset(data_, 0ULL, kAllocSize);
}

void ByteMaskBlock::ClearTail(){
    memset(data_+num_, 0ULL, (kNumTuplesPerBlock-num_)*sizeof(ByteUnit));
}

void ByteMaskBlock::And(const ByteMaskBlock* block){
    for(size_t i = 0; i < num_; i++){
        data_[i] &= block->GetByte(i);
    }
}

void ByteMaskBlock::Or(const ByteMaskBlock* block){
    for(size_t i = 0l i < num_; i++){
        data_[i] |= block->GetByte(i);
    }
}

void ByteMaskBlock::Condense(BitVectorBlock* bvblk, Bitwise opt) const{
    switch(opt){
        case Bitwise::kSet:
            return CondenseHelper<Bitwise::kSet>(bvblk);
        case Bitwise::kAnd:
            return CondenseHelper<Bitwise::kAnd>(bvblk);
        case Bitwise::kOr:
            return CondenseHelper<Bitwise::kOr>(bvblk);
    }
}

template <Bitwise OPT>
void ByteMaskBlock::CondenseHelper(BitVectorBlock* bvblk) const{
    //For every 64 masks, generates a word to be written to bvblk
    for(size_t offset = 0; offset < num_; offset += kNumWordBits){
        WordUnit word = 0ULL;
        //For every 32 masks, use movemask intruction to collect the bits
        for(size_t i = 0; i < kNumWordBits; i += sizeof(AvxUnit)){
            uint32_t mmask = _mm256_movemask_epi8(GetAvxUnit(offset+i));
            word |= (static_cast<WordUnit>(mmask) << i);
        }

        //Merge the word into bitvector
        size_t bv_word_id = offset / kNumWordBits;
        WordUnit out = word;
        switch(OPT){
            case Bitwise::kSet:
                break;
            case Bitwise::kAnd:
                out &= bvblk->GetWordUnit(bv_word_id);
                break;
            case Bitwise::kOr:
                out |= bvblk->GetWordUnit(bv_word_id);
                break;
        }
        bvblk->SetWordUnit(out, bv_word_id);
    }

    bvblk->ClearTail();

}

}   //namespace
