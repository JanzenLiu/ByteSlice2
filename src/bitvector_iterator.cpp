#include    "include/bitvector_iterator.h"

namespace byteslice{

BitVectorIterator::BitVectorIterator(const BitVector *bitvector):
    bitvector_(bitvector),
    cur_block_(bitvector_->GetBVBlock(0)){
}


BitVectorIterator::~BitVectorIterator(){
}


}   //namespace
