#include    "include/byte_mask_vector.h"
#include    <algorithm>
#include    <omp.h>

namespace byteslice{

ByteMaskVector::ByteMaskVector(const Column* column):
    ByteMaskVector(column->num_tuples()){
}

ByteMaskVector::ByteMaskVector(size_t num):
    num_(num){

    for(size_t count=0; count < num_; count += kNumTuplesPerBlock){
        ByteMaskBlock* new_block = 
            new ByteMaskBlock(std::min(kNumTuplesPerBlock, num_ - count));
        blocks_.push_back(new_block);
    }
    SetAllTrue();
}

ByteMaskVector::~ByteMaskVector(){
    while(!blocks_.empty()){
        delete blocks_.back();
        blocks_.pop_back();
    }
}

void ByteMaskVector::SetAllTrue(){
#   pragma omp parallel for schedule(dynamic)
    for(size_t i=0; i < blocks_.size(); i++){
        blocks_[i]->SetAllTrue();
    }
}

void ByteMaskVector::SetAllFalse(){
#   pragma omp parallel for schedule(dynamic)
    for(size_t i=0; i < blocks_.size(); i++){
        blocks_[i]->SetAllFalse();
    }
}

void ByteMaskVector::And(const ByteMaskVector* bmvector){
    assert(num_ == bmvector->num_);

#   pragma omp parallel for schedule(dynamic)
    for(size_t i=0; i < blocks_.size(); i++){
        blocks_[i]->And(bmvector->GetBMBlock(i));
    }
}

void ByteMaskVector::Or(const ByteMaskVector* bmvector){
    assert(num_ == bmvector->num_);

#   pragma omp parallel for schedule(dynamic)
    for(size_t i=0; i < blocks_.size(); i++){
        blocks_[i]->Or(bmvector->GetBMBlock(i));
    }
}

void ByteMaskVector::Set(const ByteMaskVector* bmvector){
	assert(num_ == bmvector->num_);

#	pragma omp parallel for schedule(dynamic)
	for(size_t i=0; i < blocks_.size(); i++){
		blocks_[i]->Set(bmvector->GetBMBlock(i));
	}
}

bool ByteMaskVector::GetByteMask(size_t pos) const{
    size_t block_id = pos / kNumTuplesPerBlock;
    size_t pos_in_block = pos % kNumTuplesPerBlock;
    return blocks_[block_id]->GetByteMask(pos_in_block);
}

void ByteMaskVector::SetByteMask(size_t pos, bool src){
    size_t block_id = pos / kNumTuplesPerBlock;
    size_t pos_in_block = pos % kNumTuplesPerBlock;
    blocks_[block_id]->SetByteMask(pos_in_block, src);
}

void ByteMaskVector::Condense(BitVector* bitvector, Bitwise opt) const{
	assert(num_ == bitvector->num());

#	pragma omp parallel for schedule(dynamic)
	for(size_t i=0; i < blocks_.size(); i++){
		blocks_[i]->Condense(bitvector->GetBVBlock(i), opt);
	}
}

}	//namespace