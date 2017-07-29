#include    "column.h"
#include    <algorithm>
#include    <iostream>
#include    <fstream>
#include    <omp.h>


namespace byteslice{

Column::Column(ColumnType type, size_t bit_width, size_t num):
    type_(type), bit_width_(bit_width), num_tuples_(num){

    for(size_t count=0; count < num; count += kNumTuplesPerBlock){
        ColumnBlock* new_block = CreateNewBlock();
        new_block->Resize(std::min(kNumTuplesPerBlock, num-count));
        blocks_.push_back(new_block);
    }
}

Column::Column():
    Column(ColumnType::kNaive, 32){
}

Column::~Column(){
    Destroy();
}

void Column::Destroy(){
    while(!blocks_.empty()){
        delete blocks_.back();
        blocks_.pop_back();
    }
}

WordUnit Column::GetTuple(size_t id) const{
    assert(id < num_tuples_);
    size_t block_id = id / kNumTuplesPerBlock;
    size_t pos_in_block = id % kNumTuplesPerBlock;
    return blocks_[block_id]->GetTuple(pos_in_block);
}

void Column::SetTuple(size_t id, WordUnit value){
    size_t block_id = id / kNumTuplesPerBlock;
    size_t pos_in_block = id % kNumTuplesPerBlock;
    blocks_[block_id]->SetTuple(pos_in_block, value);
}

size_t Column::LoadTextFile(std::string filepath){
    std::ifstream infile;
    infile.open(filepath, std::ifstream::in);
    if(!infile.good()){
        std::cerr << "Can't open file: " << filepath << std::endl;
    }
    WordUnit val;
    for(size_t id=0; (id < num_tuples()) && (infile >> val); id++){
        SetTuple(id, val);
    }
    infile.close();
}

void Column::Resize(size_t num){
    num_tuples_ = num;
    const size_t new_num_blocks = CEIL(num, kNumTuplesPerBlock);
    const size_t old_num_blocks = blocks_.size();
    if(new_num_blocks > old_num_blocks){    //need to add blocks
        //fill up the last block
        blocks_[old_num_blocks-1]->Resize(kNumTuplesPerBlock);
        //append new blocks
        for(size_t bid=old_num_blocks; bid < new_num_blocks; bid++){
            ColumnBlock* new_block = CreateNewBlock();
            new_block->Resize(kNumTuplesPerBlock);
            blocks_.push_back(new_block);
        }
    }
    else if(new_num_blocks < old_num_blocks){   //need to remove blocks
        for(size_t bid=old_num_blocks-1; bid > new_num_blocks; bid--){
            delete blocks_.back();
            blocks_.pop_back();
        }
    }
    //now the number of block is desired
    //correct the size of the last block
    size_t num_tuples_last_block = num % kNumTuplesPerBlock;
    if(0 < num_tuples_last_block){
        blocks_.back()->Resize(num_tuples_last_block);
    }

    assert(blocks_.size() == new_num_blocks);
}

void Column::SerToFile(SequentialWriteBinaryFile &file) const{

}

void Column::DeserFromFile(const SequentialReadBinaryFile &file){

}

void Column::BulkLoadArray(const WordUnit* codes, size_t num, size_t pos){
    assert(pos + num <= num_tuples_);
    size_t block_id = pos / kNumTuplesPerBlock;
    size_t pos_in_block = pos % kNumTuplesPerBlock;
    size_t num_remain_tuples = num;
    const WordUnit* data_ptr = codes;
    while(num_remain_tuples > 0){
        size_t size = std::min(
                blocks_[block_id]->num_tuples() - pos_in_block,
                num_remain_tuples);
        blocks_[block_id]->BulkLoadArray(data_ptr, size, pos_in_block);
        data_ptr += size;
        num_remain_tuples -= size;
        pos_in_block = 0;
        block_id++;
    }
}


void Column::Scan(Comparator comparator, WordUnit literal,
            BitVector* bitvector, Bitwise bit_opt) const{

    assert(num_tuples_ == bitvector->num());


#pragma omp parallel for schedule(dynamic)
    for(size_t block_id = 0; block_id < blocks_.size(); block_id++){

//    std::cout << "I'm thread" << omp_get_thread_num() << "/" << omp_get_num_threads();
//    std::cout << " Block " << block_id << std::endl;

        blocks_[block_id]->Scan(
                comparator, 
                literal, 
                bitvector->GetBVBlock(block_id), 
                bit_opt);
    }
}
 
void Column::Scan(Comparator comparator, const Column* other_column, 
            BitVector* bitvector, Bitwise bit_opt) const{
    assert(num_tuples_ == bitvector->num());
    assert(type_ == other_column->type());
    assert(bit_width_ == other_column->bit_width());
    assert(num_tuples_ == other_column->num_tuples());

#pragma omp parallel for schedule(dynamic)
    for(size_t block_id = 0; block_id < blocks_.size(); block_id++){
        blocks_[block_id]->Scan(
                comparator,
                other_column->blocks_[block_id],
                bitvector->GetBVBlock(block_id),
                bit_opt);
    }

}

void Column::ScanByte(size_t byte_id, Comparator comparator, ByteUnit literal,
        ByteMaskVector* bm_less, ByteMaskVector* bm_greater, ByteMaskVector* bm_equal, 
        ByteMaskVector* input_mask) const{
    assert(num_tuples_ == bm_less->num());
    assert(num_tuples_ == bm_greater->num());
    assert(num_tuples_ == bm_equal->num());
    assert(num_tuples_ == input_mask->num());
    assert(type_ == ColumnType::kByteSlicePadRight);

#pragma omp parallel for schedule(dynamic)
    for(size_t block_id = 0; block_id < blocks_.size(); block_id++){
        blocks_[block_id]->ScanByte(
                byte_id,
                comparator,
                literal,
                bm_less->GetBMBlock(block_id),
                bm_greater->GetBMBlock(block_id),
                bm_equal->GetBMBlock(block_id),
                input_mask->GetBMBlock(block_id));
    }
}

void Column::ScanByte(size_t byte_id, Comparator comparator, ByteUnit literal,
        ByteMaskVector* bm_less, ByteMaskVector* bm_greater, ByteMaskVector* bm_equal) const{
    assert(num_tuples_ == bm_less->num());
    assert(num_tuples_ == bm_greater->num());
    assert(num_tuples_ == bm_equal->num());
    assert(type_ == ColumnType::kByteSlicePadRight);

#pragma omp parallel for schedule(dynamic)
    for(size_t block_id = 0; block_id < blocks_.size(); block_id++){
        blocks_[block_id]->ScanByte(
                byte_id,
                comparator,
                literal,
                bm_less->GetBMBlock(block_id),
                bm_greater->GetBMBlock(block_id),
                bm_equal->GetBMBlock(block_id));
    }
}

ColumnBlock* Column::CreateNewBlock() const{
    assert(0 < bit_width_ && 32 >= bit_width_);
    if(!(0<bit_width_ && 32>= bit_width_)){
        std::cerr << "Incorrect bit width: " << bit_width_ << std::endl;
    }

    switch(type_){
        case ColumnType::kNaive:
            switch(CEIL(bit_width_, 8)){
                case 1:
                    return new NaiveColumnBlock<uint8_t>();
                case 2:
                    return new NaiveColumnBlock<uint16_t>();
                case 3:
                case 4:
                    return new NaiveColumnBlock<uint32_t>();
            }
            break;
        case ColumnType::kNaiveAvx:
            switch(CEIL(bit_width_, 8)){
                case 1:
                    return new NaiveAvxColumnBlock<uint8_t>();
                case 2:
                    return new NaiveAvxColumnBlock<uint16_t>();
                case 3:
                case 4:
                    return new NaiveAvxColumnBlock<uint32_t>();
            }
            break;
        case ColumnType::kNumpsAvx:
            return new NaiveAvxColumnBlock<uint32_t>();
        case ColumnType::kAvx2Scan:
            switch(bit_width_){
                case 1: return new Avx2ScanColumnBlock<1>();
                case 2: return new Avx2ScanColumnBlock<2>();
                case 3: return new Avx2ScanColumnBlock<3>();
                case 4: return new Avx2ScanColumnBlock<4>();
                case 5: return new Avx2ScanColumnBlock<5>();
                case 6: return new Avx2ScanColumnBlock<6>();
                case 7: return new Avx2ScanColumnBlock<7>();
                case 8: return new Avx2ScanColumnBlock<8>();
                case 9: return new Avx2ScanColumnBlock<9>();
                case 10: return new Avx2ScanColumnBlock<10>();
                case 11: return new Avx2ScanColumnBlock<11>();
                case 12: return new Avx2ScanColumnBlock<12>();
                case 13: return new Avx2ScanColumnBlock<13>();
                case 14: return new Avx2ScanColumnBlock<14>();
                case 15: return new Avx2ScanColumnBlock<15>();
                case 16: return new Avx2ScanColumnBlock<16>();
                case 17: return new Avx2ScanColumnBlock<17>();
                case 18: return new Avx2ScanColumnBlock<18>();
                case 19: return new Avx2ScanColumnBlock<19>();
                case 20: return new Avx2ScanColumnBlock<20>();
                case 21: return new Avx2ScanColumnBlock<21>();
                case 22: return new Avx2ScanColumnBlock<22>();
                case 23: return new Avx2ScanColumnBlock<23>();
                case 24: return new Avx2ScanColumnBlock<24>();
                case 25: return new Avx2ScanColumnBlock<25>();
                case 26: return new Avx2ScanColumnBlock<26>();
                case 27: return new Avx2ScanColumnBlock<27>();
                case 28: return new Avx2ScanColumnBlock<28>();
                case 29: return new Avx2ScanColumnBlock<29>();
                case 30: return new Avx2ScanColumnBlock<30>();
                case 31: return new Avx2ScanColumnBlock<31>();
                case 32: return new Avx2ScanColumnBlock<32>();
            }
            break;
        case ColumnType::kHbp:
            switch(bit_width_){
                case 1: return new HbpColumnBlock<1>();
                case 2: return new HbpColumnBlock<2>();
                case 3: return new HbpColumnBlock<3>();
                case 4: return new HbpColumnBlock<4>();
                case 5: return new HbpColumnBlock<5>();
                case 6: return new HbpColumnBlock<6>();
                case 7: return new HbpColumnBlock<7>();
                case 8: return new HbpColumnBlock<8>();
                case 9: return new HbpColumnBlock<9>();
                case 10: return new HbpColumnBlock<10>();
                case 11: return new HbpColumnBlock<11>();
                case 12: return new HbpColumnBlock<12>();
                case 13: return new HbpColumnBlock<13>();
                case 14: return new HbpColumnBlock<14>();
                case 15: return new HbpColumnBlock<15>();
                case 16: return new HbpColumnBlock<16>();
                case 17: return new HbpColumnBlock<17>();
                case 18: return new HbpColumnBlock<18>();
                case 19: return new HbpColumnBlock<19>();
                case 20: return new HbpColumnBlock<20>();
                case 21: return new HbpColumnBlock<21>();
                case 22: return new HbpColumnBlock<22>();
                case 23: return new HbpColumnBlock<23>();
                case 24: return new HbpColumnBlock<24>();
                case 25: return new HbpColumnBlock<25>();
                case 26: return new HbpColumnBlock<26>();
                case 27: return new HbpColumnBlock<27>();
                case 28: return new HbpColumnBlock<28>();
                case 29: return new HbpColumnBlock<29>();
                case 30: return new HbpColumnBlock<30>();
                case 31: return new HbpColumnBlock<31>();
                case 32: return new HbpColumnBlock<32>();
            }
            break;
        case ColumnType::kVbp:
            switch(bit_width_){
                case 1: return new VbpColumnBlock<1>();
                case 2: return new VbpColumnBlock<2>();
                case 3: return new VbpColumnBlock<3>();
                case 4: return new VbpColumnBlock<4>();
                case 5: return new VbpColumnBlock<5>();
                case 6: return new VbpColumnBlock<6>();
                case 7: return new VbpColumnBlock<7>();
                case 8: return new VbpColumnBlock<8>();
                case 9: return new VbpColumnBlock<9>();
                case 10: return new VbpColumnBlock<10>();
                case 11: return new VbpColumnBlock<11>();
                case 12: return new VbpColumnBlock<12>();
                case 13: return new VbpColumnBlock<13>();
                case 14: return new VbpColumnBlock<14>();
                case 15: return new VbpColumnBlock<15>();
                case 16: return new VbpColumnBlock<16>();
                case 17: return new VbpColumnBlock<17>();
                case 18: return new VbpColumnBlock<18>();
                case 19: return new VbpColumnBlock<19>();
                case 20: return new VbpColumnBlock<20>();
                case 21: return new VbpColumnBlock<21>();
                case 22: return new VbpColumnBlock<22>();
                case 23: return new VbpColumnBlock<23>();
                case 24: return new VbpColumnBlock<24>();
                case 25: return new VbpColumnBlock<25>();
                case 26: return new VbpColumnBlock<26>();
                case 27: return new VbpColumnBlock<27>();
                case 28: return new VbpColumnBlock<28>();
                case 29: return new VbpColumnBlock<29>();
                case 30: return new VbpColumnBlock<30>();
                case 31: return new VbpColumnBlock<31>();
                case 32: return new VbpColumnBlock<32>();
            }
            break;
        case ColumnType::kBitSlice:
            return new BitSliceColumnBlock(bit_width_);
            break;
        case ColumnType::kByteSlicePadRight:
            switch(bit_width_){
                case 1: return new ByteSliceColumnBlock<1>();
                case 2: return new ByteSliceColumnBlock<2>();
                case 3: return new ByteSliceColumnBlock<3>();
                case 4: return new ByteSliceColumnBlock<4>();
                case 5: return new ByteSliceColumnBlock<5>();
                case 6: return new ByteSliceColumnBlock<6>();
                case 7: return new ByteSliceColumnBlock<7>();
                case 8: return new ByteSliceColumnBlock<8>();
                case 9: return new ByteSliceColumnBlock<9>();
                case 10: return new ByteSliceColumnBlock<10>();
                case 11: return new ByteSliceColumnBlock<11>();
                case 12: return new ByteSliceColumnBlock<12>();
                case 13: return new ByteSliceColumnBlock<13>();
                case 14: return new ByteSliceColumnBlock<14>();
                case 15: return new ByteSliceColumnBlock<15>();
                case 16: return new ByteSliceColumnBlock<16>();
                case 17: return new ByteSliceColumnBlock<17>();
                case 18: return new ByteSliceColumnBlock<18>();
                case 19: return new ByteSliceColumnBlock<19>();
                case 20: return new ByteSliceColumnBlock<20>();
                case 21: return new ByteSliceColumnBlock<21>();
                case 22: return new ByteSliceColumnBlock<22>();
                case 23: return new ByteSliceColumnBlock<23>();
                case 24: return new ByteSliceColumnBlock<24>();
                case 25: return new ByteSliceColumnBlock<25>();
                case 26: return new ByteSliceColumnBlock<26>();
                case 27: return new ByteSliceColumnBlock<27>();
                case 28: return new ByteSliceColumnBlock<28>();
                case 29: return new ByteSliceColumnBlock<29>();
                case 30: return new ByteSliceColumnBlock<30>();
                case 31: return new ByteSliceColumnBlock<31>();
                case 32: return new ByteSliceColumnBlock<32>();
            }
            break;
        case ColumnType::kSuperscalar2ByteSlicePadRight:
            switch(bit_width_){
                case 1: return new Superscalar2ByteSliceColumnBlock<1>();
                case 2: return new Superscalar2ByteSliceColumnBlock<2>();
                case 3: return new Superscalar2ByteSliceColumnBlock<3>();
                case 4: return new Superscalar2ByteSliceColumnBlock<4>();
                case 5: return new Superscalar2ByteSliceColumnBlock<5>();
                case 6: return new Superscalar2ByteSliceColumnBlock<6>();
                case 7: return new Superscalar2ByteSliceColumnBlock<7>();
                case 8: return new Superscalar2ByteSliceColumnBlock<8>();
                case 9: return new Superscalar2ByteSliceColumnBlock<9>();
                case 10: return new Superscalar2ByteSliceColumnBlock<10>();
                case 11: return new Superscalar2ByteSliceColumnBlock<11>();
                case 12: return new Superscalar2ByteSliceColumnBlock<12>();
                case 13: return new Superscalar2ByteSliceColumnBlock<13>();
                case 14: return new Superscalar2ByteSliceColumnBlock<14>();
                case 15: return new Superscalar2ByteSliceColumnBlock<15>();
                case 16: return new Superscalar2ByteSliceColumnBlock<16>();
                case 17: return new Superscalar2ByteSliceColumnBlock<17>();
                case 18: return new Superscalar2ByteSliceColumnBlock<18>();
                case 19: return new Superscalar2ByteSliceColumnBlock<19>();
                case 20: return new Superscalar2ByteSliceColumnBlock<20>();
                case 21: return new Superscalar2ByteSliceColumnBlock<21>();
                case 22: return new Superscalar2ByteSliceColumnBlock<22>();
                case 23: return new Superscalar2ByteSliceColumnBlock<23>();
                case 24: return new Superscalar2ByteSliceColumnBlock<24>();
                case 25: return new Superscalar2ByteSliceColumnBlock<25>();
                case 26: return new Superscalar2ByteSliceColumnBlock<26>();
                case 27: return new Superscalar2ByteSliceColumnBlock<27>();
                case 28: return new Superscalar2ByteSliceColumnBlock<28>();
                case 29: return new Superscalar2ByteSliceColumnBlock<29>();
                case 30: return new Superscalar2ByteSliceColumnBlock<30>();
                case 31: return new Superscalar2ByteSliceColumnBlock<31>();
                case 32: return new Superscalar2ByteSliceColumnBlock<32>();
            }
            break;
        case ColumnType::kSuperscalar4ByteSlicePadRight:
            switch(bit_width_){
                case 1: return new Superscalar4ByteSliceColumnBlock<1>();
                case 2: return new Superscalar4ByteSliceColumnBlock<2>();
                case 3: return new Superscalar4ByteSliceColumnBlock<3>();
                case 4: return new Superscalar4ByteSliceColumnBlock<4>();
                case 5: return new Superscalar4ByteSliceColumnBlock<5>();
                case 6: return new Superscalar4ByteSliceColumnBlock<6>();
                case 7: return new Superscalar4ByteSliceColumnBlock<7>();
                case 8: return new Superscalar4ByteSliceColumnBlock<8>();
                case 9: return new Superscalar4ByteSliceColumnBlock<9>();
                case 10: return new Superscalar4ByteSliceColumnBlock<10>();
                case 11: return new Superscalar4ByteSliceColumnBlock<11>();
                case 12: return new Superscalar4ByteSliceColumnBlock<12>();
                case 13: return new Superscalar4ByteSliceColumnBlock<13>();
                case 14: return new Superscalar4ByteSliceColumnBlock<14>();
                case 15: return new Superscalar4ByteSliceColumnBlock<15>();
                case 16: return new Superscalar4ByteSliceColumnBlock<16>();
                case 17: return new Superscalar4ByteSliceColumnBlock<17>();
                case 18: return new Superscalar4ByteSliceColumnBlock<18>();
                case 19: return new Superscalar4ByteSliceColumnBlock<19>();
                case 20: return new Superscalar4ByteSliceColumnBlock<20>();
                case 21: return new Superscalar4ByteSliceColumnBlock<21>();
                case 22: return new Superscalar4ByteSliceColumnBlock<22>();
                case 23: return new Superscalar4ByteSliceColumnBlock<23>();
                case 24: return new Superscalar4ByteSliceColumnBlock<24>();
                case 25: return new Superscalar4ByteSliceColumnBlock<25>();
                case 26: return new Superscalar4ByteSliceColumnBlock<26>();
                case 27: return new Superscalar4ByteSliceColumnBlock<27>();
                case 28: return new Superscalar4ByteSliceColumnBlock<28>();
                case 29: return new Superscalar4ByteSliceColumnBlock<29>();
                case 30: return new Superscalar4ByteSliceColumnBlock<30>();
                case 31: return new Superscalar4ByteSliceColumnBlock<31>();
                case 32: return new Superscalar4ByteSliceColumnBlock<32>();
            }
            break;
        case ColumnType::kHybridSlice:
            switch(bit_width_){
                case 1: return new HybridSliceColumnBlock<1>();
                case 2: return new HybridSliceColumnBlock<2>();
                case 3: return new HybridSliceColumnBlock<3>();
                case 4: return new HybridSliceColumnBlock<4>();
                case 5: return new HybridSliceColumnBlock<5>();
                case 6: return new HybridSliceColumnBlock<6>();
                case 7: return new HybridSliceColumnBlock<7>();
                case 8: return new HybridSliceColumnBlock<8>();
                case 9: return new HybridSliceColumnBlock<9>();
                case 10: return new HybridSliceColumnBlock<10>();
                case 11: return new HybridSliceColumnBlock<11>();
                case 12: return new HybridSliceColumnBlock<12>();
                case 13: return new HybridSliceColumnBlock<13>();
                case 14: return new HybridSliceColumnBlock<14>();
                case 15: return new HybridSliceColumnBlock<15>();
                case 16: return new HybridSliceColumnBlock<16>();
                case 17: return new HybridSliceColumnBlock<17>();
                case 18: return new HybridSliceColumnBlock<18>();
                case 19: return new HybridSliceColumnBlock<19>();
                case 20: return new HybridSliceColumnBlock<20>();
                case 21: return new HybridSliceColumnBlock<21>();
                case 22: return new HybridSliceColumnBlock<22>();
                case 23: return new HybridSliceColumnBlock<23>();
                case 24: return new HybridSliceColumnBlock<24>();
                case 25: return new HybridSliceColumnBlock<25>();
                case 26: return new HybridSliceColumnBlock<26>();
                case 27: return new HybridSliceColumnBlock<27>();
                case 28: return new HybridSliceColumnBlock<28>();
                case 29: return new HybridSliceColumnBlock<29>();
                case 30: return new HybridSliceColumnBlock<30>();
                case 31: return new HybridSliceColumnBlock<31>();
                case 32: return new HybridSliceColumnBlock<32>();
            }
            break;
        case ColumnType::kSmartHybridSlice:
            if(/*bit_width_ <= 6 ||*/ (bit_width_ % 8 < 4 && bit_width_ % 8 > 0)){
                //use hybrid-slice
                switch(bit_width_){
                    case 1: return new HybridSliceColumnBlock<1>();
                    case 2: return new HybridSliceColumnBlock<2>();
                    case 3: return new HybridSliceColumnBlock<3>();
                    case 4: return new HybridSliceColumnBlock<4>();
                    case 5: return new HybridSliceColumnBlock<5>();
                    case 6: return new HybridSliceColumnBlock<6>();
                    case 7: return new HybridSliceColumnBlock<7>();
                    case 8: return new HybridSliceColumnBlock<8>();
                    case 9: return new HybridSliceColumnBlock<9>();
                    case 10: return new HybridSliceColumnBlock<10>();
                    case 11: return new HybridSliceColumnBlock<11>();
                    case 12: return new HybridSliceColumnBlock<12>();
                    case 13: return new HybridSliceColumnBlock<13>();
                    case 14: return new HybridSliceColumnBlock<14>();
                    case 15: return new HybridSliceColumnBlock<15>();
                    case 16: return new HybridSliceColumnBlock<16>();
                    case 17: return new HybridSliceColumnBlock<17>();
                    case 18: return new HybridSliceColumnBlock<18>();
                    case 19: return new HybridSliceColumnBlock<19>();
                    case 20: return new HybridSliceColumnBlock<20>();
                    case 21: return new HybridSliceColumnBlock<21>();
                    case 22: return new HybridSliceColumnBlock<22>();
                    case 23: return new HybridSliceColumnBlock<23>();
                    case 24: return new HybridSliceColumnBlock<24>();
                    case 25: return new HybridSliceColumnBlock<25>();
                    case 26: return new HybridSliceColumnBlock<26>();
                    case 27: return new HybridSliceColumnBlock<27>();
                    case 28: return new HybridSliceColumnBlock<28>();
                    case 29: return new HybridSliceColumnBlock<29>();
                    case 30: return new HybridSliceColumnBlock<30>();
                    case 31: return new HybridSliceColumnBlock<31>();
                    case 32: return new HybridSliceColumnBlock<32>();
                }
            }
            else{
                //use pure byte-slice
                switch(bit_width_){
                    case 1: return new ByteSliceColumnBlock<1>();
                    case 2: return new ByteSliceColumnBlock<2>();
                    case 3: return new ByteSliceColumnBlock<3>();
                    case 4: return new ByteSliceColumnBlock<4>();
                    case 5: return new ByteSliceColumnBlock<5>();
                    case 6: return new ByteSliceColumnBlock<6>();
                    case 7: return new ByteSliceColumnBlock<7>();
                    case 8: return new ByteSliceColumnBlock<8>();
                    case 9: return new ByteSliceColumnBlock<9>();
                    case 10: return new ByteSliceColumnBlock<10>();
                    case 11: return new ByteSliceColumnBlock<11>();
                    case 12: return new ByteSliceColumnBlock<12>();
                    case 13: return new ByteSliceColumnBlock<13>();
                    case 14: return new ByteSliceColumnBlock<14>();
                    case 15: return new ByteSliceColumnBlock<15>();
                    case 16: return new ByteSliceColumnBlock<16>();
                    case 17: return new ByteSliceColumnBlock<17>();
                    case 18: return new ByteSliceColumnBlock<18>();
                    case 19: return new ByteSliceColumnBlock<19>();
                    case 20: return new ByteSliceColumnBlock<20>();
                    case 21: return new ByteSliceColumnBlock<21>();
                    case 22: return new ByteSliceColumnBlock<22>();
                    case 23: return new ByteSliceColumnBlock<23>();
                    case 24: return new ByteSliceColumnBlock<24>();
                    case 25: return new ByteSliceColumnBlock<25>();
                    case 26: return new ByteSliceColumnBlock<26>();
                    case 27: return new ByteSliceColumnBlock<27>();
                    case 28: return new ByteSliceColumnBlock<28>();
                    case 29: return new ByteSliceColumnBlock<29>();
                    case 30: return new ByteSliceColumnBlock<30>();
                    case 31: return new ByteSliceColumnBlock<31>();
                    case 32: return new ByteSliceColumnBlock<32>();
                }
            }
            break;
        case ColumnType::kDualByteSlicePadRight:
            switch(bit_width_){
                case 1: return new DualByteSliceColumnBlock<1>();
                case 2: return new DualByteSliceColumnBlock<2>();
                case 3: return new DualByteSliceColumnBlock<3>();
                case 4: return new DualByteSliceColumnBlock<4>();
                case 5: return new DualByteSliceColumnBlock<5>();
                case 6: return new DualByteSliceColumnBlock<6>();
                case 7: return new DualByteSliceColumnBlock<7>();
                case 8: return new DualByteSliceColumnBlock<8>();
                case 9: return new DualByteSliceColumnBlock<9>();
                case 10: return new DualByteSliceColumnBlock<10>();
                case 11: return new DualByteSliceColumnBlock<11>();
                case 12: return new DualByteSliceColumnBlock<12>();
                case 13: return new DualByteSliceColumnBlock<13>();
                case 14: return new DualByteSliceColumnBlock<14>();
                case 15: return new DualByteSliceColumnBlock<15>();
                case 16: return new DualByteSliceColumnBlock<16>();
                case 17: return new DualByteSliceColumnBlock<17>();
                case 18: return new DualByteSliceColumnBlock<18>();
                case 19: return new DualByteSliceColumnBlock<19>();
                case 20: return new DualByteSliceColumnBlock<20>();
                case 21: return new DualByteSliceColumnBlock<21>();
                case 22: return new DualByteSliceColumnBlock<22>();
                case 23: return new DualByteSliceColumnBlock<23>();
                case 24: return new DualByteSliceColumnBlock<24>();
                case 25: return new DualByteSliceColumnBlock<25>();
                case 26: return new DualByteSliceColumnBlock<26>();
                case 27: return new DualByteSliceColumnBlock<27>();
                case 28: return new DualByteSliceColumnBlock<28>();
                case 29: return new DualByteSliceColumnBlock<29>();
                case 30: return new DualByteSliceColumnBlock<30>();
                case 31: return new DualByteSliceColumnBlock<31>();
                case 32: return new DualByteSliceColumnBlock<32>();
            }
            break;

        default:
            std::cerr << "Unknown column type." << std::endl;
            exit(1);
    }
}

}   //namespace
