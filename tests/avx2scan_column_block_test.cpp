#include    "include/avx2scan_column_block.h"
#include    <cstdlib>
#include    "gtest/gtest.h"
#include    "include/bitvector_block.h"

#define CODEWIDTH 13

namespace byteslice{

class Avx2ScanColumnBlockTest: public ::testing::Test{
public:
    virtual void SetUp(){
        std::srand(std::time(0));
        block_ = new Avx2ScanColumnBlock<CODEWIDTH>(num_);
    }

    virtual void TearDown(){
        delete block_;
    }

protected:
    Avx2ScanColumnBlock<CODEWIDTH>* block_;
    size_t num_ = kNumTuplesPerBlock * 0.54;
    WordUnit mask_ = (1ULL << CODEWIDTH) - 1;

};


TEST_F(Avx2ScanColumnBlockTest, GetSetTuple){
    WordUnit* codes = new WordUnit[num_];

    for(size_t i=0; i < num_; i++){
        codes[i] = std::rand() & mask_;
    }

    block_->BulkLoadArray(codes, num_);

    //Verify
    for(size_t i=0; i < num_; i++){
        EXPECT_EQ(codes[i], block_->GetTuple(i));
    }

    delete codes;
}

TEST_F(Avx2ScanColumnBlockTest, ScanLiteral){
    WordUnit* codes = new WordUnit[num_];
    BitVectorBlock* bvblock = new BitVectorBlock(num_);

    for(size_t i=0; i < num_; i++){
        codes[i] = std::rand() & mask_;
        //codes[i] = i & mask_;
    }

    block_->BulkLoadArray(codes, num_);

    WordUnit literal = std::rand() & mask_;
    //WordUnit literal = 8;
    block_->Scan(Comparator::kLess, literal, bvblock, Bitwise::kSet);

    //Verify
    size_t count = 0;
    for(size_t i=0; i < num_; i++){
        EXPECT_EQ(bvblock->GetBit(i), codes[i] < literal);
        count += (codes[i] < literal);
    }
    EXPECT_EQ(count, bvblock->CountOnes());

    //And
    WordUnit literal2 = literal / 2;

    block_->Scan(Comparator::kGreater, literal2, bvblock, Bitwise::kAnd);

    //Verify
    count = 0;
    for(size_t i=0; i < num_; i++){
        EXPECT_EQ(bvblock->GetBit(i), codes[i] < literal && codes[i] > literal2);
        count += (codes[i] < literal && codes[i] > literal2);
    }
    EXPECT_EQ(count, bvblock->CountOnes());

    delete codes;
    delete bvblock;   
}

TEST_F(Avx2ScanColumnBlockTest, ScanOtherBlock){
    Avx2ScanColumnBlock<CODEWIDTH>* block2 = new Avx2ScanColumnBlock<CODEWIDTH>(num_);
    BitVectorBlock* bvblock = new BitVectorBlock(num_);

    for(size_t i=0; i < num_; i++){
        block_->SetTuple(i, std::rand() & mask_);
        block2->SetTuple(i, std::rand() & mask_);
    }

    block_->Scan(Comparator::kLess, block2, bvblock, Bitwise::kSet);

    //Verify
    for(size_t i=0; i < num_; i++){
        EXPECT_EQ(bvblock->GetBit(i), block_->GetTuple(i) < block2->GetTuple(i));
    }

    delete bvblock;
    delete block2;
}



}   //namespace
