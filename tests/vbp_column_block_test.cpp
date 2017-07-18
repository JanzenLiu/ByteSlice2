#include    "include/vbp_column_block.h"
#include    <cstdlib>
#include    "gtest/gtest.h"
#include    "include/bitvector_block.h"
#include    <iostream>

namespace byteslice{

class VbpColumnBlockTest: public ::testing::Test{
public:
    virtual void SetUp(){
        std::srand(std::time(0));
        codes_ = new WordUnit[num_];
        for(size_t i=0; i < num_; i++){
            codes_[i] = std::rand() & mask_;
            //codes_[i] = 511;
        }

        block_ = new VbpColumnBlock<12>(num_);
        block_->BulkLoadArray(codes_, num_);
    }

    virtual void TearDown(){
        delete [] codes_;
        delete block_;
    }

protected:
    VbpColumnBlock<12>* block_;
    WordUnit mask_ = (1ULL << 12) - 1;
    size_t num_ = 0.64*kNumTuplesPerBlock;
    WordUnit* codes_;
};

TEST_F(VbpColumnBlockTest, BulkLoadAndGetTuple){
    for(size_t i=0; i < num_; i++){
        EXPECT_EQ(codes_[i], block_->GetTuple(i));
    }
}

TEST_F(VbpColumnBlockTest, ScanLiteral){
    BitVectorBlock* bvblock = new BitVectorBlock(num_);

    WordUnit literal = std::rand() & mask_;
    block_->Scan(Comparator::kLess, literal, bvblock, Bitwise::kSet);

    //Verify
    for(size_t i=0; i < num_; i++){
        EXPECT_EQ(bvblock->GetBit(i), (codes_[i] < literal));
    }

    WordUnit literal2 = literal / 2;
    block_->Scan(Comparator::kGreater, literal2, bvblock, Bitwise::kAnd);

    //Verify
    for(size_t i=0; i < num_; i++){
        EXPECT_EQ(bvblock->GetBit(i), (codes_[i] < literal && codes_[i] > literal2));
    }

    delete bvblock;
}

TEST_F(VbpColumnBlockTest, ScanOtherBlock){
    VbpColumnBlock<12>* block2 = new VbpColumnBlock<12>(num_);
    BitVectorBlock* bvblock = new BitVectorBlock(num_);
    WordUnit* codes2 = new WordUnit[num_];

    for(size_t i = 0; i < num_; i++){
        codes2[i] = std::rand() & mask_;
    }
    block2->BulkLoadArray(codes2, num_);

    block_->Scan(Comparator::kGreaterEqual, block2, bvblock, Bitwise::kSet);

    //Verify
    for(size_t i=0; i < num_; i++){
        EXPECT_EQ(bvblock->GetBit(i), codes_[i] >= codes2[i]);
    }

    delete [] codes2;
    delete bvblock;
    delete block2;
}


}   //namespace
