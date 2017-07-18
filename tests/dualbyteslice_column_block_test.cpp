#include    "include/dualbyteslice_column_block.h"
#include    <cstdlib>
#include    "gtest/gtest.h"
#include    <algorithm>
#include    "include/bitvector_block.h"

namespace byteslice{
    
constexpr size_t kBitWidth = 20;

class DualByteSliceColumnBlockTest: public ::testing::Test{
public:
    virtual void SetUp(){
        std::srand(std::time(0));
        num_ = kNumTuplesPerBlock*0.8;
        block_ = new DualByteSliceColumnBlock<kBitWidth>(num_);

        codes_ = new WordUnit[num_];
        for(size_t i=0; i < num_; i++){
            codes_[i] = std::rand() & mask_;
        }
        block_->BulkLoadArray(codes_, num_);
    }

    virtual void TearDown(){
        delete block_;
        delete [] codes_;
    }

protected:
    const WordUnit mask_ = (1ULL << kBitWidth) - 1;
    WordUnit* codes_;
    DualByteSliceColumnBlock<kBitWidth>* block_;
    size_t num_;
};


TEST_F(DualByteSliceColumnBlockTest, BulkLoadAndGetTuple){
    for(size_t i=0; i<num_; i++){
        EXPECT_EQ(codes_[i], block_->GetTuple(i));
    }
}


TEST_F(DualByteSliceColumnBlockTest, ScanLiteral){
    BitVectorBlock* bvblock = new BitVectorBlock(num_);

    std::srand(std::time(0));
    WordUnit lit = std::rand() & mask_;
    block_->Scan(Comparator::kLess, lit, bvblock, Bitwise::kSet);
    auto pred = [lit](WordUnit x){return x < lit;};
    EXPECT_EQ(std::count_if(codes_, codes_+num_, pred),
              bvblock->CountOnes());

    delete bvblock;
}

TEST_F(DualByteSliceColumnBlockTest, ScanOtherBlock){
    BitVectorBlock* bvblock = new BitVectorBlock(num_);
    DualByteSliceColumnBlock<kBitWidth>* block2 = 
        new DualByteSliceColumnBlock<kBitWidth>(num_);

    std::srand(std::time(0));
    for(size_t i=0; i < num_; i++){
        block2->SetTuple(i, std::rand() & mask_);
    }

    block_->Scan(Comparator::kGreaterEqual, block2, bvblock, Bitwise::kSet);
    for(size_t i=0; i < num_; i++){
        EXPECT_EQ(bvblock->GetBit(i), block_->GetTuple(i) >= block2->GetTuple(i));
    }

    delete block2;
    delete bvblock;
}

}   //namespace
