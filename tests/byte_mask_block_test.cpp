#include    "include/byte_mask_block.h"
#include    "gtest/gtest.h"
#include    <cstdlib>

namespace byteslice{

class ByteMaskBlockTest: public ::testing::Test{
public:
    virtual void SetUp(){
        std::srand(std::time(0));
        block_ = new ByteMaskBlock(num_);
        //populate the block with randome truth value
        for(size_t i=0; i < num_; i++){
            block_->SetByteMask(i, (0 == std::rand() % 2));
        }
    }

    virtual void TearDown(){
        delete block_;
    }

protected:
    ByteMaskBlock* block_;
    size_t num_ = kNumTuplesPerBlock*0.7;

};


TEST_F(ByteMaskBlockTest, Condense){
    BitVectorBlock* bvblk = new BitVectorBlock(num_);

    block_->Condense(bvblk);

    //Verify
    size_t count = 0;
    for(size_t i=0; i < num_; i++){
        EXPECT_EQ(block_->GetByteMask(i), bvblk->GetBit(i));
        count += block_->GetByteMask(i);
    }

    EXPECT_EQ(count, bvblk->CountOnes());
    delete bvblk;
}

}   //namespace
