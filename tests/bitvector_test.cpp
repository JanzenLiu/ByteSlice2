#include    "include/common.h"
#include    "include/types.h"
#include    "include/bitvector.h"
#include    "gtest/gtest.h"

namespace byteslice{

class BitVectorTest: public ::testing::Test{
public:
    virtual void SetUp(){
    }

    virtual void TearDown(){
    }

protected:
    const size_t num_ = 3*kNumTuplesPerBlock + 2000;

};

TEST_F(BitVectorTest, Ctor){
    BitVector *bitvector = new BitVector(num_);

    EXPECT_EQ(4, bitvector->GetNumBlocks());
    EXPECT_EQ(num_, bitvector->num());
    EXPECT_EQ(kNumTuplesPerBlock, bitvector->GetBVBlock(0)->num());
    EXPECT_EQ(2000, bitvector->GetBVBlock(3)->num());

    delete bitvector;
}


}   //namespace
