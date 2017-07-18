#include    "include/hybridslice_column_block.h"
#include    <cstdlib>
#include    "gtest/gtest.h"
#include    "include/bitvector_block.h"

namespace byteslice{

class HybridSliceColumnBlockTest: public ::testing::Test{
public:
    virtual void SetUp(){
        std::srand(std::time(0));

    }

    virtual void TearDown(){

    }

protected:
    ColumnBlock* block_;
    size_t num_ = 0.8*kNumTuplesPerBlock;

};

TEST_F(HybridSliceColumnBlockTest, BulkloadAndGetTuple){
    HybridSliceColumnBlock<18>* block = new HybridSliceColumnBlock<18>(num_);
    WordUnit *codes = new WordUnit[num_];
    WordUnit mask = (1ULL << 18) - 1;
    //generate random tuple
    for(size_t i=0; i < num_; i++){
        codes[i] = std::rand() & mask;
    }
    
    block->BulkLoadArray(codes, num_);

    //Verify
    for(size_t i=0; i < num_; i++){
        EXPECT_EQ(codes[i], block->GetTuple(i));
    }

    delete [] codes;
    delete block;
}

TEST_F(HybridSliceColumnBlockTest, ScanLiteral16){
    HybridSliceColumnBlock<16>* block = new HybridSliceColumnBlock<16>(num_);
    BitVectorBlock* bvblock = new BitVectorBlock(num_);

//    EXPECT_EQ(2, block->GetNumByteSlices());
//    EXPECT_EQ(0, block->GetNumBitSlices());

    WordUnit mask = (1ULL << 16) - 1;
    for(size_t i=0; i < num_; i++){
        block->SetTuple(i, std::rand() & mask);
    }

    WordUnit literal = std::rand() & mask;
    block->Scan(Comparator::kLess, literal, bvblock, Bitwise::kSet);

    //Verify
    size_t count = 0;
    for(size_t i=0; i < num_; i++){
        EXPECT_EQ(bvblock->GetBit(i), block->GetTuple(i) < literal);
        count += (block->GetTuple(i) < literal);
    }
    //std::cout << "Count=" << count << std::endl;
    //std::cout << "Actual=" << bvblock->CountOnes() << std::endl;
    EXPECT_EQ(count, bvblock->CountOnes());


    //AND
    WordUnit literal2 = literal / 2;
    block->Scan(Comparator::kGreater, literal2, bvblock, Bitwise::kAnd);

    //Verify
    for(size_t i=0; i < num_; i++){
        WordUnit val = block->GetTuple(i);
        EXPECT_EQ(bvblock->GetBit(i), (val < literal)&&(val > literal2));
    }

    delete bvblock;
    delete block;
}
TEST_F(HybridSliceColumnBlockTest, ScanLiteral18){
    HybridSliceColumnBlock<18>* block = new HybridSliceColumnBlock<18>(num_);
    BitVectorBlock* bvblock = new BitVectorBlock(num_);

//    EXPECT_EQ(2, block->GetNumByteSlices());
//    EXPECT_EQ(2, block->GetNumBitSlices());

    WordUnit mask = (1ULL << 18) - 1;
    for(size_t i=0; i < num_; i++){
        block->SetTuple(i, std::rand() & mask);
    }

    WordUnit literal = std::rand() & mask;
    block->Scan(Comparator::kLess, literal, bvblock, Bitwise::kSet);

    //Verify
    size_t count = 0;
    for(size_t i=0; i < num_; i++){
        EXPECT_EQ(bvblock->GetBit(i), block->GetTuple(i) < literal);
        count += (block->GetTuple(i) < literal);
    }
//    std::cout << "Count=" << count << std::endl;
//    std::cout << "Actual=" << bvblock->CountOnes() << std::endl;
    EXPECT_EQ(count, bvblock->CountOnes());


    //AND
    WordUnit literal2 = literal / 2;
    block->Scan(Comparator::kGreater, literal2, bvblock, Bitwise::kAnd);

    //Verify
    for(size_t i=0; i < num_; i++){
        WordUnit val = block->GetTuple(i);
        EXPECT_EQ(bvblock->GetBit(i), (val < literal)&&(val > literal2));
    }

    delete bvblock;
    delete block;
}

TEST_F(HybridSliceColumnBlockTest, ScanLiteral22){
    HybridSliceColumnBlock<22>* block = new HybridSliceColumnBlock<22>(num_);
    BitVectorBlock* bvblock = new BitVectorBlock(num_);

//    EXPECT_EQ(3, block->GetNumByteSlices());
//    EXPECT_EQ(0, block->GetNumBitSlices());

    WordUnit mask = (1ULL << 22) - 1;
    for(size_t i=0; i < num_; i++){
        block->SetTuple(i, std::rand() & mask);
    }

    WordUnit literal = std::rand() & mask;
    block->Scan(Comparator::kLess, literal, bvblock, Bitwise::kSet);

    //Verify
    size_t count = 0;
    for(size_t i=0; i < num_; i++){
        EXPECT_EQ(bvblock->GetBit(i), block->GetTuple(i) < literal);
        count += (block->GetTuple(i) < literal);
    }
//    std::cout << "Count=" << count << std::endl;
//    std::cout << "Actual=" << bvblock->CountOnes() << std::endl;
    EXPECT_EQ(count, bvblock->CountOnes());

    //AND
    WordUnit literal2 = literal / 2;
    block->Scan(Comparator::kGreater, literal2, bvblock, Bitwise::kAnd);

    //Verify
    for(size_t i=0; i < num_; i++){
        WordUnit val = block->GetTuple(i);
        EXPECT_EQ(bvblock->GetBit(i), (val < literal)&&(val > literal2));
    }

    delete bvblock;
    delete block;
}

TEST_F(HybridSliceColumnBlockTest, ScanOtherBlock21){
    HybridSliceColumnBlock<21>* block = new HybridSliceColumnBlock<21>(num_);
    HybridSliceColumnBlock<21>* block2 = new HybridSliceColumnBlock<21>(num_);
    BitVectorBlock* bvblock = new BitVectorBlock(num_);

    WordUnit mask = (1ULL << 21) - 1;
    for(size_t i=0; i < num_; i++){
        block->SetTuple(i, std::rand() & mask);
        block2->SetTuple(i, std::rand() & mask);
    }

    block->Scan(Comparator::kLess, block2, bvblock, Bitwise::kSet);

    //verify
    size_t count = 0;
    for(size_t i=0; i < num_; i++){
        EXPECT_EQ(bvblock->GetBit(i), block->GetTuple(i) < block2->GetTuple(i));
        count += (block->GetTuple(i) < block2->GetTuple(i));
    }

    EXPECT_EQ(count, bvblock->CountOnes());

    delete bvblock;
    delete block2;
    delete block;

}

TEST_F(HybridSliceColumnBlockTest, ScanOtherBlock16){
    HybridSliceColumnBlock<16>* block = new HybridSliceColumnBlock<16>(num_);
    HybridSliceColumnBlock<16>* block2 = new HybridSliceColumnBlock<16>(num_);
    BitVectorBlock* bvblock = new BitVectorBlock(num_);

    WordUnit mask = (1ULL << 16) - 1;
    for(size_t i=0; i < num_; i++){
        block->SetTuple(i, std::rand() & mask);
        block2->SetTuple(i, std::rand() & mask);
    }

    block->Scan(Comparator::kLess, block2, bvblock, Bitwise::kSet);

    //verify
    size_t count = 0;
    for(size_t i=0; i < num_; i++){
        EXPECT_EQ(bvblock->GetBit(i), block->GetTuple(i) < block2->GetTuple(i));
        count += (block->GetTuple(i) < block2->GetTuple(i));
    }

    EXPECT_EQ(count, bvblock->CountOnes());

    delete bvblock;
    delete block2;
    delete block;

}

}   //namespace
