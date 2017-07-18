#include    "include/hbp_column_block.h"
#include    <cstdlib>
#include    "gtest/gtest.h"
#include    "include/bitvector_block.h"
#include    <iostream>

namespace byteslice{

class HbpColumnBlockTest: public ::testing::Test{
public:
    virtual void SetUp(){
        num_ = 2000;
        block_ = new HbpColumnBlock<12>(num_);

        WordUnit* codes = new WordUnit[num_];
        for(size_t i=0; i < num_; i++){
            codes[i] = i;
        }

        block_->BulkLoadArray(codes, num_);
        delete[] codes;
    }

    virtual void TearDown(){
        delete block_;
    }

protected:
    ColumnBlock* block_;
    size_t num_;

};

TEST_F(HbpColumnBlockTest, BulkLoadAndGetTuple){
    for(size_t i=0; i<num_; i++){
        EXPECT_EQ(i, block_->GetTuple(i));
    }
}

TEST_F(HbpColumnBlockTest, DISABLED_SerDeser){
    std::string filename = std::string("temp/test.dat");
    //Serialize this block
    SequentialWriteBinaryFile outfile;
    outfile.Open(filename);
    block_->SerToFile(outfile);
    outfile.Close();

    //Deserialize from file
    ColumnBlock* block2 = new HbpColumnBlock<12>(num_);
    SequentialReadBinaryFile infile;
    infile.Open(filename);
    block2->DeserFromFile(infile);
    infile.Close();

    //Verify
    for(size_t i=0; i<num_; i++){
        EXPECT_EQ(block_->GetTuple(i), block2->GetTuple(i));
    }

    delete block2;
}


TEST_F(HbpColumnBlockTest, ScanOtherBlock){
    HbpColumnBlock<12>* block2 = new HbpColumnBlock<12>(num_);
    BitVectorBlock *bvblock = new BitVectorBlock(num_);

    std::srand(std::time(0));
    WordUnit mask = (1ULL << 12) - 1;
    WordUnit* codes = new WordUnit[num_];
    for(size_t i=0; i < num_; i++){
        codes[i] = std::rand() & mask;
    }
    block2->BulkLoadArray(codes, num_);

    block_->Scan(Comparator::kLess, block2, bvblock, Bitwise::kSet);

    for(size_t i=0; i < num_; i++){
        EXPECT_EQ(bvblock->GetBit(i), block_->GetTuple(i) < codes[i]);
    }

    delete[] codes;
    delete bvblock;
    delete block2;
}

TEST_F(HbpColumnBlockTest, ScanLiteral){
    BitVectorBlock *bvblock = new BitVectorBlock(num_);

    block_->Scan(Comparator::kLess, 900, bvblock, Bitwise::kSet);
    EXPECT_EQ(900ULL, bvblock->CountOnes());

    for(size_t i=0; i<num_; i++){
        EXPECT_EQ(bvblock->GetBit(i), block_->GetTuple(i) < 900);
    }

    block_->Scan(Comparator::kGreaterEqual, 500, bvblock, Bitwise::kAnd);
    EXPECT_EQ(400, bvblock->CountOnes());

    block_->Scan(Comparator::kEqual, 1999, bvblock, Bitwise::kOr);
    EXPECT_EQ(401, bvblock->CountOnes());
    EXPECT_EQ(false, bvblock->GetBit(2000));
    EXPECT_EQ(true, bvblock->GetBit(1999));
    EXPECT_EQ(false, bvblock->GetBit(1998));

    delete bvblock;
}



}   //namespace
