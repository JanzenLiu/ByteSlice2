#include    "include/bitslice_column_block.h"
#include    <cstdlib>
#include    "gtest/gtest.h"
#include    "include/bitvector_block.h"

namespace byteslice{

class BitSliceColumnBlockTest: public ::testing::Test{
public:
    virtual void SetUp(){
        std::srand(std::time(0));
        block_ = new BitSliceColumnBlock(num_, bit_width_);
        codes_ = new WordUnit[num_];
        for(size_t i=0; i<num_; i++){
            codes_[i] = std::rand() & mask_;
        }

        block_->BulkLoadArray(codes_, num_);
    }

    virtual void TearDown(){
        delete codes_;
        delete block_;
    }

protected:
    WordUnit* codes_;
    ColumnBlock* block_;
    size_t bit_width_ = 13;
    size_t num_ = 0.8 * kNumTuplesPerBlock;
    WordUnit mask_ = (1ULL << bit_width_) - 1;
};

TEST_F(BitSliceColumnBlockTest, DISABLED_SerDeser){
    std::string filename = std::string("temp/test.dat");
    //Serialize this block
    SequentialWriteBinaryFile outfile;
    outfile.Open(filename);
    block_->SerToFile(outfile);
    outfile.Close();

    //Deserialize from file
    ColumnBlock* block2 = new BitSliceColumnBlock(num_, 13);
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

TEST_F(BitSliceColumnBlockTest, SetGetTuple){
    for(size_t i=0; i<num_; i++){
        block_->SetTuple(i, codes_[i]/2);
    }
    for(size_t i=0; i<num_; i++){
        EXPECT_EQ(codes_[i]/2, block_->GetTuple(i));
    }

}

TEST_F(BitSliceColumnBlockTest, BulkLoadArray){
    for(size_t i=0; i < num_; i++){
        EXPECT_EQ(codes_[i], block_->GetTuple(i));
    }
}

TEST_F(BitSliceColumnBlockTest, ScanLiteral){
    BitVectorBlock* bvblock = new BitVectorBlock(num_);

    WordUnit literal = std::rand() & mask_;

    block_->Scan(Comparator::kLess, literal, bvblock, Bitwise::kSet);

    //Verify
    size_t count = 0;
    for(size_t i=0; i < num_; i++){
        if(codes_[i] < literal){
            count++;
        }
        EXPECT_EQ(bvblock->GetBit(i), codes_[i] < literal);
    }
    EXPECT_EQ(count, bvblock->CountOnes());

    //block_->Scan(Comparator::kLessEqual, 1024, bvblock, Bitwise::kSet);
    //count = bvblock->CountOnes();
    //EXPECT_EQ(1025, count);

    //block_->Scan(Comparator::kGreater, 512, bvblock, Bitwise::kAnd);
    //count = bvblock->CountOnes();
    //EXPECT_EQ(512, count);

    //block_->Scan(Comparator::kEqual, 10, bvblock, Bitwise::kOr);
    //count = bvblock->CountOnes();
    //EXPECT_EQ(513, count);
    //EXPECT_EQ(true, bvblock->GetBit(10));
    //EXPECT_EQ(false, bvblock->GetBit(9));

    delete bvblock;
}

TEST_F(BitSliceColumnBlockTest, ScanOtherBlock){
    size_t count;
    BitVectorBlock* bvblock = new BitVectorBlock(num_);
    BitSliceColumnBlock* block2 = new BitSliceColumnBlock(num_, 13);

    //generate a random array
    WordUnit mask = (1ULL << 13) - 1;
    std::srand(std::time(0));
    WordUnit* codes = new WordUnit[num_];
    for(size_t i=0; i<num_; i++){
        codes[i] = std::rand() & mask;
    }
    block2->BulkLoadArray(codes, num_);

    block_->Scan(Comparator::kGreaterEqual, block2, bvblock, Bitwise::kSet);

    for(size_t i=0; i < num_; i++){
        EXPECT_EQ(bvblock->GetBit(i), block_->GetTuple(i) >= codes[i]);
    }


    delete[] codes;
    delete block2;
    delete bvblock;
}

}   //namespace
