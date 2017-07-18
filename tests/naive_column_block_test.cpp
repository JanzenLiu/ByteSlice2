#include    "include/naive_column_block.h"
#include    <cstdint>
#include    "gtest/gtest.h"
#include    "include/bitvector_block.h"

namespace byteslice{

class NaiveColumnBlockTest: public ::testing::Test{
public:
    virtual void SetUp(){
        num = 5000;
        col_block = new NaiveColumnBlock<uint16_t>(num);
        //populate data
        WordUnit* codes = new WordUnit[num];
        for(size_t i=0; i<num; i++){
            codes[i] = static_cast<WordUnit>(i);
        }
        col_block->BulkLoadArray(codes, num);
        delete[] codes;
    }

    virtual void TearDown(){
        delete col_block;
    }

protected:
    ColumnBlock* col_block;
    size_t num;
};

TEST_F(NaiveColumnBlockTest, DISABLED_SerDeser){
    std::string filename = std::string("temp/test.dat");
    //Serialize this block
    SequentialWriteBinaryFile outfile;
    outfile.Open(filename);
    col_block->SerToFile(outfile);
    outfile.Close();

    //Deserialize from file
    ColumnBlock* block2 = new NaiveColumnBlock<uint16_t>(num);
    SequentialReadBinaryFile infile;
    infile.Open(filename);
    block2->DeserFromFile(infile);
    infile.Close();

    //Verify
    for(size_t i=0; i<num; i++){
        EXPECT_EQ(col_block->GetTuple(i), block2->GetTuple(i));
    }

    delete block2;
}

TEST_F(NaiveColumnBlockTest, Endian){
    BitVectorBlock* bvblock = new BitVectorBlock(num);
    size_t count;
    //less than 8, shoud have 8 tuples
    col_block->Scan(Comparator::kLess, 13, bvblock, Bitwise::kSet);
    count = bvblock->CountOnes();
    EXPECT_EQ(13, count);

    WordUnit w1 = bvblock->GetWordUnit(0);
    EXPECT_EQ(0x1fff, w1);
    WordUnit w2 = bvblock->GetWordUnit(1);
    EXPECT_EQ(0x0, w2);

    delete bvblock;
}

TEST_F(NaiveColumnBlockTest, BulkLoadArray){
    for(size_t i=0; i<10; i++){
        EXPECT_EQ(i, col_block->GetTuple(i));
    }
}

TEST_F(NaiveColumnBlockTest, ScanLiteral){
    BitVectorBlock* bvblock = new BitVectorBlock(num);
    size_t count;

    //Less
    col_block->Scan(Comparator::kLess, num/10, bvblock, Bitwise::kSet);
    count = bvblock->CountOnes();
    EXPECT_EQ(num/10, count);
    //Or Greater
    col_block->Scan(Comparator::kGreater, num*9/10-1, bvblock, Bitwise::kOr);
    count = bvblock->CountOnes();
    EXPECT_EQ(num/5, count);
    //And Equal
    col_block->Scan(Comparator::kEqual, num/2, bvblock, Bitwise::kAnd);
    count = bvblock->CountOnes();
    EXPECT_EQ(0, count);

    //LessEqual
    bvblock->SetOnes();
    col_block->Scan(Comparator::kLessEqual, num/10, bvblock, Bitwise::kSet);
    count = bvblock->CountOnes();
    EXPECT_EQ(num/10 +1, count);

    delete bvblock;
}

TEST_F(NaiveColumnBlockTest, ScanAgainstBlock){
    BitVectorBlock* bvblock = new BitVectorBlock(num);
    size_t count;
    //create another block
    NaiveColumnBlock<uint16_t>* block2 = new NaiveColumnBlock<uint16_t>(num);
    for(size_t pos=0; pos < num; pos++){
        block2->SetTuple(pos, num/10);
    }
    //Scan against block2
    col_block->Scan(Comparator::kLess, block2, bvblock, Bitwise::kSet);
    count = bvblock->CountOnes();
    EXPECT_EQ(num/10, count);
    col_block->Scan(Comparator::kEqual, block2, bvblock, Bitwise::kOr);
    count = bvblock->CountOnes();
    EXPECT_EQ(num/10 + 1, count);

    delete bvblock;
    delete block2;
}


}   //namespace byteslice

