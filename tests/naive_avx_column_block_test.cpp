#include    "include/naive_avx_column_block.h"
#include    <cstdint>
#include    "gtest/gtest.h"
#include    "include/bitvector_block.h"

namespace byteslice{

class NaiveAvxColumnBlockTest: public ::testing::Test{
public:
    virtual void SetUp(){
        num = 5000;
        block = new NaiveAvxColumnBlock<uint16_t>(num);
        //populate data
        WordUnit* codes = new WordUnit[num];
        for(size_t i=0; i<num; i++){
            codes[i] = static_cast<WordUnit>(i);
        }
        block->BulkLoadArray(codes, num);
        delete[] codes;
    }

    virtual void TearDown(){
        delete block;
    }

protected:
    ColumnBlock* block;
    size_t num;
};

TEST_F(NaiveAvxColumnBlockTest, BulkLoadArray){
    for(size_t i=0; i<10; i++){
        EXPECT_EQ(i, block->GetTuple(i));
    }
}

TEST_F(NaiveAvxColumnBlockTest, DISABLED_SerDeser){
    std::string filename = std::string("temp/test.dat");
    //Serialize this block
    SequentialWriteBinaryFile outfile;
    outfile.Open(filename);
    block->SerToFile(outfile);
    outfile.Close();

    //Deserialize from file
    ColumnBlock* block2 = new NaiveAvxColumnBlock<uint16_t>(num);
    SequentialReadBinaryFile infile;
    infile.Open(filename);
    block2->DeserFromFile(infile);
    infile.Close();

    //Verify
    for(size_t i=0; i<num; i++){
        EXPECT_EQ(block->GetTuple(i), block2->GetTuple(i));
    }

    delete block2;
}


TEST_F(NaiveAvxColumnBlockTest, ScanLiteral){
    BitVectorBlock* bvblock = new BitVectorBlock(num);
    size_t count;

    //Less than 87 = 64 + 23, 23 = 16 + 7
    block->Scan(Comparator::kLess, 87, bvblock, Bitwise::kSet);
    count = bvblock->CountOnes();
    EXPECT_EQ(87, count);

    EXPECT_EQ(0xffffffffffffffff, bvblock->GetWordUnit(0));
    EXPECT_EQ(0x7fffff, bvblock->GetWordUnit(1));

    //test AND/OR
    block->Scan(Comparator::kLess, 1024, bvblock, Bitwise::kOr);
    count = bvblock->CountOnes();
    EXPECT_EQ(1024, count);
    block->Scan(Comparator::kGreaterEqual, 1023, bvblock, Bitwise::kAnd);
    count = bvblock->CountOnes();
    EXPECT_EQ(1, count);

    delete bvblock;
}

TEST_F(NaiveAvxColumnBlockTest, ScanOtherBlock){
    BitVectorBlock* bvblock = new BitVectorBlock(num);
    NaiveAvxColumnBlock<uint16_t>* block2 = new NaiveAvxColumnBlock<uint16_t>(num);
    size_t count;
    //populate a constant
    WordUnit* codes = new WordUnit[num];
    for(size_t i=0; i<num; i++){
        codes[i] = static_cast<WordUnit>(87);
    }
    block2->BulkLoadArray(codes, num);
    delete[] codes;

    block->Scan(Comparator::kLess, block2, bvblock, Bitwise::kSet);
    count = bvblock->CountOnes();
    EXPECT_EQ(87, count);

    EXPECT_EQ(0xffffffffffffffff, bvblock->GetWordUnit(0));
    EXPECT_EQ(0x7fffff, bvblock->GetWordUnit(1));

    delete block2;
    delete bvblock;
}

}   //namespace
