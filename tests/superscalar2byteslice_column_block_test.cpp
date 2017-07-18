#include    "include/superscalar2_byteslice_column_block.h"
#include    <cstdlib>
#include    "gtest/gtest.h"
#include    "include/bitvector_block.h"

namespace byteslice{

class Superscalar2ByteSliceColumnBlockTest: public ::testing::Test{
public:
    virtual void SetUp(){
        num_ = kNumTuplesPerBlock*0.8;
        block_ = new Superscalar2ByteSliceColumnBlock<20>(num_);

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
    Superscalar2ByteSliceColumnBlock<20>* block_;
    size_t num_;
};

TEST_F(Superscalar2ByteSliceColumnBlockTest, DISABLED_SerDeser){
    std::string filename = std::string("temp/test.dat");
    //Serialize this block
    SequentialWriteBinaryFile outfile;
    outfile.Open(filename);
    block_->SerToFile(outfile);
    outfile.Close();

    //Deserialize from file
    ColumnBlock* block2 = new Superscalar2ByteSliceColumnBlock<20>(num_);
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

TEST_F(Superscalar2ByteSliceColumnBlockTest, BulkLoadAndGetTuple){
    for(size_t i=0; i<num_; i++){
        EXPECT_EQ(i, block_->GetTuple(i));
    }
}

/*
TEST_F(Superscalar2ByteSliceColumnBlockTest, ScanWithByteMask){
    BitVectorBlock* bvblock1 = new BitVectorBlock(num_);
    BitVectorBlock* bvblock2 = new BitVectorBlock(num_);
    ByteMaskBlock* bmblk = new ByteMaskBlock(num_);

    std::srand(std::time(0));
    const WordUnit lit = std::rand() % num_;

    //Scan with bit vector
    block_->Scan(Comparator::kLess, lit, bvblock1, Bitwise::kSet);
    EXPECT_EQ(lit, bvblock1->CountOnes());

    //Scan with ByteMaskBlock
    block_->Scan(Comparator::kLess, lit, bmblk, Bitwise::kSet);
    bmblk->Condense(bvblock2);
    EXPECT_EQ(lit, bvblock2->CountOnes());

    //Check: bvblock1 and bvblock2 should be the same
    for(size_t i = 0; i < bvblock1->num_word_units(); i++){
        EXPECT_EQ(bvblock1->GetWordUnit(i), bvblock2->GetWordUnit(i));
    }
    
    delete bvblock1;
    delete bvblock2;
    delete bmblk;
   
}
*/

TEST_F(Superscalar2ByteSliceColumnBlockTest, ScanLiteral){
    BitVectorBlock* bvblock = new BitVectorBlock(num_);

    std::srand(std::time(0));
    const WordUnit lit = std::rand() % num_;
    block_->Scan(Comparator::kLess, lit, bvblock, Bitwise::kSet);
    EXPECT_EQ(lit, bvblock->CountOnes());

	//debug: find why *bvblock->CountOnes()* larger than *lit* sometimes
	/*
	std::cout << "literal: " << lit << std::endl << std::endl;
	for (size_t i = lit; i < num_; i++) {
		if (bvblock->GetBit(i)) {
			std::cout << "unexpected index:" << i << std::endl;
		}
	}
	*/

    delete bvblock;
}

TEST_F(Superscalar2ByteSliceColumnBlockTest, ScanOtherBlock){
    BitVectorBlock* bvblock = new BitVectorBlock(num_);
    Superscalar2ByteSliceColumnBlock<20>* block2 = new Superscalar2ByteSliceColumnBlock<20>(num_);

    std::srand(std::time(0));
    for(size_t i=0; i < num_; i++){
        block2->SetTuple(i, std::rand() % num_);
    }

    block_->Scan(Comparator::kGreaterEqual, block2, bvblock, Bitwise::kSet);
    for(size_t i=0; i < num_; i++){
        EXPECT_EQ(bvblock->GetBit(i), block_->GetTuple(i) >= block2->GetTuple(i));
    }

    delete block2;
    delete bvblock;
}



}   //namespace
