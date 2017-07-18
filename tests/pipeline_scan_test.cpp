#include    "include/pipeline_scan.h"
#include    "gtest/gtest.h"
#include    <cstdlib>

namespace byteslice{

class PipelineScanTest: public ::testing::Test{
public:
    virtual void SetUp(){
        std::srand(std::time(0));
        column1 = new Column(ColumnType::kByteSlicePadRight, bit_width_, num_);
        column2 = new Column(ColumnType::kByteSlicePadRight, bit_width_, num_);
        //Populate with random values
        for(size_t i=0; i < num_; i++){
            column1->SetTuple(i, std::rand() & mask_);
        }
        for(size_t i=0; i < num_; i++){
            column2->SetTuple(i, std::rand() & mask_);
        }

    }

    virtual void TearDown(){
        delete column1;
        delete column2;
    }

protected:
    size_t num_ = 13.5*kNumTuplesPerBlock;
    Column* column1;
    Column* column2;
    const size_t bit_width_ = 16;
    const WordUnit mask_ = (1ULL << bit_width_) - 1;
};

TEST_F(PipelineScanTest, Blockwise){
    WordUnit lit1 = mask_ * 0.2;
    WordUnit lit2 = mask_ * 0.5;
    PipelineScan scan;
    scan.AddPredicate(AtomPredicate(column1, Comparator::kLess, lit1));
    scan.AddPredicate(AtomPredicate(column2, Comparator::kLess, lit2));
    BitVector* bitvector = new BitVector(num_);
    bitvector->SetOnes();

    scan.ExecuteBlockwise(bitvector);
    size_t result = bitvector->CountOnes();
    std::cout << result << "\t";

    //Verify
    for(size_t i=0; i < num_; i++){
        EXPECT_EQ(bitvector->GetBit(i), column1->GetTuple(i) < lit1 && column2->GetTuple(i) < lit2);
    }

    //Compare with columnwise 
    bitvector->SetOnes();
    column1->Scan(Comparator::kLess, lit1, bitvector, Bitwise::kSet);
    std::cout << bitvector->CountOnes() << "\t";
    column2->Scan(Comparator::kLess, lit2, bitvector, Bitwise::kAnd);
    std::cout << bitvector->CountOnes() << std::endl;

    EXPECT_EQ(result, bitvector->CountOnes());

    delete bitvector;
}

TEST_F(PipelineScanTest, Columnwise){
    WordUnit lit1 = mask_ * 0.2;
    WordUnit lit2 = mask_ * 0.5;
    PipelineScan scan;
    scan.AddPredicate(AtomPredicate(column1, Comparator::kLess, lit1));
    scan.AddPredicate(AtomPredicate(column2, Comparator::kLess, lit2));
    BitVector* bitvector = new BitVector(num_);
    bitvector->SetOnes();

    scan.ExecuteColumnwise(bitvector);
    size_t result = bitvector->CountOnes();
    std::cout << result << "\t";

    //Verify
    for(size_t i=0; i < num_; i++){
        EXPECT_EQ(bitvector->GetBit(i), column1->GetTuple(i) < lit1 && column2->GetTuple(i) < lit2);
    }

    //Compare with columnwise 
    bitvector->SetOnes();
    column1->Scan(Comparator::kLess, lit1, bitvector, Bitwise::kSet);
    std::cout << bitvector->CountOnes() << "\t";
    column2->Scan(Comparator::kLess, lit2, bitvector, Bitwise::kAnd);
    std::cout << bitvector->CountOnes() << std::endl;

    EXPECT_EQ(result, bitvector->CountOnes());

    delete bitvector;
}

TEST_F(PipelineScanTest, Standard){
    WordUnit lit1 = mask_ * 0.2;
    WordUnit lit2 = mask_ * 0.5;
    PipelineScan scan;
    scan.AddPredicate(AtomPredicate(column1, Comparator::kLess, lit1));
    scan.AddPredicate(AtomPredicate(column2, Comparator::kLess, lit2));
    BitVector* bitvector = new BitVector(num_);
    bitvector->SetOnes();

    scan.ExecuteStandard(bitvector);
    size_t result = bitvector->CountOnes();
    std::cout << result << "\t";

    //Verify
    for(size_t i=0; i < num_; i++){
        EXPECT_EQ(bitvector->GetBit(i), column1->GetTuple(i) < lit1 && column2->GetTuple(i) < lit2);
    }

    //Compare with columnwise 
    bitvector->SetOnes();
    column1->Scan(Comparator::kLess, lit1, bitvector, Bitwise::kSet);
    std::cout << bitvector->CountOnes() << "\t";
    column2->Scan(Comparator::kLess, lit2, bitvector, Bitwise::kAnd);
    std::cout << bitvector->CountOnes() << std::endl;

    EXPECT_EQ(result, bitvector->CountOnes());

    delete bitvector;
}

TEST_F(PipelineScanTest, Naive){
    WordUnit lit1 = mask_ * 0.2;
    WordUnit lit2 = mask_ * 0.5;
    PipelineScan scan;
    scan.AddPredicate(AtomPredicate(column1, Comparator::kLess, lit1));
    scan.AddPredicate(AtomPredicate(column2, Comparator::kLess, lit2));
    BitVector* bitvector = new BitVector(num_);
    bitvector->SetOnes();

    scan.ExecuteNaive(bitvector);
    size_t result = bitvector->CountOnes();
    std::cout << result << "\t";

    //Verify
    for(size_t i=0; i < num_; i++){
        EXPECT_EQ(bitvector->GetBit(i), column1->GetTuple(i) < lit1 && column2->GetTuple(i) < lit2);
    }

    //Compare with columnwise 
    bitvector->SetOnes();
    column1->Scan(Comparator::kLess, lit1, bitvector, Bitwise::kSet);
    std::cout << bitvector->CountOnes() << "\t";
    column2->Scan(Comparator::kLess, lit2, bitvector, Bitwise::kAnd);
    std::cout << bitvector->CountOnes() << std::endl;

    EXPECT_EQ(result, bitvector->CountOnes());

    delete bitvector;
}

}
