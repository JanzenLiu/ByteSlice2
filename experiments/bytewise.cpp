#include    <iostream>
#include    <iomanip>
#include    <fstream>
#include    <unistd.h>
#include    <cstring>
#include    <cstdlib>
#include    <ctime>

#include    "include/types.h"
#include    "include/column.h"
#include    "include/bitvector.h"
#include    "include/bitvector_block.h"

using namespace byteslice;

int main(){
	//default parameters
	ColumnType type = ColumnType::kByteSlicePadRight;
    size_t num_rows = 2*1024*1024;
    size_t code_length = 22;
    double selectivity = 0.3;
    Comparator comparator = Comparator::kLess;
    size_t repeat = 1;

    std::srand(std::time(0)); //set random seed

    //initalize experimental variables
    Column* column = new Column(ColumnType::kByteSlicePadRight, code_length, num_rows);
    BitVector* bitvector1 = new BitVector(num_rows);
    BitVector* bitvector2 = new BitVector(num_rows);
    (void) bitvector1;
    ByteMaskVector* bm_less = new ByteMaskVector(num_rows);
    ByteMaskVector* bm_greater = new ByteMaskVector(num_rows);
    ByteMaskVector* bm_equal = new ByteMaskVector(num_rows);
    // ByteMaskBlock* bm_result = new ByteMaskBlock(num_rows);
    // ByteMaskBlock* bm_less0 = new ByteMaskBlock(num_rows);
    // ByteMaskBlock* bm_greater0 = new ByteMaskBlock(num_rows);
    // ByteMaskBlock* bm_equal0 = new ByteMaskBlock(num_rows);
    // ByteMaskBlock* bm_less1 = new ByteMaskBlock(num_rows);
    // ByteMaskBlock* bm_greater1 = new ByteMaskBlock(num_rows);
    // ByteMaskBlock* bm_equal1 = new ByteMaskBlock(num_rows);
    // ByteMaskBlock* bm_less2 = new ByteMaskBlock(num_rows);
    // ByteMaskBlock* bm_greater2 = new ByteMaskBlock(num_rows);
    // ByteMaskBlock* bm_equal2 = new ByteMaskBlock(num_rows);
    bitvector1->SetOnes();
    bitvector2->SetOnes();
    bm_less->SetAllFalse();
    bm_greater->SetAllFalse();
    bm_equal->SetAllTrue();
    // bm_less0->SetAllFalse();
    // bm_greater0->SetAllFalse();
    // bm_equal0->SetAllTrue();
    // bm_less1->SetAllFalse();
    // bm_greater1->SetAllFalse();
    // bm_equal1->SetAllTrue();
    // bm_less2->SetAllFalse();
    // bm_greater2->SetAllFalse();
    // bm_equal2->SetAllTrue();


    const WordUnit mask = (1ULL << code_length) - 1;
    WordUnit literal = static_cast<WordUnit>(mask * selectivity);
    ByteUnit byte0 = static_cast<ByteUnit>(literal >> 14);
    ByteUnit byte1 = static_cast<ByteUnit>(literal >> 6);
    ByteUnit byte2 = static_cast<ByteUnit>(literal << 2) >> 2;
    //set column randomly
    for(size_t i = 0; i < num_rows; i++){
        ByteUnit code = std::rand() & mask;
        column->SetTuple(i, code);
    }

	//single-byte column test
	// column->ScanByte(comparator, literal, 0, bitvector1, Bitwise::kSet);
    column->ScanByte(0, comparator, byte0, bm_less, bm_greater, bm_equal);
    column->ScanByte(1, comparator, byte1, bm_less, bm_greater, bm_equal, bm_equal);
    column->ScanByte(2, comparator, byte2, bm_less, bm_greater, bm_equal, bm_equal);
	column->Scan(comparator, literal, bitvector2, Bitwise::kSet);

    bm_less->Condense(bitvector1);
    // bm_less1->And(bm_equal0);
    // bm_equal1->And(bm_equal0);
    // bm_less2->And(bm_equal1);
    // bm_less0->Or(bm_less1);
    // bm_less0->Or(bm_less2);
    // bm_result->Set(bm_less0);
    // bm_result->Condense(bitvector1->GetBVBlock(0));

	//calculate accuracy
	size_t corr = 0; //count correct tuples
	double acc = 0;
    std::cout << "Scan" << "\t" << "ScanByte" << std::endl;
    for(size_t i = 0; i < num_rows; i++){ 
        // if(bm_less->GetByteMask(i) == bitvector2->GetBit(i))
        if(bitvector1->GetBit(i) == bitvector2->GetBit(i)) 
            corr++; 
        // std::cout << bitvector2->GetBit(i) << "\t\t" << bm_less->GetByteMask(i) << std::endl;
    }
    acc = (double)corr / num_rows;
    std::cout << "Number of correct tuples: " << corr << std::endl; 
    std::cout << "Accuracy: " << acc << std::endl; 

}   