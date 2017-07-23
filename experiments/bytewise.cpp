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
#include    "include/byte_mask_block.h"

using namespace byteslice;

int main(){
	//default parameters
	ColumnType type = ColumnType::kByteSlicePadRight;
    size_t num_rows = 64;
    size_t code_length = 8;
    double selectivity = 0.3;
    Comparator comparator = Comparator::kLess;
    size_t repeat = 1;

    std::srand(std::time(0)); //set random seed

    //initalize experimental variables
    Column* column = new Column(ColumnType::kByteSlicePadRight, code_length, num_rows);
    BitVector* bitvector1 = new BitVector(num_rows);
    BitVector* bitvector2 = new BitVector(num_rows);
    (void) bitvector1;
    ByteMaskBlock bm_less = new ByteMaskBlock(num_rows);
    ByteMaskBlock bm_greater = new ByteMaskBlock(num_rows);
    ByteMaskBlock bm_equal = new ByteMaskBlock(num_rows);
    //bitvector1->SetOnes();
    bitvector2->SetOnes();
    bm_less->SetAllFalse();
    bm_greater->SetAllFalse();
    bm_equal->SetAllTrue();
    const ByteUnit mask = (1ULL << code_length) - 1;
    ByteUnit literal = static_cast<ByteUnit>(mask * selectivity);

    //set column randomly
    for(size_t i = 0; i < num_rows; i++){
        ByteUnit code = std::rand() & mask;
        column->SetTuple(i, code);
    }

	//single-byte column test
	// column->ScanByte(comparator, literal, 0, bitvector1, Bitwise::kSet);
    columm->GetBlock(0)->ScanByte(comparator, literal, 0, bm_less, bm_greater, bm_equal);
	column->Scan(comparator, literal, bitvector2, Bitwise::kSet);

	//calculate accuracy
	size_t corr = 0; //count correct tuples
	double acc = 0;
    std::cout << "Scan" << "\t" << "ScanByte" << std::endl;
    for(size_t i = 0; i < num_rows; i++){ 
        if(bitvector1->GetBit(i) == bitvector2->GetBit(i)) 
            corr++; 
        std::cout << bitvector2->GetBit(i) << "\t\t" << bm_less->GetByteMask(i) << std::endl;
    }
    acc = (double)corr / num_rows;
    std::cout << "Number of correct tuples: " << corr << std::endl; 
    std::cout << "Accuracy: " << acc << std::endl; 

}