#include    <ctime>
#include    <vector>

#include    "include/hybrid_timer.h"

#include    "include/types.h"
#include    "include/column.h"
#include    "include/bitvector.h"
#include    "include/pipeline_scan.h"

using namespace byteslice;

int main(int argc, char* argv[]){
	// set default parameters
	size_t num_rows = 1024*1024;
	size_t code_length1 = 12;
	size_t code_length2 = 14;
	double selectivity1 = 0.4;
	double selectivity2 = 0.5;
	Comparator comparator = Comparator::kLess;

	// initialize variables

	// initialize experimental variables
	const WordUnit mask1 = (1ULL << code_length1) - 1;
    const WordUnit mask2 = (1ULL << code_length2) - 1;
    WordUnit literal1 = static_cast<WordUnit>(mask1 * selectivity1);
    WordUnit literal2 = static_cast<WordUnit>(mask2 * selectivity2);
    std::cout << "Literal#1: " << literal1 << std::endl;
    std::cout << "Literal#2: " << literal2 << std::endl;

    Column* column1 = new Column(ColumnType::kByteSlicePadRight, code_length1, num_rows);
    Column* column2 = new Column(ColumnType::kByteSlicePadRight, code_length2, num_rows);
    BitVector* bitvector1 = new BitVector(num_rows);
    BitVector* bitvector2 = new BitVector(num_rows);
    BitVector* bitvector3 = new BitVector(num_rows);
    std::srand(std::time(0));

    //populate the column with random data
    for(size_t i = 0; i < num_rows; i++){
        WordUnit code1 = std::rand() & mask1;
        column1->SetTuple(i, code1);
        // std::cout << "Tuple#" << i << ": " << column1->GetBlock(0)->GetTuple(i) << std::endl;
    }
    for(size_t i = 0; i < num_rows; i++){
        WordUnit code2 = std::rand() & mask2;
        column2->SetTuple(i, code2);
        // std::cout << "Tuple#" << i << ": " << column2->GetBlock(0)->GetTuple(i) << std::endl;
    }

    bitvector1->SetOnes();
    bitvector2->SetOnes();
    bitvector3->SetOnes();
    std::cout << bitvector1->CountOnes() << std::endl;
    std::cout << bitvector2->CountOnes() << std::endl;
    std::cout << bitvector3->CountOnes() << std::endl;
    PipelineScan scan;
    scan.AddPredicate(AtomPredicate(column1, comparator, literal1));
    scan.AddPredicate(AtomPredicate(column2, comparator, literal2));

    column1->Scan(comparator, literal1, bitvector1, Bitwise::kSet);
	column2->Scan(comparator, literal2, bitvector1, Bitwise::kAnd);

	scan.ExecuteBlockwise(bitvector2);

	scan.ExecuteBytewiseNaive(bitvector3);

	//std::cout << "Scan Columnwise: " << *reinterpret_cast<WordUnit *>(bitvector1->GetBVBlock(0)) << std::endl;
	//std::cout << "Scan Blockwise:  " << *reinterpret_cast<WordUnit *>(bitvector2->GetBVBlock(0)) << std::endl;
	//std::cout << "Scan Bytewise:   " << *reinterpret_cast<WordUnit *>(bitvector3->GetBVBlock(0)) << std::endl;
	std::cout << "Scan Columnwise: " << bitvector1->CountOnes() << std::endl;
	std::cout << "Scan Blockwise : " << bitvector2->CountOnes() << std::endl;
	std::cout << "Scan Bytewise  : " << bitvector3->CountOnes() << std::endl;
	delete bitvector1;
	delete bitvector2;
    delete column1;
    delete column2;
}
