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
	size_t num_rows = 1024*1024*1024;
	size_t code_length1 = 12;
	size_t code_length2 = 14;
	double selectivity1 = 0.1;
	double selectivity2 = 0.2;
	Comparator comparator = Comparator::kLess;

	// initialize variables

	// initialize experimental variables
	const WordUnit mask1 = (1ULL << code_length1) - 1;
    const WordUnit mask2 = (1ULL << code_length2) - 1;
    WordUnit literal1 = static_cast<WordUnit>(mask1 * selectivity1);
    WordUnit literal2 = static_cast<WordUnit>(mask2 * selectivity2);

    Column* column1 = new Column(ColumnType::kByteSlicePadRight, code_length1, num_rows);
    Column* column2 = new Column(ColumnType::kByteSlicePadRight, code_length2, num_rows);
    BitVector* bitvector1 = new BitVector(num_rows);
    BitVector* bitvector2 = new BitVector(num_rows);
    BitVector* bitvector3 = new BitVector(num_rows);

    //populate the column with random data
    for(size_t i = 0; i < num_rows; i++){
        WordUnit code1 = std::rand() & mask1;
        column1->SetTuple(i, code1);
    }
    for(size_t i = 0; i < num_rows; i++){
        WordUnit code2 = std::rand() & mask2;
        column2->SetTuple(i, code2);
    }

    bitvector1->SetOnes();
    bitvector2->SetOnes();
    PipelineScan scan;
    scan.AddPredicate(AtomPredicate(column1, comparator, literal1));
    scan.AddPredicate(AtomPredicate(column2, comparator, literal2));

    column1->Scan(comparator, literal1, bitvector1, Bitwise::kSet);
	column2->Scan(comparator, literal2, bitvector1, Bitwise::kAnd);

	scan.ExecuteBlockwise(bitvector2);

	scan.ExecuteBytewiseNaive(bitvector3);

	std::cout << "Scan Columnwise: " << bitvector1->GetBVBlock(0) << std::endl;
	std::cout << "Scan Bytewise:   " << bitvector2->GetBVBlock(0) << std::endl;
	std::cout << "Scan Bytewise:   " << bitvector3->GetBVBlock(0) << std::endl;

	delete bitvector1;
	delete bitvector2;
    delete column1;
    delete column2;
}