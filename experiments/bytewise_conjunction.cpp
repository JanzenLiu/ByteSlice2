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
#include	"include/bytewise_scan.h"

using namespace byteslice;

int main(){
	size_t num_rows = 64;
	Comparator comparator = Comparator::kLess;
	double selectivity = 0.3;

	Column* column1 = new Column(ColumnType::kByteSlicePadRight, 15, num_rows);
	Column* column2 = new Column(ColumnType::kByteSlicePadRight, 20, num_rows);
	Column* column3 = new Column(ColumnType::kByteSlicePadRight, 25, num_rows);

	// testing Scan
	std::srand(std::time(0)); //set random seed
	const WordUnit mask = (1ULL << 15) - 1;
	WordUnit literal = static_cast<WordUnit>(mask * selectivity);
	for(size_t i = 0; i < num_rows; i++){
        ByteUnit code = std::rand() & mask;
        column1->SetTuple(i, code);
    }

	BytewiseScan scan;
	scan.AddPredicate(BytewiseAtomPredicate(column1, comparator, literal));
	// scan.AddPredicate(BytewiseAtomPredicate(column2, comparator, literal));
	// scan.AddPredicate(BytewiseAtomPredicate(column3, comparator, literal));

	BitVector* bitvector1 = new BitVector(num_rows);
	BitVector* bitvector2 = new BitVector(num_rows);
	bitvector1->SetOnes();
    bitvector2->SetOnes();
	scan.Scan(bitvector1);
	column1->Scan(comparator, literal, bitvector2, Bitwise::kSet);

	//calculate accuracy
	size_t corr = 0; //count correct tuples
	double acc = 0;
    std::cout << "Scan" << "\t" << "ScanByte" << std::endl;
    for(size_t i = 0; i < num_rows; i++){ 
        if(bitvector1->GetBit(i) == bitvector2->GetBit(i)) 
            corr++; 
        std::cout << bitvector2->GetBit(i) << "\t\t" << bitvector1->GetBit(i) << std::endl;
    }
    acc = (double)corr / num_rows;
    std::cout << "Number of correct tuples: " << corr << std::endl; 
    std::cout << "Accuracy: " << acc << std::endl;

	// testing ValidSequence
	// Sequence seq;
	// seq.push_back(ByteInColumn(0, 0));
	// seq.push_back(ByteInColumn(0, 1));
	// seq.push_back(ByteInColumn(1, 0));
	// seq.push_back(ByteInColumn(1, 1));
	// seq.push_back(ByteInColumn(1, 2));
	// seq.push_back(ByteInColumn(2, 0));
	// seq.push_back(ByteInColumn(2, 1));
	// seq.push_back(ByteInColumn(2, 2));
	// seq.push_back(ByteInColumn(2, 3));

	// bool valid = scan.ValidSequence(seq);
	// std::cout << "#Bytes: " << scan.num_bytes_all() << std::endl;
	// std::cout << valid << std::endl;

	// testing NaturalSequence
	// Sequence seq = scan.NaturalSequence();
	// std::cout << "Column" << "\t" << "Byte" << std::endl;
	// for(size_t i = 0; i < seq.size(); i++){
	// 	std::cout << seq[i].column_id << "\t" << seq[i].byte_id << std::endl;
	// }

	// testing RandomSequence
	// Sequence seq = scan.RandomSequence();
	// std::cout << "Column" << "\t" << "Byte" << std::endl;
	// for(size_t i = 0; i < seq.size(); i++){
	// 	std::cout << seq[i].column_id << "\t" << seq[i].byte_id << std::endl;
	// }
}
