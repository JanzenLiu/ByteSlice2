#include    <iostream>
#include    <iomanip>
#include    <fstream>
#include    <unistd.h>
#include    <cstring>
#include    <cstdlib>
#include    <ctime>
#include	<bitset>

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
        WordUnit code = std::rand() & mask;
        column1->SetTuple(i, code);
        // std::cout << static_cast<uint16_t>(code) << " " << std::endl;
        std::cout << "Tuple#" << i << ": " << std::bitset<16>(static_cast<uint16_t>(code)) << std::endl;   
    }

    // std::cout << "Byte#0: " << std::bitset<64>(static_cast<WordUnit>(column1->GetBlock(0)->GetAvxUnit(0, 0))) << std::endl;
    // std::cout << "Byte#1: " << std::bitset<64>(static_cast<WordUnit>(column1->GetBlock(0)->GetAvxUnit(0, 1))) << std::endl;
    AvxUnit avx = column1->GetBlock(0)->GetAvxUnit(0, 0);
    WordUnit word1 = static_cast<WordUnit>(avx[0]);
    WordUnit word2 = *reinterpret_cast<WordUnit*>(&avx);
    // WordUnit word3 = static_cast<WordUnit>(avx);
    std::cout << std::bitset<64>(word1) << std::endl;
    std::cout << std::bitset<64>(word2) << std::endl;

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
	std::cout << "Literal: " << std::bitset<16>(static_cast<uint16_t>(literal)) << std::endl;
	// std::cout << "Converted to Byte: " << std::bitset<8>(static_cast<ByteUnit>(literal)) << std::endl;
    for(size_t i = 0; i < num_rows; i++){ 
        if(bitvector1->GetBit(i) == bitvector2->GetBit(i)) 
            corr++; 
        // std::cout << literal <<  "\t" << column1->GetTuple(i) << "\t"
        // 	<< bitvector2->GetBit(i) << "\t" << bitvector1->GetBit(i) << std::endl;
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
