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

int main(int argc, char* argv[]){
	size_t num_rows = 1024*1024*1024;
	Comparator comparator = Comparator::kLess;
	size_t code_length1 = 15;
	size_t code_length2 = 20;
	size_t code_length3 = 25;
	double selectivity1 = 0.8;
	double selectivity2 = 0.2;
	double selectivity3 = 0.3;

	//get options:
    //s - column size; p - predicate

    int c;
    while((c = getopt(argc, argv, "s:p:")) != -1){
        switch(c){
            case 'p':
                if(0 == strcmp(optarg, "lt"))
                    comparator = Comparator::kLess;
                else if(0 == strcmp(optarg, "le"))
                    comparator = Comparator::kLessEqual;
                else if(0 == strcmp(optarg, "gt"))
                    comparator = Comparator::kGreater;
                else if(0 == strcmp(optarg, "ge"))
                    comparator = Comparator::kGreaterEqual;
                else if(0 == strcmp(optarg, "eq"))
                    comparator = Comparator::kEqual;
                else if(0 == strcmp(optarg, "ne"))
                    comparator = Comparator::kInequal;
                else{
                    std::cerr << "Unknown predicate: " << optarg << std::endl;
                    exit(1);
                }
                break;
            case 's':
                num_rows = atoi(optarg);
                break;
        }
    }


	Column* column1 = new Column(ColumnType::kByteSlicePadRight, code_length1, num_rows);
	Column* column2 = new Column(ColumnType::kByteSlicePadRight, code_length2, num_rows);
	Column* column3 = new Column(ColumnType::kByteSlicePadRight, code_length3, num_rows);

	// testing Scan
	std::srand(std::time(0)); //set random seed
	const WordUnit mask1 = (1ULL << code_length1) - 1;
	const WordUnit mask2 = (1ULL << code_length2) - 1;
	const WordUnit mask3 = (1ULL << code_length3) - 1;
	WordUnit literal1 = static_cast<WordUnit>(mask1 * selectivity1);
	WordUnit literal2 = static_cast<WordUnit>(mask2 * selectivity2);
	WordUnit literal3 = static_cast<WordUnit>(mask3 * selectivity3);

	for(size_t i = 0; i < num_rows; i++){
        WordUnit code = std::rand() & mask1;
        column1->SetTuple(i, code);   
    }

    for(size_t i = 0; i < num_rows; i++){
        WordUnit code = std::rand() & mask2;
        column2->SetTuple(i, code);   
    }

    for(size_t i = 0; i < num_rows; i++){
        WordUnit code = std::rand() & mask3;
        column3->SetTuple(i, code);   
    }

	BytewiseScan scan;
	scan.AddPredicate(BytewiseAtomPredicate(column1, comparator, literal1));
	scan.AddPredicate(BytewiseAtomPredicate(column2, comparator, literal2));
	scan.AddPredicate(BytewiseAtomPredicate(column3, comparator, literal3));
	scan.ShuffleSequence();
	scan.PrintSequence();

	BitVector* bitvector1 = new BitVector(num_rows);
	BitVector* bitvector2 = new BitVector(num_rows);
	bitvector1->SetOnes();
    bitvector2->SetOnes();
	scan.Scan(bitvector1);
	column1->Scan(comparator, literal1, bitvector2, Bitwise::kSet);
	column2->Scan(comparator, literal2, bitvector2, Bitwise::kAnd);
	column3->Scan(comparator, literal3, bitvector2, Bitwise::kAnd);

	//calculate accuracy
	size_t corr = 0; //count correct tuples
	double acc = 0;
    for(size_t i = 0; i < num_rows; i++){ 
        if(bitvector1->GetBit(i) == bitvector2->GetBit(i)) 
            corr++; 
        // std::cout 
        	// << literal1 << "\t" <<  column1->GetTuple(i) << "\t"
        	// << literal2 << "\t" << column2->GetTuple(i) << "\t"
        	// << bitvector2->GetBit(i) << "\t" << bitvector1->GetBit(i) << std::endl;
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
