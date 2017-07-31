#include    <iostream>
#include    <iomanip>
#include    <fstream>
#include    <unistd.h>
#include    <cstring>
#include    <cstdlib>
#include    <ctime>
#include	<bitset>

#include    "include/hybrid_timer.h"

#include    "include/types.h"
#include    "include/column.h"
#include    "include/bitvector.h"
#include	"include/bytewise_scan.h"

using namespace byteslice;

int main(int argc, char* argv[]){
	// default parameters
	size_t num_rows = 1024*1024*32;
	Comparator comparator = Comparator::kLess;
	size_t code_length1 = 15;
	size_t code_length2 = 20;
	size_t code_length3 = 25;
	double selectivity1 = 0.1;
	double selectivity2 = 0.2;
	double selectivity3 = 0.3;
	size_t repeat = 10;

	//get options:
    //s - column size; p - predicate; r - repeat
    int c;
    while((c = getopt(argc, argv, "s:p:r:")) != -1){
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
            case 'r':
                repeat = atoi(optarg);
                break;
        }
    }

    //initialize experimental variables
	Column* column1 = new Column(ColumnType::kByteSlicePadRight, code_length1, num_rows);
	Column* column2 = new Column(ColumnType::kByteSlicePadRight, code_length2, num_rows);
	Column* column3 = new Column(ColumnType::kByteSlicePadRight, code_length3, num_rows);

	std::srand(std::time(0)); //set random seed
	const WordUnit mask1 = (1ULL << code_length1) - 1;
	const WordUnit mask2 = (1ULL << code_length2) - 1;
	const WordUnit mask3 = (1ULL << code_length3) - 1;
	WordUnit literal1 = static_cast<WordUnit>(mask1 * selectivity1);
	WordUnit literal2 = static_cast<WordUnit>(mask2 * selectivity2);
	WordUnit literal3 = static_cast<WordUnit>(mask3 * selectivity3);

	BitVector* bitvector1 = new BitVector(num_rows);
	BitVector* bitvector2 = new BitVector(num_rows);

	//set bytewise_scan
	BytewiseScan scan;
	scan.AddPredicate(BytewiseAtomPredicate(column1, comparator, literal1));
	scan.AddPredicate(BytewiseAtomPredicate(column2, comparator, literal2));
	scan.AddPredicate(BytewiseAtomPredicate(column3, comparator, literal3));
	// scan.ShuffleSequence();
	// scan.PrintSequence();

    HybridTimer t1;

    uint64_t cycles_bytewise = 0, cycles_bytewise_columnfirst = 0, cycles_columnwise = 0;
    uint64_t cycles_bytewise_012 = 0, cycles_bytewise_021 = 0, cycles_bytewise_102 = 0;
    uint64_t cycles_bytewise_120 = 0, cycles_bytewise_201 = 0, cycles_bytewise_210 = 0;
    uint64_t cycles_columnfirst_012 = 0, cycles_columnfirst_021 = 0, cycles_columnfirst_102 = 0;
    uint64_t cycles_columnfirst_120 = 0, cycles_columnfirst_201 = 0, cycles_columnfirst_210 = 0;
    uint64_t cycles_columnar1 = 0, cycles_columnar2 = 0, cycles_columnar3 = 0;

    ByteInColumn bc00 = ByteInColumn(0, 0);
    ByteInColumn bc01 = ByteInColumn(0, 1);
    ByteInColumn bc10 = ByteInColumn(1, 0);
    ByteInColumn bc11 = ByteInColumn(1, 1);
    ByteInColumn bc12 = ByteInColumn(1, 2);
    ByteInColumn bc20 = ByteInColumn(2, 0);
    ByteInColumn bc21 = ByteInColumn(2, 1);
    ByteInColumn bc22 = ByteInColumn(2, 2);
    ByteInColumn bc23 = ByteInColumn(2, 3);

    ByteInColumn arr012[] = {bc00, bc10, bc20, bc01, bc11, bc21, bc12, bc22, bc23};
    ByteInColumn arr021[] = {bc00, bc20, bc10, bc01, bc11, bc21, bc12, bc22, bc23};
    ByteInColumn arr102[] = {bc10, bc00, bc20, bc01, bc11, bc21, bc12, bc22, bc23};
    ByteInColumn arr120[] = {bc10, bc20, bc00, bc01, bc11, bc21, bc12, bc22, bc23};
    ByteInColumn arr201[] = {bc20, bc00, bc10, bc01, bc11, bc21, bc12, bc22, bc23};
    ByteInColumn arr210[] = {bc20, bc10, bc00, bc01, bc11, bc21, bc12, bc22, bc23};

    Sequence seq012, seq021, seq102, seq120, seq201, seq210;
    seq012.assign(arr012, arr012 + 9);
    seq021.assign(arr021, arr021 + 9);
    seq102.assign(arr102, arr102 + 9);
    seq120.assign(arr120, arr120 + 9);
    seq201.assign(arr201, arr201 + 9);
    seq210.assign(arr210, arr210 + 9);

	for(size_t turn = 0; turn < repeat; turn++){
		bitvector1->SetOnes();
	    bitvector2->SetOnes();

		//set columns randomly
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


		//SCAN
		//with random sequence
		//bytewise scan
		scan.ShuffleSequence();
		scan.PrintSequence();
		t1.Start();
		scan.Scan(bitvector1);
		t1.Stop();
		cycles_bytewise += t1.GetNumCycles();
		t1.Start();
		scan.ScanColumnwise(bitvector1);
		t1.Stop();
		cycles_bytewise_columnfirst += t1.GetNumCycles();

		//with pre-defined sequence
		scan.SetSequence(seq012);
		t1.Start();
		scan.Scan(bitvector1);
		t1.Stop();
		cycles_bytewise_012 += t1.GetNumCycles();
		t1.Start();
		scan.ScanColumnwise(bitvector1);
		t1.Stop();
		cycles_columnfirst_012 += t1.GetNumCycles();

		scan.SetSequence(seq021);
		t1.Start();
		scan.Scan(bitvector1);
		t1.Stop();
		cycles_bytewise_021 += t1.GetNumCycles();
		t1.Start();
		scan.ScanColumnwise(bitvector1);
		t1.Stop();
		cycles_columnfirst_021 += t1.GetNumCycles();

		scan.SetSequence(seq102);
		t1.Start();
		scan.Scan(bitvector1);
		t1.Stop();
		cycles_bytewise_102 += t1.GetNumCycles();
		t1.Start();
		scan.ScanColumnwise(bitvector1);
		t1.Stop();
		cycles_columnfirst_102 += t1.GetNumCycles();

		scan.SetSequence(seq120);
		t1.Start();
		scan.Scan(bitvector1);
		t1.Stop();
		cycles_bytewise_120 += t1.GetNumCycles();
		t1.Start();
		scan.ScanColumnwise(bitvector1);
		t1.Stop();
		cycles_columnfirst_120 += t1.GetNumCycles();

		scan.SetSequence(seq201);
		t1.Start();
		scan.Scan(bitvector1);
		t1.Stop();
		cycles_bytewise_201 += t1.GetNumCycles();
		t1.Start();
		scan.ScanColumnwise(bitvector1);
		t1.Stop();
		cycles_columnfirst_201 += t1.GetNumCycles();

		scan.SetSequence(seq210);
		t1.Start();
		scan.Scan(bitvector1);
		t1.Stop();
		cycles_bytewise_210 += t1.GetNumCycles();
		t1.Start();
		scan.ScanColumnwise(bitvector1);
		t1.Stop();
		cycles_columnfirst_210 += t1.GetNumCycles();


		//columnwise scan
		t1.Start();
		column1->Scan(comparator, literal1, bitvector2, Bitwise::kSet);
		t1.Stop();
		cycles_columnar1 += t1.GetNumCycles();

		t1.Start();
		column2->Scan(comparator, literal2, bitvector2, Bitwise::kAnd);
		t1.Stop();
		cycles_columnar2 += t1.GetNumCycles();

		t1.Start();
		column3->Scan(comparator, literal3, bitvector2, Bitwise::kAnd);
		t1.Stop();
		cycles_columnar3 += t1.GetNumCycles();

		//calculate accuracy
		// size_t corr = 0; //count correct tuples
		// double acc = 0;
	 //    for(size_t i = 0; i < num_rows; i++){ 
	 //        if(bitvector1->GetBit(i) == bitvector2->GetBit(i)) 
	 //            corr++; 
	 //        // else{
	 //        // 	std::cout << std::bitset<15>(literal1) << "\t" << std::bitset<15>(column1->GetTuple(i)) << "\t"
	 //        // 			<< std::bitset<20>(literal2) << "\t" << std::bitset<20>(column2->GetTuple(i)) << "\t"
	 //        // 			<< std::bitset<25>(literal3) << "\t" << std::bitset<25>(column3->GetTuple(i)) << std::endl;
	 //        // }
	 //    }
	 //    acc = (double)corr / num_rows;
	 //    std::cout << "Number of correct tuples: " << corr << std::endl; 
	 //    std::cout << "Accuracy: " << acc << std::endl;
	 //    std::cout << std::endl;
	}

    //calcuate average cycles
    cycles_columnwise = cycles_columnar1 + cycles_columnar2 + cycles_columnar3;
    std::cout << "bytewise  bw-columnfirst  "
    	<< "bytewise012  bw-columnfirst012  "
    	<< "bytewise021  bw-columnfirst021  "
    	<< "bytewise102  bw-columnfirst102  "
    	<< "bytewise120  bw-columnfirst120  "
    	<< "bytewise201  bw-columnfirst201  "
    	<< "bytewise210  bw-columnfirst210  "
    	<< "columnwise col(1)  col(2)  col(3)  " << std::endl;
	std::cout
	    << double(cycles_bytewise / repeat) / num_rows << "\t\t"
	    << double(cycles_bytewise_columnfirst / repeat) / num_rows << "\t\t\t"
	    << double(cycles_bytewise_012 / repeat) / num_rows << "\t\t"
	    << double(cycles_columnfirst_012 / repeat) / num_rows << "\t\t\t"
	    << double(cycles_bytewise_021 / repeat) / num_rows << "\t\t"
	    << double(cycles_columnfirst_021 / repeat) / num_rows << "\t\t\t"
	    << double(cycles_bytewise_102 / repeat) / num_rows << "\t\t"
	    << double(cycles_columnfirst_102 / repeat) / num_rows << "\t\t\t"
	    << double(cycles_bytewise_120 / repeat) / num_rows << "\t\t"
	    << double(cycles_columnfirst_120 / repeat) / num_rows << "\t\t\t"
	    << double(cycles_bytewise_201 / repeat) / num_rows << "\t\t"
	    << double(cycles_columnfirst_201 / repeat) / num_rows << "\t\t\t"
	    << double(cycles_bytewise_210 / repeat) / num_rows << "\t\t"
	    << double(cycles_columnfirst_210 / repeat) / num_rows << "\t\t\t"
	    << double((cycles_columnwise) / repeat) / num_rows << "\t\t"
	    << double(cycles_columnar1 / repeat) / num_rows << "\t"
	    << double(cycles_columnar2 / repeat) / num_rows << "\t"
	    << double(cycles_columnar3 / repeat) / num_rows << std::endl;

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
