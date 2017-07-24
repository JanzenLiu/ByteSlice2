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
	size_t num_rows = 1024;
	Comparator comparator = Comparator::kLess;
	WordUnit literal = 0ULL;

	Column* column1 = new Column(ColumnType::kByteSlicePadRight, 15, num_rows);
	Column* column2 = new Column(ColumnType::kByteSlicePadRight, 20, num_rows);
	Column* column3 = new Column(ColumnType::kByteSlicePadRight, 25, num_rows);

	BytewiseScan scan;
	scan.AddPredicate(BytewiseAtomPredicate(column1, comparator, literal));
	scan.AddPredicate(BytewiseAtomPredicate(column2, comparator, literal));
	scan.AddPredicate(BytewiseAtomPredicate(column3, comparator, literal));

	// generate test case
	Sequence seq;
	seq.push_back(ByteInColumn(0, 0));
	seq.push_back(ByteInColumn(1, 0));
	seq.push_back(ByteInColumn(1, 1));
	seq.push_back(ByteInColumn(2, 0));
	seq.push_back(ByteInColumn(0, 1));
	seq.push_back(ByteInColumn(2, 1));
	seq.push_back(ByteInColumn(0, 2));
	seq.push_back(ByteInColumn(2, 2));
	seq.push_back(ByteInColumn(2, 3));

	bool valid = scan.ValidSequence(seq);
	std::cout << "#Bytes: " << scan.num_bytes_all() << std::endl;
	std::cout << valid << std::endl;
}
