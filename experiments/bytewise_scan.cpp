#include    <ctime>

#include    "include/common.h"
#include    "include/types.h"
#include    "include/byteslice_column_block.h"
#include    "include/bitvector_block.h"
#include	"include/avx-utility.h"

using namespace byteslice;

int main(){
	size_t num_tuples = 64;
	double selectivity = 0.3;
	Comparator comp = Comparator::kLess;

	ByteSliceColumnBlock<12, Direction::kRight>* col_block = new ByteSliceColumnBlock<Direction::12, kRight>();
	ByteSliceColumnBlock<8, Direction::kRight>* byte_block = new ByteSliceColumnBlock<Direction::8, kRight>();
	BitVectorBlock* bv_block1 = new BitVectorBlock(num_tuples);
	BitVectorBlock* bv_block2 = new BitVectorBlock(num_tuples);

	const WordUnit mask = (1ULL << 12) - 1;
	WordUnit literal = static_cast<WordUnit>(mask * selectivity);
	WordUnit byte_literal = literal >> 8;

	std::srand(std::time(0));
    for(size_t i = 0; i < num_tuples; i++){
        WordUnit code = std::rand() & mask;
        ByteUnit byte = FLIP(static_cast<ByteUnit>(literal >> 8));
        col_block->SetTuple(i, code);
        byte_block->setTuple(i, byte);
    }

    bv_block1->SetOnes();
    bv_block2->SetOnes();

    col_block->ScanByte(comp, literal, 0, bv_block1, Bitwise::kSet);
    byte_block->Scan(comp, byte_literal, bv_block2, Bitwise::kSet);

    std::cout << "ScanByte: " << bv_block1 << std::endl;
    std::cout << "Scan    : " << bv_block2 << std::endl;
}