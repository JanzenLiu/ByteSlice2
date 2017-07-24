#include    "include/bytewise_scan.h"
#include    <omp.h>
#include    <vector>
#include	<random>

namespace byteslice{

void BytewiseScan::AddPredicate(BytewiseAtomPredicate predicate){
	assert(predicate.column->type() == ColumnType::kByteSlicePadRight);
    conjunctions_.push_back(predicate);
    for(size_t i = 0; i < predicate.num_bytes; i++){
    	sequence_.push_back(ByteInColumn(conjunctions_.size() - 1, i));
    }
    num_bytes_all_ += predicate.num_bytes;
}

void BytewiseScan::SetSequence(const Sequence seq){
	assert(ValidSequence(seq));
	sequence_.clear();
	for(size_t i = 0; i < seq.size(); i++){
		sequence_.push_back(ByteInColumn(seq[i].column_id, seq[i].byte_id));
	}
}

bool BytewiseScan::ValidSequence(Sequence seq) const{
	if(sequence_.size() != seq.size())
		return false;

	// counter to record next expected byte to appear in the sequence for each column/predicate
	size_t* next_bytes = (size_t*)malloc(conjunctions_.size() * sizeof(size_t));
	for(size_t i = 0; i < conjunctions_.size(); i++){
		next_bytes[i] = 0;
	}

	// validate the sequence
	for(size_t i = 0; i < seq.size(); i++){
		size_t col = seq[i].column_id;
		size_t byte = seq[i].byte_id;
		if(next_bytes[col] != -1 && next_bytes[col] == byte){
			next_bytes[col] = (byte == conjunctions_[col].num_bytes - 1)? -1 : byte + 1;
		}
		else
			return false;
	}
	return true;
}

Sequence BytewiseScan::NaturalSequence() const{
	Sequence seq;
	for(size_t i = 0; i < conjunctions_.size(); i++){
		for(size_t j = 0; j < conjunctions_[i].num_bytes; j++){
			seq.push_back(ByteInColumn(i, j));
		}
	}
	return seq;
}

Sequence BytewiseScan::RandomSequence() const{
	Sequence seq;
	for(size_t i = 0; i < conjunctions_.size(); i++){
		for(size_t j = 0; j < conjunctions_[i].num_bytes; j++){
			seq.push_back(ByteInColumn(i, -1));
		}
	}
	std::srand(std::time(0));
	std::random_shuffle(seq.begin(), seq.end());
	return seq;
}

BytewiseAtomPredicate BytewiseScan::GetPredicate(size_t pid) const{
	return conjunctions_[pid];
}

Sequence BytewiseScan::sequence() const{
	return sequence_;
}

size_t BytewiseScan::num_bytes_all() const{
	return num_bytes_all_;
}

}	//namespace