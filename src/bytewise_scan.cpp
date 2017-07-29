#include    "include/bytewise_scan.h"
#include    <omp.h>
#include	<typeinfo>
#include	<ctime>
#include    <vector>
#include	<random>
#include	<algorithm>
#include	<bitset>

namespace byteslice{

#ifdef      NEARLYSTOP
#warning    "Early-stop is disabled in ByteSliceColumnBlock!"
#endif

static constexpr size_t kPrefetchDistance = 512*2;

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

void BytewiseScan::ShuffleSequence(){
	Sequence seq = RandomSequence();
	SetSequence(seq);
}

void BytewiseScan::PrintSequence(){
	std::cout << "Sequence of Bytes to Scan with:" << std::endl;
	for(size_t i = 0; i < sequence_.size(); i++){
		std::cout << "Column#" << sequence_[i].column_id << ", "
			<< "Byte#" << sequence_[i].byte_id << std::endl;
	}
	std::cout << std::endl;
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
			// byte_id is meaningless here
			seq.push_back(ByteInColumn(i, -1));
		}
	}

	// shuffle the orders in which each column appears
	std::srand(std::time(0));
	std::random_shuffle(seq.begin(), seq.end());

	// reset valid byte_id of each ByteInColumn
	size_t* next_bytes = (size_t*)malloc(conjunctions_.size() * sizeof(size_t));
	for(size_t i = 0; i < conjunctions_.size(); i++){
		next_bytes[i] = 0;
	}
	for(size_t i = 0; i < seq.size(); i++){
		seq[i].byte_id = next_bytes[seq[i].column_id]++;
	}

	return seq;
}

void BytewiseScan::Scan(BitVector* bitvector){
	// initialize variables reference to frequently used values
	size_t num_blocks = conjunctions_[0].column->GetNumBlocks();
	assert(num_blocks == bitvector->GetNumBlocks());
	size_t num_cols = conjunctions_.size();
	size_t* num_bytes = (size_t*)malloc(num_cols * sizeof(size_t));
	for(size_t i = 0; i < num_cols; i++){
		num_bytes[i] = conjunctions_[i].num_bytes;
	}

	// initailize Avx mask for each byte in each column
	// std::vector<std::vector<AvxUnit>> mask_byte;
	AvxUnit** mask_byte = (AvxUnit**)malloc(num_cols * sizeof(AvxUnit*));
	for(size_t col = 0; col < num_cols; col++){
		// std::vector<AvxUnit> col_mask_byte;
		mask_byte[col] = (AvxUnit*)malloc(num_bytes[col] * sizeof(AvxUnit));
		WordUnit lit = conjunctions_[col].literal;
		size_t num_bits_shift = 8 * num_bytes[col] - conjunctions_[col].column->bit_width();
		lit <<= num_bits_shift;

		for(size_t byte = 0; byte < num_bytes[col]; byte++){
			ByteUnit lit_byte = FLIP(static_cast<ByteUnit>(lit >> 8*(num_bytes[col] - 1 - byte)));
			// mask_byte[col][byte] = avx_set1(lit_byte);
			AvxUnit avx_mask = avx_set1(lit_byte);
	        _mm256_storeu_si256(&mask_byte[col][byte], avx_mask);
	        // col_mask_byte.push_back(_mm256_set1_epi8(static_cast<int8_t>(lit_byte)));
		}
		// mask_byte.push_back(col_mask_byte);
	}


	// do the scanning job
#pragma omp parallel for schedule(dynamic)
    for(size_t block_id = 0; block_id < num_blocks; block_id++){
    	BitVectorBlock* bvblk = bitvector->GetBVBlock(block_id);

    	for(size_t offset = 0, bv_word_id = 0; offset < bvblk->num(); offset += kNumWordBits, bv_word_id++){
	        WordUnit bitvector_word = WordUnit(0);
	        for(size_t i = 0; i < kNumWordBits; i += kNumAvxBits/8){
	        	// initialize Avx mask for less, greater and equal results
				AvxUnit* m_less = (AvxUnit*)malloc(num_cols * sizeof(AvxUnit));
				AvxUnit* m_greater = (AvxUnit*)malloc(num_cols * sizeof(AvxUnit));
				AvxUnit* m_equal = (AvxUnit*)malloc(num_cols * sizeof(AvxUnit));
				for(size_t j = 0; j < num_cols; j++){
					AvxUnit m_zero = avx_zero();
					AvxUnit m_ones = avx_ones();
					// m_less[i] = avx_zero();
					// m_greater[i] = avx_zero();
					// m_equal[i] = avx_ones();
					_mm256_storeu_si256(&m_less[j], m_zero);
					_mm256_storeu_si256(&m_greater[j], m_zero);
					_mm256_storeu_si256(&m_equal[j], m_ones);
				}

	        	// scan each byte in the specified sequence
	        	for(size_t j = 0; j < sequence_.size(); j++){
	        		AvxUnit input_mask = avx_zero();
	        		for(size_t k = 0; k < num_cols; k++){
	        			input_mask = avx_or(input_mask, _mm256_lddqu_si256(&m_equal[k]));
	        		} 

	        		// std::cout << "Column# " <<  sequence_[j].column_id << ", "
	        		// 	<< "Byte#" << sequence_[j].byte_id << ": "
	        		// 	<< std::bitset<32>(_mm256_movemask_epi8(input_mask)) << std::endl;
	        		
	        		if(avx_iszero(input_mask))
	        			break;
					size_t col = sequence_[j].column_id;
	        		size_t byte = sequence_[j].byte_id;
	        		ColumnBlock* col_block = conjunctions_[col].column->GetBlock(block_id);
	        		col_block->Prefetch(byte, offset + i, kPrefetchDistance);

	        		AvxUnit avx_data = col_block->GetAvxUnit(offset + i, byte);
	        		AvxUnit avx_lit = _mm256_lddqu_si256(&mask_byte[col][byte]);
	        		AvxUnit avx_less = _mm256_lddqu_si256(&m_less[col]);
	        		AvxUnit avx_greater = _mm256_lddqu_si256(&m_greater[col]);
	        		AvxUnit avx_equal = _mm256_lddqu_si256(&m_equal[col]);
	        		ScanKernel(conjunctions_[col].comparator,
	        			avx_data,
    					avx_lit,
    					avx_less,
    					avx_greater,
    					avx_equal);
	        		_mm256_storeu_si256(&m_less[col], avx_less);
	        		_mm256_storeu_si256(&m_greater[col], avx_greater);
	        		_mm256_storeu_si256(&m_equal[col], avx_equal);
	        	}

	        	// get columnar result, and combine to get the final result
	        	uint32_t m_result = -1U;
	        	for(size_t col = 0; col < num_cols; col++){
	        		uint32_t m_col_result = 0U;
	        		uint32_t m_col_less, m_col_greater, m_col_equal;
	        		switch(conjunctions_[col].comparator){
	        			case Comparator::kEqual:
	        				m_col_equal = _mm256_movemask_epi8(_mm256_lddqu_si256(&m_equal[col]));
	        				m_col_result = m_col_equal;
	        				break;
	        			case Comparator::kInequal:
	        				m_col_equal = _mm256_movemask_epi8(_mm256_lddqu_si256(&m_equal[col]));
	        				m_col_result = ~m_col_equal;
	        				break;
	        			case Comparator::kLess:
	        				m_col_less = _mm256_movemask_epi8(_mm256_lddqu_si256(&m_less[col]));
	        				m_col_result = m_col_less;
	        				break;
	        			case Comparator::kLessEqual:
	        				m_col_less = _mm256_movemask_epi8(_mm256_lddqu_si256(&m_less[col]));
	        				m_col_equal = _mm256_movemask_epi8(_mm256_lddqu_si256(&m_equal[col]));
	        				m_col_result = m_col_less | m_col_equal;
	        				break;
	        			case Comparator::kGreater:
	        				m_col_greater = _mm256_movemask_epi8(_mm256_lddqu_si256(&m_greater[col]));
	        				m_col_result = m_col_greater;
	        				break;
	        			case Comparator::kGreaterEqual:
	        				m_col_greater = _mm256_movemask_epi8(_mm256_lddqu_si256(&m_greater[col]));
	        				m_col_equal = _mm256_movemask_epi8(_mm256_lddqu_si256(&m_equal[col]));
	        				m_col_result = m_col_greater | m_col_equal;
	        				break;

	        		}
	        		m_result &= m_col_result;
	        	}
	        	bitvector_word |= (static_cast<WordUnit>(m_result) << i);
	        }
	        bvblk->SetWordUnit(bitvector_word, bv_word_id);
    	}
    	bvblk->ClearTail();
	}
}

void BytewiseScan::ScanColumnwise(BitVector* bitvector){
	//set all byte mask vectors
	size_t num_tuples = conjunctions_[0].column->num_tuples();	
	ByteMaskVector* input_mask = new ByteMaskVector(num_tuples);
	// input_mask->SetAllTrue();
	std::vector<ByteMaskVector*> bm_less;
	std::vector<ByteMaskVector*> bm_greater;
	std::vector<ByteMaskVector*> bm_equal;

    for(size_t i = 0; i < conjunctions_.size(); i++){
        ByteMaskVector* new_bm_less = new ByteMaskVector(num_tuples);
        ByteMaskVector* new_bm_greater = new ByteMaskVector(num_tuples);
        ByteMaskVector* new_bm_equal = new ByteMaskVector(num_tuples);
        new_bm_less->SetAllFalse();
        new_bm_greater->SetAllFalse();
        // new_bm_equal->SetAllTrue();
        bm_less.push_back(new_bm_less);
        bm_greater.push_back(new_bm_greater);
        bm_equal.push_back(new_bm_equal);
    }

    //Scan in columnwise approach, but using ByteMask as intermediate.
    for(size_t i = 0; i < sequence_.size(); i++){
    	size_t column_id = sequence_[i].column_id;
    	size_t byte_id = sequence_[i].byte_id;
    	size_t num_bytes = conjunctions_[column_id].num_bytes;
    	const Column* column = conjunctions_[column_id].column;
    	Comparator comparator = conjunctions_[column_id].comparator;
    	size_t num_bits_shift = 8 * num_bytes - column->bit_width();
    	WordUnit literal = conjunctions_[column_id].literal;
		literal <<= num_bits_shift;
		ByteUnit byte_literal = static_cast<ByteUnit>(literal >> 8*(num_bytes - 1 - byte_id));

    	column->ScanByte(
    			byte_id,
    			comparator,
    			byte_literal,
    			bm_less[column_id],
    			bm_greater[column_id],
    			bm_equal[column_id],
    			input_mask);

    	//re-calculate input mask
    	input_mask->SetAllFalse();
    	for(size_t j = 0; j < conjunctions_.size(); j++){
    		input_mask->Or(bm_equal[j]);
    	}
    }

    //Calculate condensed result
    bitvector->SetOnes();
    for(size_t i = 0; i < conjunctions_.size(); i++){
    	Comparator comparator = conjunctions_[i].comparator;
    	BitVector* col_result = new BitVector(num_tuples);

    	switch(comparator){
    		case Comparator::kEqual:
    			bm_equal[i]->Condense(col_result);
    			break;
    		case Comparator::kInequal:
    			bm_equal[i]->Condense(col_result);
    			col_result->Not();
    			break;
    		case Comparator::kLess:
    			bm_less[i]->Condense(col_result);
    			break;
    		case Comparator::kLessEqual:
    			bm_less[i]->Condense(col_result);
    			bm_equal[i]->Condense(col_result, Bitwise::kOr);
    			break;
    		case Comparator::kGreater:
    			bm_greater[i]->Condense(col_result);
    			break;
    		case Comparator::kGreaterEqual:
    			bm_greater[i]->Condense(col_result);
    			bm_equal[i]->Condense(col_result, Bitwise::kOr);
    			break;
    	}

    	bitvector->And(col_result);
    }

    //free byte mask vectors
    delete input_mask;
    for(size_t i = 0; i < conjunctions_.size(); i++){
    	delete bm_less[i];
    	delete bm_greater[i];
    	delete bm_equal[i];
    }
}

inline void BytewiseScan::ScanKernel(Comparator comparator,
		const AvxUnit &byteslice1, const AvxUnit &byteslice2,
        AvxUnit &mask_less, AvxUnit &mask_greater, AvxUnit &mask_equal) const{
	 switch(comparator){

        case Comparator::kEqual:
        case Comparator::kInequal:
            _mm256_storeu_si256(&mask_equal,
                avx_and(mask_equal, avx_cmpeq<ByteUnit>(byteslice1, byteslice2)));
            break;
        case Comparator::kLess:
        case Comparator::kLessEqual:
            _mm256_storeu_si256(&mask_less, 
                avx_or(mask_less, avx_and(mask_equal, avx_cmplt<ByteUnit>(byteslice1, byteslice2))));
            _mm256_storeu_si256(&mask_equal, 
                avx_and(mask_equal, avx_cmpeq<ByteUnit>(byteslice1, byteslice2)));
            break;
        case Comparator::kGreater:
        case Comparator::kGreaterEqual:
            _mm256_storeu_si256(&mask_greater,
                avx_or(mask_greater, avx_and(mask_equal, avx_cmpgt<ByteUnit>(byteslice1, byteslice2))));
            _mm256_storeu_si256(&mask_equal, 
                avx_and(mask_equal, avx_cmpeq<ByteUnit>(byteslice1, byteslice2)));
            break;
    }

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
