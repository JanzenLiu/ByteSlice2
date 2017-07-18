#include    "include/superscalar2_byteslice_column_block.h"
#include    <cstdlib>
#include    <cstring>
#include    <include/avx-utility.h>

namespace byteslice{
    
#ifdef      NEARLYSTOP
#warning    "Early-stop is disabled in Superscalar2ByteSliceColumnBlock!"
#endif

static constexpr size_t kPrefetchDistance = 512*2;

template <size_t BIT_WIDTH, Direction PDIRECTION>
Superscalar2ByteSliceColumnBlock<BIT_WIDTH, PDIRECTION>::Superscalar2ByteSliceColumnBlock(size_t num):
    ColumnBlock(
            PDIRECTION==Direction::kLeft ? 
                ColumnType::kByteSlicePadLeft:ColumnType::kByteSlicePadRight, 
            BIT_WIDTH, 
            num)    
{
    //allocate memory space
    assert(num <= kNumTuplesPerBlock);
    for(size_t i=0; i < kNumBytesPerCode; i++){
        size_t ret = posix_memalign((void**)&data_[i], 32, kMemSizePerSuperscalar2ByteSlice);                    
        ret = ret;
        memset(data_[i], 0x0, kMemSizePerSuperscalar2ByteSlice);
    }

}

template <size_t BIT_WIDTH, Direction PDIRECTION>
Superscalar2ByteSliceColumnBlock<BIT_WIDTH, PDIRECTION>::~Superscalar2ByteSliceColumnBlock(){
    for(size_t i=0; i < kNumBytesPerCode; i++){
        free(data_[i]);
    }
}

template <size_t BIT_WIDTH, Direction PDIRECTION>
bool Superscalar2ByteSliceColumnBlock<BIT_WIDTH, PDIRECTION>::Resize(size_t num){
    num_tuples_ = num;
    return true;
}

template <size_t BIT_WIDTH, Direction PDIRECTION>
void Superscalar2ByteSliceColumnBlock<BIT_WIDTH, PDIRECTION>::
                    SerToFile(SequentialWriteBinaryFile &file) const{
    file.Append(&num_tuples_, sizeof(num_tuples_));
    for(size_t byte_id = 0; byte_id < kNumBytesPerCode; byte_id++){
        file.Append(data_[byte_id], kMemSizePerSuperscalar2ByteSlice);
    }
}

template <size_t BIT_WIDTH, Direction PDIRECTION>
void Superscalar2ByteSliceColumnBlock<BIT_WIDTH, PDIRECTION>::
                    DeserFromFile(const SequentialReadBinaryFile &file){
    file.Read(&num_tuples_, sizeof(num_tuples_));
    for(size_t byte_id = 0; byte_id < kNumBytesPerCode; byte_id++){
        file.Read(data_[byte_id], kMemSizePerSuperscalar2ByteSlice);
    }
}

//Scan that takes in and output ByteMask
template <size_t BIT_WIDTH, Direction PDIRECTION>
void Superscalar2ByteSliceColumnBlock<BIT_WIDTH, PDIRECTION>::Scan(Comparator comparator,
        WordUnit literal, ByteMaskBlock* bmblk, Bitwise opt) const{
    assert(bmblk->num() == num_tuples_);
    switch(comparator){
        case Comparator::kLess:
            return ScanHelper1<Comparator::kLess>(literal, bmblk, opt);
        case Comparator::kGreater:
            return ScanHelper1<Comparator::kGreater>(literal, bmblk, opt);
        case Comparator::kLessEqual:
            return ScanHelper1<Comparator::kLessEqual>(literal, bmblk, opt);
        case Comparator::kGreaterEqual:
            return ScanHelper1<Comparator::kGreaterEqual>(literal, bmblk, opt);
        case Comparator::kEqual:
            return ScanHelper1<Comparator::kEqual>(literal, bmblk, opt);
        case Comparator::kInequal:
            return ScanHelper1<Comparator::kInequal>(literal, bmblk, opt);
    }
}

template <size_t BIT_WIDTH, Direction PDIRECTION>
template <Comparator CMP>
void Superscalar2ByteSliceColumnBlock<BIT_WIDTH, PDIRECTION>::ScanHelper1(WordUnit literal,
        ByteMaskBlock* bmblk, Bitwise opt) const{
     switch(opt){
        case Bitwise::kSet:
            return ScanHelper2<CMP, Bitwise::kSet>(literal, bmblk);
        case Bitwise::kAnd:
            return ScanHelper2<CMP, Bitwise::kAnd>(literal, bmblk);
        case Bitwise::kOr:
            return ScanHelper2<CMP, Bitwise::kOr>(literal, bmblk);
    }
}

template <size_t BIT_WIDTH, Direction PDIRECTION>
template <Comparator CMP, Bitwise OPT>
void Superscalar2ByteSliceColumnBlock<BIT_WIDTH, PDIRECTION>::ScanHelper2(WordUnit literal,
        ByteMaskBlock* bmblk) const{

    //Do the real work here 
    //Prepare byte-slices of literal
    AvxUnit mask_literal[kNumBytesPerCode];
    literal &= kCodeMask;
    if(Direction::kRight == PDIRECTION){
        literal <<= kNumPaddingBits;
    }
    for(size_t byte_id=0; byte_id < kNumBytesPerCode; byte_id++){
         ByteUnit byte = FLIP(static_cast<ByteUnit>(literal >> 8*(kNumBytesPerCode - 1 - byte_id)));
         mask_literal[byte_id] = avx_set1<ByteUnit>(byte);
    }
    
    //For every 32 tuple which relates to 256-bit mask
    for(size_t offset = 0; offset < num_tuples_; offset += sizeof(AvxUnit)){

        AvxUnit m_less = avx_zero();
        AvxUnit m_greater = avx_zero();
        AvxUnit m_equal; 

        switch(OPT){
            case Bitwise::kSet:
                m_equal = avx_ones();
                break;
            case Bitwise::kAnd:
                m_equal = bmblk->GetAvxUnit(offset);
                break;
            case Bitwise::kOr:
                m_equal = avx_not(bmblk->GetAvxUnit(offset));
                break;
        }

        if((OPT==Bitwise::kSet) || !avx_iszero(m_equal)){
            __builtin_prefetch(data_[0] + offset + kPrefetchDistance);
            ScanKernel2<CMP, 0>(
                    _mm256_lddqu_si256(reinterpret_cast<__m256i*>(data_[0]+offset)),
                    mask_literal[0],
                    m_less,
                    m_greater,
                    m_equal);
            if(kNumBytesPerCode > 1 && /*((OPT==Bitwise::kSet) ||*/ !avx_iszero(m_equal)/*)*/){
                //pruning after 2-nd BitSlice is not fruitful (when OPT is kSet)
                //(possibly due to branch miss penalty)
                //eliminating this pruning brings ~10% improvement
                __builtin_prefetch(data_[1] + offset + kPrefetchDistance);
                ScanKernel2<CMP, 1>(
                        _mm256_lddqu_si256(reinterpret_cast<__m256i*>(data_[1]+offset)),
                        mask_literal[1],
                        m_less,
                        m_greater,
                        m_equal);
                if(kNumBytesPerCode > 2 && !avx_iszero(m_equal)){
                    ScanKernel2<CMP, 2>(
                            _mm256_lddqu_si256(reinterpret_cast<__m256i*>(data_[2]+offset)),
                            mask_literal[2],
                            m_less,
                            m_greater,
                            m_equal);
                    if(kNumBytesPerCode > 3 && !avx_iszero(m_equal)){
                        ScanKernel2<CMP, 3>(
                                _mm256_lddqu_si256(reinterpret_cast<__m256i*>(data_[3]+offset)),
                                mask_literal[3],
                                m_less,
                                m_greater,
                                m_equal);
                    }
                }
            }
        }

        AvxUnit m_result;
        switch(CMP){
            case Comparator::kLessEqual:
                m_result = avx_or(m_less, m_equal);
                break;
            case Comparator::kLess:
                m_result = m_less;
                break;
            case Comparator::kGreaterEqual:
                m_result = avx_or(m_greater, m_equal);
                break;
            case Comparator::kGreater:
                m_result = m_greater;
                break;
            case Comparator::kEqual:
                m_result = m_equal;
                break;
            case Comparator::kInequal:
                m_result = avx_not(m_equal);
                break;
        }
        //NO NEED to movemask!
        AvxUnit x = m_result;
        switch(OPT){
            case Bitwise::kSet:
                break;
            case Bitwise::kAnd:
                x = avx_and(x, bmblk->GetAvxUnit(offset));
                break;
            case Bitwise::kOr:
                x = avx_or(x, bmblk->GetAvxUnit(offset));
                break;
        }

        bmblk->SetAvxUnit(offset, x);
    }
    //bmblk->ClearTail();

}



//Scan against literal
template <size_t BIT_WIDTH, Direction PDIRECTION>
void Superscalar2ByteSliceColumnBlock<BIT_WIDTH, PDIRECTION>::Scan(Comparator comparator,
        WordUnit literal, BitVectorBlock* bvblock, Bitwise bit_opt) const{
    assert(bvblock->num() == num_tuples_);
    switch(comparator){
        case Comparator::kLess:
            return ScanHelper1<Comparator::kLess>(literal, bvblock, bit_opt);
        case Comparator::kGreater:
            return ScanHelper1<Comparator::kGreater>(literal, bvblock, bit_opt);
        case Comparator::kLessEqual:
            return ScanHelper1<Comparator::kLessEqual>(literal, bvblock, bit_opt);
        case Comparator::kGreaterEqual:
            return ScanHelper1<Comparator::kGreaterEqual>(literal, bvblock, bit_opt);
        case Comparator::kEqual:
            return ScanHelper1<Comparator::kEqual>(literal, bvblock, bit_opt);
        case Comparator::kInequal:
            return ScanHelper1<Comparator::kInequal>(literal, bvblock, bit_opt);
    }
}

template <size_t BIT_WIDTH, Direction PDIRECTION>
template <Comparator CMP>
void Superscalar2ByteSliceColumnBlock<BIT_WIDTH, PDIRECTION>::ScanHelper1(WordUnit literal,
                                    BitVectorBlock* bvblock, Bitwise bit_opt) const{
     switch(bit_opt){
        case Bitwise::kSet:
            return ScanHelper2<CMP, Bitwise::kSet>(literal, bvblock);
        case Bitwise::kAnd:
            return ScanHelper2<CMP, Bitwise::kAnd>(literal, bvblock);
        case Bitwise::kOr:
            return ScanHelper2<CMP, Bitwise::kOr>(literal, bvblock);
    }
}

template <size_t BIT_WIDTH, Direction PDIRECTION>
template <Comparator CMP, Bitwise OPT>
void Superscalar2ByteSliceColumnBlock<BIT_WIDTH, PDIRECTION>::ScanHelper2(WordUnit literal,
                                            BitVectorBlock* bvblock) const {
    //Prepare byte-slices of literal
    AvxUnit mask_literal[kNumBytesPerCode];
    literal &= kCodeMask;
    if(Direction::kRight == PDIRECTION){
        literal <<= kNumPaddingBits;
    }
    for(size_t byte_id=0; byte_id < kNumBytesPerCode; byte_id++){
         ByteUnit byte = FLIP(static_cast<ByteUnit>(literal >> 8*(kNumBytesPerCode - 1 - byte_id)));
         mask_literal[byte_id] = avx_set1<ByteUnit>(byte);
    }
    
    //for every kNumWordBits (64) tuples
    for(size_t offset = 0, bv_word_id = 0; offset < num_tuples_; offset += kNumWordBits, bv_word_id++){
        WordUnit bitvector_word = WordUnit(0);
        
		//XU: simulating 256*2-bit SIMD
        //for(size_t i=0; i < kNumWordBits; i += kNumAvxBits/8){
        AvxUnit m_less1 = avx_zero();
		AvxUnit m_less2 = avx_zero();
        AvxUnit m_greater1 = avx_zero();
		AvxUnit m_greater2 = avx_zero();
        AvxUnit m_equal1 = avx_zero();
		AvxUnit m_equal2 = avx_zero();

        int input_mask1 = static_cast<int>(-1ULL);
		int input_mask2 = static_cast<int>(-1ULL);

        switch(OPT){
            case Bitwise::kSet:
                m_equal1 = avx_ones();
				m_equal2 = avx_ones();	
                break;
            case Bitwise::kAnd:
                input_mask1 = static_cast<int>(bvblock->GetWordUnit(bv_word_id));
                input_mask2 = static_cast<int>(bvblock->GetWordUnit(bv_word_id) >> 32);
                m_equal1 = avx_ones();
				m_equal2 = avx_ones();
                break;
            case Bitwise::kOr:
                input_mask1 = ~static_cast<int>(bvblock->GetWordUnit(bv_word_id));
                input_mask2 = ~static_cast<int>(bvblock->GetWordUnit(bv_word_id) >> 32);
                m_equal1 = avx_ones();
				m_equal2 = avx_ones();
                break;
        }

            if(
#ifndef         NEARLYSTOP
                (OPT==Bitwise::kSet) || !((0==input_mask1) && (0==input_mask2))
#else           
                true
#endif
              ){
                __builtin_prefetch(data_[0] + offset + 1536);	//modified

				//scan the first 32 tuples (a AvxUnit)
                ScanKernel2<CMP, 0>(
                        _mm256_lddqu_si256(reinterpret_cast<__m256i*>(data_[0]+offset)),
                        mask_literal[0],
                        m_less1,
                        m_greater1,
                        m_equal1);

				//scan the next 32 tuples
                ScanKernel2<CMP, 0>(
                        _mm256_lddqu_si256(reinterpret_cast<__m256i*>(data_[0]+offset+32)),
                        mask_literal[0],
                        m_less2,
                        m_greater2,
                        m_equal2);
				

                if(kNumBytesPerCode > 1
#ifndef                 NEARLYSTOP
                        && ((OPT==Bitwise::kSet && !(avx_iszero(m_equal1) && avx_iszero(m_equal2)) )
                            || (OPT!=Bitwise::kSet && 
								!(0==(input_mask1 & _mm256_movemask_epi8(m_equal1)) && 0==(input_mask2 & _mm256_movemask_epi8(m_equal2)) )))

#endif
                  ){
                    __builtin_prefetch(data_[1] + offset + 1536);

					//scan the first 32 tuples
                    ScanKernel2<CMP, 1>(
                            _mm256_lddqu_si256(reinterpret_cast<__m256i*>(data_[1]+offset)),
                            mask_literal[1],
                            m_less1,
                            m_greater1,
                            m_equal1);


					//scan the next 32 tuples
                    ScanKernel2<CMP, 1>(
                            _mm256_lddqu_si256(reinterpret_cast<__m256i*>(data_[1]+offset+32)),
                            mask_literal[1],
                            m_less2,
                            m_greater2,
                            m_equal2);

                    if(kNumBytesPerCode > 2
#ifndef                     NEARLYSTOP
                            && ((OPT==Bitwise::kSet && !(avx_iszero(m_equal1) && avx_iszero(m_equal2))) 
                                || (OPT!=Bitwise::kSet && 
									!(0==(input_mask1 & _mm256_movemask_epi8(m_equal1)) && 0==(input_mask2 & _mm256_movemask_epi8(m_equal2)) )))
#endif
                      ){
						//scan the first 32 tuples
                        ScanKernel2<CMP, 2>(
                                _mm256_lddqu_si256(reinterpret_cast<__m256i*>(data_[2]+offset)),
                                mask_literal[2],
                                m_less1,
                                m_greater1,
                                m_equal1);

						//scan the next 32 tuples
                        ScanKernel2<CMP, 2>(
                                _mm256_lddqu_si256(reinterpret_cast<__m256i*>(data_[2]+offset+32)),
                                mask_literal[2],
                                m_less2,
                                m_greater2,
                                m_equal2);
                        if(kNumBytesPerCode > 3
#ifndef                         NEARLYSTOP
                            && ((OPT==Bitwise::kSet && !(avx_iszero(m_equal1) && avx_iszero(m_equal2))) 
                                || (OPT!=Bitwise::kSet && 
									!(0==(input_mask1 & _mm256_movemask_epi8(m_equal1)) && 0==(input_mask2 & _mm256_movemask_epi8(m_equal2)) )))
#endif
                          ){
							//scan the first 32 tuples
                            ScanKernel2<CMP, 3>(
                                    _mm256_lddqu_si256(reinterpret_cast<__m256i*>(data_[3]+offset)),
                                    mask_literal[3],
                                    m_less1,
                                    m_greater1,
                                    m_equal1);
							
							//scan the next 32 tuples
                            ScanKernel2<CMP, 3>(
                                    _mm256_lddqu_si256(reinterpret_cast<__m256i*>(data_[3]+offset+32)),
                                    mask_literal[3],
                                    m_less2,
                                    m_greater2,
                                    m_equal2);


                        }
                    }
                }
            }
			
        //put result bitvector into bitvector block
		/*	
        switch(CMP){
            case Comparator::kLessEqual:
                bitvector_word= (static_cast<WordUnit>(_mm256_movemask_epi8(avx_or(m_less1, m_equal1)))) | 
					(static_cast<WordUnit>(_mm256_movemask_epi8(avx_or(m_less2, m_equal2))) << 32);
                break;
            case Comparator::kLess:
                bitvector_word= (static_cast<WordUnit>((uint32_t)_mm256_movemask_epi8(m_less1))) | 
					(static_cast<WordUnit>((uint32_t)_mm256_movemask_epi8(m_less2)) << 32);
                break;
            case Comparator::kGreaterEqual:
                bitvector_word= (static_cast<WordUnit>(_mm256_movemask_epi8(avx_or(m_greater1, m_equal1)))) | 
					(static_cast<WordUnit>(_mm256_movemask_epi8(avx_or(m_greater2, m_equal2))) << 32);
                break;
            case Comparator::kGreater:
                bitvector_word= (static_cast<WordUnit>(_mm256_movemask_epi8(m_greater1))) | 
					(static_cast<WordUnit>(_mm256_movemask_epi8(m_greater2)) << 32);
                break;
            case Comparator::kEqual:
                bitvector_word= (static_cast<WordUnit>(_mm256_movemask_epi8(m_equal1))) | 
					(static_cast<WordUnit>(_mm256_movemask_epi8(m_equal2)) << 32);
                break;
            case Comparator::kInequal:
                bitvector_word= (static_cast<WordUnit>(_mm256_movemask_epi8(avx_not(m_equal1)))) | 
					(static_cast<WordUnit>(_mm256_movemask_epi8(avx_not(m_equal2))) << 32);
            break;
        }
		*/
		AvxUnit m_result1;
		AvxUnit m_result2;
		switch(CMP) {
			case Comparator::kLessEqual:
				m_result1 = avx_or(m_less1, m_equal1);
				m_result2 = avx_or(m_less2, m_equal2);
				break;
			case Comparator::kLess:
				m_result1 = m_less1;
				m_result2 = m_less2;
				break;
			case Comparator::kGreaterEqual:
				m_result1 = avx_or(m_greater1, m_equal1);
				m_result2 = avx_or(m_greater2, m_equal2);
				break;
			case Comparator::kGreater:
				m_result1 = m_greater1;
				m_result2 = m_greater2;
				break;
			case Comparator::kEqual:
				m_result1 = m_equal1;
				m_result2 = m_equal2;
				break;
			case Comparator::kInequal:
				m_result1 = avx_not(m_equal1);
				m_result2 = avx_not(m_equal2);
				break;
		}
		uint32_t mmask1 = _mm256_movemask_epi8(m_result1);
		uint32_t mmask2 = _mm256_movemask_epi8(m_result2);
		
		bitvector_word |= (static_cast<WordUnit>(mmask1));
		bitvector_word |= (static_cast<WordUnit>(mmask2) << 32);

        //size_t bv_word_id = offset / kNumWordBits;
        WordUnit x = bitvector_word;
        switch(OPT){
            case Bitwise::kSet:
                break;
            case Bitwise::kAnd:
                x &= bvblock->GetWordUnit(bv_word_id);
                break;
            case Bitwise::kOr:
                x |= bvblock->GetWordUnit(bv_word_id);
                break;
        }
        bvblock->SetWordUnit(x, bv_word_id);
    }
    bvblock->ClearTail();
}

//Scan against other block
template <size_t BIT_WIDTH, Direction PDIRECTION>
void Superscalar2ByteSliceColumnBlock<BIT_WIDTH, PDIRECTION>::Scan(Comparator comparator,
        const ColumnBlock* other_block, BitVectorBlock* bvblock, Bitwise bit_opt) const{

    assert(bvblock->num() == num_tuples_);
    assert(other_block->num_tuples() == num_tuples_);
    assert(other_block->type() == type_);
    assert(other_block->bit_width() == bit_width_);

    const Superscalar2ByteSliceColumnBlock<BIT_WIDTH, PDIRECTION>* block2 =
        static_cast<const Superscalar2ByteSliceColumnBlock<BIT_WIDTH, PDIRECTION>*>(other_block);
    
    //multiplexing
    switch(comparator){
        case Comparator::kLess:
            return ScanHelper1<Comparator::kLess>(block2, bvblock, bit_opt);
        case Comparator::kGreater:
            return ScanHelper1<Comparator::kGreater>(block2, bvblock, bit_opt);
        case Comparator::kLessEqual:
            return ScanHelper1<Comparator::kLessEqual>(block2, bvblock, bit_opt);
        case Comparator::kGreaterEqual:
            return ScanHelper1<Comparator::kGreaterEqual>(block2, bvblock, bit_opt);
        case Comparator::kEqual:
            return ScanHelper1<Comparator::kEqual>(block2, bvblock, bit_opt);
        case Comparator::kInequal:
            return ScanHelper1<Comparator::kInequal>(block2, bvblock, bit_opt);
    }
}

template <size_t BIT_WIDTH, Direction PDIRECTION>
template <Comparator CMP>
void Superscalar2ByteSliceColumnBlock<BIT_WIDTH, PDIRECTION>::ScanHelper1(
                            const Superscalar2ByteSliceColumnBlock<BIT_WIDTH, PDIRECTION>* other_block,
                            BitVectorBlock* bvblock, 
                            Bitwise bit_opt) const {
    switch(bit_opt){
        case Bitwise::kSet:
            return ScanHelper2<CMP, Bitwise::kSet>(other_block, bvblock);
        case Bitwise::kAnd:
            return ScanHelper2<CMP, Bitwise::kAnd>(other_block, bvblock);
        case Bitwise::kOr:
            return ScanHelper2<CMP, Bitwise::kOr>(other_block, bvblock);
    }
}

template <size_t BIT_WIDTH, Direction PDIRECTION>
template <Comparator CMP, Bitwise OPT>
void Superscalar2ByteSliceColumnBlock<BIT_WIDTH, PDIRECTION>::ScanHelper2(
                            const Superscalar2ByteSliceColumnBlock<BIT_WIDTH, PDIRECTION>* other_block,
                            BitVectorBlock* bvblock) const {

    //for every kNumWordBits (64) tuples
    for(size_t offset = 0, bv_word_id = 0; offset < num_tuples_; offset += kNumWordBits, bv_word_id++){
        WordUnit bitvector_word = WordUnit(0);
        //need several iteration of AVX scan
        //for(size_t i=0; i < kNumWordBits; i += kNumAvxBits/8){
        AvxUnit m_less1 = avx_zero();
		AvxUnit m_less2 = avx_zero();
        AvxUnit m_greater1 = avx_zero();
		AvxUnit m_greater2 = avx_zero();
        AvxUnit m_equal1;
	    AvxUnit m_equal2;
        
		int input_mask1 = static_cast<int>(-1ULL);
		int input_mask2 = static_cast<int>(-1ULL);

		switch(OPT){
            case Bitwise::kSet:
                m_equal1 = avx_ones();
				m_equal2 = avx_ones();
                break;
            case Bitwise::kAnd:
                input_mask1 = static_cast<int>(bvblock->GetWordUnit(bv_word_id));
				input_mask2 = static_cast<int>(bvblock->GetWordUnit(bv_word_id) >> 32);
                m_equal1 = avx_ones();
				m_equal2 = avx_ones();
                break;
            case Bitwise::kOr:
                input_mask1 = ~static_cast<int>(bvblock->GetWordUnit(bv_word_id));
				input_mask2 = ~static_cast<int>(bvblock->GetWordUnit(bv_word_id) >> 32);
                m_equal1 = avx_ones();
				m_equal2 = avx_ones();
                break;
        }

            if((OPT==Bitwise::kSet) ||  !((0==input_mask1) && (0==input_mask2))){
                __builtin_prefetch(data_[0] + offset + 1024);
                __builtin_prefetch(other_block->data_[0] + offset + 1024);

				//scan the first 32 tuples (a AvxUnit)
                ScanKernel2<CMP, 0>(
                        _mm256_lddqu_si256(reinterpret_cast<__m256i*>(data_[0]+offset)),
                        _mm256_lddqu_si256(reinterpret_cast<__m256i*>(other_block->data_[0]+offset)),
                        m_less1,
                        m_greater1,
                        m_equal1);

				//scan the next 32 tuples
                ScanKernel2<CMP, 0>(
                        _mm256_lddqu_si256(reinterpret_cast<__m256i*>(data_[0]+offset+32)),
                        _mm256_lddqu_si256(reinterpret_cast<__m256i*>(other_block->data_[0]+offset+32)),
                        m_less2,
                        m_greater2,
                        m_equal2);

                if(kNumBytesPerCode > 1 && 
                        ((OPT==Bitwise::kSet && !(avx_iszero(m_equal1) && avx_iszero(m_equal2))) 
                        || (OPT!=Bitwise::kSet && 
								!(0==(input_mask1 & _mm256_movemask_epi8(m_equal1)) && 0==(input_mask2 & _mm256_movemask_epi8(m_equal2)) )))){

                    __builtin_prefetch(data_[1] + offset + 1024);
                    __builtin_prefetch(other_block->data_[1] + offset  + 1024);

					//scan the first 32 tuples
                    ScanKernel2<CMP, 1>(
                            _mm256_lddqu_si256(reinterpret_cast<__m256i*>(data_[1]+offset)),
                            _mm256_lddqu_si256(reinterpret_cast<__m256i*>(other_block->data_[1]+offset)),
                            m_less1,
                            m_greater1,
                            m_equal1);
					
					//scan the next 32 tuples
                    ScanKernel2<CMP, 1>(
                            _mm256_lddqu_si256(reinterpret_cast<__m256i*>(data_[1]+offset+32)),
                            _mm256_lddqu_si256(reinterpret_cast<__m256i*>(other_block->data_[1]+offset+32)),
                            m_less2,
                            m_greater2,
                            m_equal2);


                    if(kNumBytesPerCode > 2 && 
                        ((OPT==Bitwise::kSet && !(avx_iszero(m_equal1) && avx_iszero(m_equal2))) 
                        || (OPT!=Bitwise::kSet && 
								!(0==(input_mask1 & _mm256_movemask_epi8(m_equal1)) && 0==(input_mask2 & _mm256_movemask_epi8(m_equal2)) )))){

						//scan he first 32 tuples
                        ScanKernel2<CMP, 2>(
                                _mm256_lddqu_si256(reinterpret_cast<__m256i*>(data_[2]+offset)),
                                _mm256_lddqu_si256(reinterpret_cast<__m256i*>(other_block->data_[2]+offset)),
                                m_less1,
                                m_greater1,
                                m_equal1);

						//scan the next 32 tuples
                        ScanKernel2<CMP, 2>(
                                _mm256_lddqu_si256(reinterpret_cast<__m256i*>(data_[2]+offset+32)),
                                _mm256_lddqu_si256(reinterpret_cast<__m256i*>(other_block->data_[2]+offset+32)),
                                m_less2,
                                m_greater2,
                                m_equal2);
                        if(kNumBytesPerCode > 3 && 
                        	((OPT==Bitwise::kSet && !(avx_iszero(m_equal1) && avx_iszero(m_equal2))) 
                        	|| (OPT!=Bitwise::kSet && 
									!(0==(input_mask1 & _mm256_movemask_epi8(m_equal1)) && 0==(input_mask2 & _mm256_movemask_epi8(m_equal2)) )))){
							//scan the first 32 tuples
                            ScanKernel2<CMP, 3>(
                                    _mm256_lddqu_si256(reinterpret_cast<__m256i*>(data_[3]+offset)),
                                    _mm256_lddqu_si256(reinterpret_cast<__m256i*>(other_block->data_[3]+offset)),
                                    m_less1,
                                    m_greater1,
                                    m_equal1);
							
							//scan the next 32 tuples
                            ScanKernel2<CMP, 3>(
                                    _mm256_lddqu_si256(reinterpret_cast<__m256i*>(data_[3]+offset+32)),
                                    _mm256_lddqu_si256(reinterpret_cast<__m256i*>(other_block->data_[3]+offset+32)),
                                    m_less2,
                                    m_greater2,
                                    m_equal2);
                        }
                    }
                }
            }
        
		//put result bitvector into bitvector block
		AvxUnit m_result1;
		AvxUnit m_result2;
		switch(CMP) {
			case Comparator::kLessEqual:
				m_result1 = avx_or(m_less1, m_equal1);
				m_result2 = avx_or(m_less2, m_equal2);
				break;
			case Comparator::kLess:
				m_result1 = m_less1;
				m_result2 = m_less2;
				break;
			case Comparator::kGreaterEqual:
				m_result1 = avx_or(m_greater1, m_equal1);
				m_result2 = avx_or(m_greater2, m_equal2);
				break;
			case Comparator::kGreater:
				m_result1 = m_greater1;
				m_result2 = m_greater2;
				break;
			case Comparator::kEqual:
				m_result1 = m_equal1;
				m_result2 = m_equal2;
				break;
			case Comparator::kInequal:
				m_result1 = avx_not(m_equal1);
				m_result2 = avx_not(m_equal2);
				break;
		}
		uint32_t mmask1 = _mm256_movemask_epi8(m_result1);
		uint32_t mmask2 = _mm256_movemask_epi8(m_result2);
		
		bitvector_word |= (static_cast<WordUnit>(mmask1));
		bitvector_word |= (static_cast<WordUnit>(mmask2) << 32);

        //size_t bv_word_id = offset / kNumWordBits;
        WordUnit x = bitvector_word;
        switch(OPT){
            case Bitwise::kSet:
                break;
            case Bitwise::kAnd:
                x &= bvblock->GetWordUnit(bv_word_id);
                break;
            case Bitwise::kOr:
                x |= bvblock->GetWordUnit(bv_word_id);
                break;
        }
        bvblock->SetWordUnit(x, bv_word_id);
    }
    bvblock->ClearTail();

}


//Scan Kernel
template <size_t BIT_WIDTH, Direction PDIRECTION>
template <Comparator CMP>
inline void Superscalar2ByteSliceColumnBlock<BIT_WIDTH, PDIRECTION>::ScanKernel
                                                        (const AvxUnit &byteslice1,
                                                         const AvxUnit &byteslice2,
                                                         AvxUnit &mask_less,
                                                         AvxUnit &mask_greater,
                                                         AvxUnit &mask_equal) const {
    switch(CMP){
        case Comparator::kEqual:
        case Comparator::kInequal:
            mask_equal = 
                avx_and(mask_equal, avx_cmpeq<ByteUnit>(byteslice1, byteslice2));
            break;
        case Comparator::kLess:
        case Comparator::kLessEqual:
            mask_less = 
                avx_or(mask_less, avx_and(mask_equal, avx_cmplt<ByteUnit>(byteslice1, byteslice2)));
            mask_equal = 
                avx_and(mask_equal, avx_cmpeq<ByteUnit>(byteslice1, byteslice2));
            break;
        case Comparator::kGreater:
        case Comparator::kGreaterEqual:
            mask_greater =
                avx_or(mask_greater, avx_and(mask_equal, avx_cmpgt<ByteUnit>(byteslice1, byteslice2)));
            mask_equal = 
                avx_and(mask_equal, avx_cmpeq<ByteUnit>(byteslice1, byteslice2));
            break;
    }
}

//Scan Kernel2 --- Optimized on Scan Kernel
//to remove unnecessary equal comparison for last bit slice
template <size_t BIT_WIDTH, Direction PDIRECTION>
template <Comparator CMP, size_t BYTE_ID>
inline void Superscalar2ByteSliceColumnBlock<BIT_WIDTH, PDIRECTION>::ScanKernel2
                                                        (const AvxUnit &byteslice1,
                                                         const AvxUnit &byteslice2,
                                                         AvxUnit &mask_less,
                                                         AvxUnit &mask_greater,
                                                         AvxUnit &mask_equal) const {

    //internal ByteSlice --- not last BS                                                        
    if(BYTE_ID < kNumBytesPerCode - 1){ 
        switch(CMP){
            case Comparator::kEqual:
            case Comparator::kInequal:
                mask_equal = 
                    avx_and(mask_equal, avx_cmpeq<ByteUnit>(byteslice1, byteslice2));
                break;
            case Comparator::kLess:
            case Comparator::kLessEqual:
                mask_less = 
                    avx_or(mask_less, avx_and(mask_equal, avx_cmplt<ByteUnit>(byteslice1, byteslice2)));
                mask_equal = 
                    avx_and(mask_equal, avx_cmpeq<ByteUnit>(byteslice1, byteslice2));
                break;
            case Comparator::kGreater:
            case Comparator::kGreaterEqual:
                mask_greater =
                    avx_or(mask_greater, avx_and(mask_equal, avx_cmpgt<ByteUnit>(byteslice1, byteslice2)));
                mask_equal = 
                    avx_and(mask_equal, avx_cmpeq<ByteUnit>(byteslice1, byteslice2));
                break;
        }
    }
    //last BS: no need to compute mask_equal for some comparisons
    else if(BYTE_ID == kNumBytesPerCode - 1){   
        switch(CMP){
            case Comparator::kEqual:
            case Comparator::kInequal:
                mask_equal = 
                    avx_and(mask_equal, avx_cmpeq<ByteUnit>(byteslice1, byteslice2));
                break;
            case Comparator::kLessEqual:
                mask_less = 
                    avx_or(mask_less, avx_and(mask_equal, avx_cmplt<ByteUnit>(byteslice1, byteslice2)));
                mask_equal = 
                    avx_and(mask_equal, avx_cmpeq<ByteUnit>(byteslice1, byteslice2));
                break;
            case Comparator::kLess:
                mask_less = 
                    avx_or(mask_less, avx_and(mask_equal, avx_cmplt<ByteUnit>(byteslice1, byteslice2)));
                break;
            case Comparator::kGreaterEqual:
                mask_greater =
                    avx_or(mask_greater, avx_and(mask_equal, avx_cmpgt<ByteUnit>(byteslice1, byteslice2)));
                mask_equal = 
                    avx_and(mask_equal, avx_cmpeq<ByteUnit>(byteslice1, byteslice2));
                break;
            case Comparator::kGreater:
                mask_greater =
                    avx_or(mask_greater, avx_and(mask_equal, avx_cmpgt<ByteUnit>(byteslice1, byteslice2)));
                break;
        }
    }
    //otherwise, do nothing

}


template <size_t BIT_WIDTH, Direction PDIRECTION>
void Superscalar2ByteSliceColumnBlock<BIT_WIDTH, PDIRECTION>::BulkLoadArray(const WordUnit* codes,
                                                        size_t num, size_t start_pos){
    assert(start_pos + num <= num_tuples_);
    for(size_t i = 0; i < num; i++){
        SetTuple(start_pos+i, codes[i]);
    }
}



//explicit specialization
//default padding: right
template class Superscalar2ByteSliceColumnBlock<1>;
template class Superscalar2ByteSliceColumnBlock<2>;
template class Superscalar2ByteSliceColumnBlock<3>;
template class Superscalar2ByteSliceColumnBlock<4>;
template class Superscalar2ByteSliceColumnBlock<5>;
template class Superscalar2ByteSliceColumnBlock<6>;
template class Superscalar2ByteSliceColumnBlock<7>;
template class Superscalar2ByteSliceColumnBlock<8>;
template class Superscalar2ByteSliceColumnBlock<9>;
template class Superscalar2ByteSliceColumnBlock<10>;
template class Superscalar2ByteSliceColumnBlock<11>;
template class Superscalar2ByteSliceColumnBlock<12>;
template class Superscalar2ByteSliceColumnBlock<13>;
template class Superscalar2ByteSliceColumnBlock<14>;
template class Superscalar2ByteSliceColumnBlock<15>;
template class Superscalar2ByteSliceColumnBlock<16>;
template class Superscalar2ByteSliceColumnBlock<17>;
template class Superscalar2ByteSliceColumnBlock<18>;
template class Superscalar2ByteSliceColumnBlock<19>;
template class Superscalar2ByteSliceColumnBlock<20>;
template class Superscalar2ByteSliceColumnBlock<21>;
template class Superscalar2ByteSliceColumnBlock<22>;
template class Superscalar2ByteSliceColumnBlock<23>;
template class Superscalar2ByteSliceColumnBlock<24>;
template class Superscalar2ByteSliceColumnBlock<25>;
template class Superscalar2ByteSliceColumnBlock<26>;
template class Superscalar2ByteSliceColumnBlock<27>;
template class Superscalar2ByteSliceColumnBlock<28>;
template class Superscalar2ByteSliceColumnBlock<29>;
template class Superscalar2ByteSliceColumnBlock<30>;
template class Superscalar2ByteSliceColumnBlock<31>;
template class Superscalar2ByteSliceColumnBlock<32>;

}   //namespace
