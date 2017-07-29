#include    "include/hybridslice_column_block.h"
#include    <cstdlib>
#include    <cstring>
#include    <algorithm>
#include    <include/avx-utility.h>

namespace byteslice{

static constexpr size_t kPrefetchDistanceByteSlice = 512;
static constexpr size_t kPrefetchDistanceBitSlice = 64;

template <size_t BIT_WIDTH>
HybridSliceColumnBlock<BIT_WIDTH>::HybridSliceColumnBlock(size_t num):
    ColumnBlock(ColumnType::kHybridSlice, BIT_WIDTH, num){

    //allocate byteslice space
    for(size_t i=0; i < kNumByteSlices; i++){
        size_t size = CEIL(kNumTuplesPerBlock, sizeof(AvxUnit))*sizeof(AvxUnit);
        size_t ret = posix_memalign((void**)&data_[i], 32, size);
        ret = ret;
        memset(data_[i], 0x0, size);
    }

    //allocate bitslice space
    for(size_t i=0; i < kNumBitSlices; i++){
        size_t size = sizeof(AvxUnit) * CEIL(kNumTuplesPerBlock, kNumAvxBits);
        size_t ret = posix_memalign((void**)&bit_data_[i], 32, size);
        ret = ret;
        memset(bit_data_[i], 0x0, size);
    }
}

template <size_t BIT_WIDTH>
HybridSliceColumnBlock<BIT_WIDTH>::~HybridSliceColumnBlock(){
    //free byteslice
    for(size_t i=0; i < kNumByteSlices; i++){
        free(data_[i]);
    }

    //free bit slice
    for(size_t i=0; i < kNumBitSlices; i++){
        free(bit_data_[i]);
    }
}

template <size_t BIT_WIDTH>
bool HybridSliceColumnBlock<BIT_WIDTH>::Resize(size_t num){
    num_tuples_ = num;
    return true;
}

template <size_t BIT_WIDTH>
void HybridSliceColumnBlock<BIT_WIDTH>::SerToFile(SequentialWriteBinaryFile &file) const{

}

template <size_t BIT_WIDTH>
void HybridSliceColumnBlock<BIT_WIDTH>::DeserFromFile(const SequentialReadBinaryFile &file){

}

//Scan literal
template <size_t BIT_WIDTH>
void HybridSliceColumnBlock<BIT_WIDTH>::Scan(Comparator comparator,
                                             WordUnit literal,
                                             BitVectorBlock* bvblock,
                                             Bitwise bit_opt) const{
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

template <size_t BIT_WIDTH>
template <Comparator CMP>
void HybridSliceColumnBlock<BIT_WIDTH>::ScanHelper1(WordUnit literal,
                                                    BitVectorBlock* bvblock,
                                                    Bitwise bit_opt) const{
     switch(bit_opt){
        case Bitwise::kSet:
            return ScanHelper2<CMP, Bitwise::kSet>(literal, bvblock);
        case Bitwise::kAnd:
            return ScanHelper2<CMP, Bitwise::kAnd>(literal, bvblock);
        case Bitwise::kOr:
            return ScanHelper2<CMP, Bitwise::kOr>(literal, bvblock);
    }
}

template <size_t BIT_WIDTH>
template <Comparator CMP, Bitwise OPT>
void HybridSliceColumnBlock<BIT_WIDTH>::ScanHelper2(WordUnit literal,
                                                    BitVectorBlock* bvblock) const{
    AvxUnit literal_byteslice[4];
    AvxUnit literal_bitslice[kHybridSliceThreshold];
    //prepare the literal masks: two types
    if(kNumBitSlices > 0){
        for(size_t bit_id = kNumBitSlices - 1; bit_id < kNumBitSlices; bit_id--){
            literal_bitslice[bit_id] = avx_set1<WordUnit>(0ULL - (literal & 1ULL));
            literal >>= 1;
        }
    }
    literal <<= kNumPaddingBits;
//    for(size_t byte_id = 0; byte_id < kNumByteSlices; byte_id++){
//        ByteUnit byte = FLIP(static_cast<ByteUnit>(literal >> (8*(kNumByteSlices - 1 - byte_id))));
//        literal_byteslice[byte_id] = avx_set1<ByteUnit>(byte);
//    }

    switch(kNumByteSlices){
        case 4:
            literal_byteslice[0] = avx_set1<ByteUnit>(FLIP(static_cast<ByteUnit>(literal >> 24)));
            literal_byteslice[1] = avx_set1<ByteUnit>(FLIP(static_cast<ByteUnit>(literal >> 16)));
            literal_byteslice[2] = avx_set1<ByteUnit>(FLIP(static_cast<ByteUnit>(literal >> 8)));
            literal_byteslice[3] = avx_set1<ByteUnit>(FLIP(static_cast<ByteUnit>(literal)));
            break;
        case 3:
            literal_byteslice[0] = avx_set1<ByteUnit>(FLIP(static_cast<ByteUnit>(literal >> 16)));
            literal_byteslice[1] = avx_set1<ByteUnit>(FLIP(static_cast<ByteUnit>(literal >> 8)));
            literal_byteslice[2] = avx_set1<ByteUnit>(FLIP(static_cast<ByteUnit>(literal)));
            break;
        case 2:
            literal_byteslice[0] = avx_set1<ByteUnit>(FLIP(static_cast<ByteUnit>(literal >> 8)));
            literal_byteslice[1] = avx_set1<ByteUnit>(FLIP(static_cast<ByteUnit>(literal)));
            break;
        case 1:
            literal_byteslice[0] = avx_set1<ByteUnit>(FLIP(static_cast<ByteUnit>(literal)));
            break;
        case 0:
            break;
    }

    //For every 256 tuples
    for(size_t offset = 0; offset < num_tuples_; offset += kNumAvxBits){

        size_t bv_word_id = offset / kNumWordBits;

        AvxUnit m_lt, m_gt, m_eq;
        m_lt = avx_zero();
        m_gt = avx_zero();
        switch(OPT){
            case Bitwise::kSet:
                m_eq = avx_ones();
                break;
            case Bitwise::kAnd:
                m_eq = bvblock->GetAvxUnit(bv_word_id);
                break;
            case Bitwise::kOr:
                m_eq = avx_not(bvblock->GetAvxUnit(bv_word_id));
                break;
        }

        //The byte-slice component
        if(kNumByteSlices > 0){
            WordUnit* m_eq_ptr64 = reinterpret_cast<WordUnit*>(&m_eq);
            WordUnit* m_lt_ptr64 = reinterpret_cast<WordUnit*>(&m_lt);
            WordUnit* m_gt_ptr64 = reinterpret_cast<WordUnit*>(&m_gt);

            //For every 64 tuples
            for(size_t offset2 = 0; offset2 < kNumAvxBits; offset2 += kNumWordBits){
                WordUnit working_word_lt = 0ULL;
                WordUnit working_word_gt = 0ULL;
                WordUnit working_word_eq = 0ULL;
                size_t sub_word_id = offset2 / kNumWordBits;

                //For every 32 tuples
                for(size_t offset3 = 0; offset3 < kNumWordBits; offset3 += sizeof(AvxUnit)){
                    
                    AvxUnit mask_less, mask_greater, mask_equal;
                    mask_less = avx_zero();
                    mask_greater = avx_zero();
                    mask_equal = avx_ones();

                    int input_mask;
                    (void) input_mask;
                    input_mask = static_cast<int>(m_eq_ptr64[sub_word_id] >> offset3);

                    if((OPT==Bitwise::kSet) ||  0 != input_mask){
                        __builtin_prefetch(
                            data_[0] + offset + offset2 + offset3 + kPrefetchDistanceByteSlice);
                        ScanKernelByteSlice<CMP, 0>(
                                _mm256_lddqu_si256(reinterpret_cast<__m256i*>(data_[0]+offset+offset2+offset3)),
                                literal_byteslice[0],
                                mask_less,
                                mask_greater,
                                mask_equal);
                        if(kNumByteSlices > 1 &&
                                ((OPT==Bitwise::kSet && !avx_iszero(mask_equal)) 
                                || (OPT!=Bitwise::kSet && 0!=(input_mask & _mm256_movemask_epi8(mask_equal))))){
                            __builtin_prefetch(data_[1] + offset + offset2 + kPrefetchDistanceByteSlice);
                            ScanKernelByteSlice<CMP, 1>(
                                    _mm256_lddqu_si256(reinterpret_cast<__m256i*>(data_[1]+offset+offset2+offset3)),
                                    literal_byteslice[1],
                                    mask_less,
                                    mask_greater,
                                    mask_equal);
                            if(kNumByteSlices > 2 && 
                                    ((OPT==Bitwise::kSet && !avx_iszero(mask_equal)) 
                                    || (OPT!=Bitwise::kSet && 0!=(input_mask & _mm256_movemask_epi8(mask_equal))))){
                                ScanKernelByteSlice<CMP, 2>(
                                        _mm256_lddqu_si256(reinterpret_cast<__m256i*>(data_[2]+offset+offset2+offset3)),
                                        literal_byteslice[2],
                                        mask_less,
                                        mask_greater,
                                        mask_equal);
                                if(kNumByteSlices > 3 && 
                                        ((OPT==Bitwise::kSet && !avx_iszero(mask_equal)) 
                                        || (OPT!=Bitwise::kSet && 0!=(input_mask & _mm256_movemask_epi8(mask_equal))))){
                                    ScanKernelByteSlice<CMP, 3>(
                                            _mm256_lddqu_si256(reinterpret_cast<__m256i*>(data_[3]+offset+offset2+offset3)),
                                            literal_byteslice[3],
                                            mask_less,
                                            mask_greater,
                                            mask_equal);
                                }
                            }
                        }
                    }

                    //Condense 8-bit masks to 1-bit masks
                    //special treatment of equal mask
                    if(!(0 == kNumBitSlices && 
                                (CMP == Comparator::kLess || CMP == Comparator::kGreater))){
                        uint32_t mmask_eq = _mm256_movemask_epi8(mask_equal);
                        working_word_eq |= (static_cast<WordUnit>(mmask_eq) << offset3);
                    }
                    switch(CMP){
                        case Comparator::kEqual:
                        case Comparator::kInequal:
                            break;
                        case Comparator::kLess:
                        case Comparator::kLessEqual:
                            {
                            uint32_t mmask_lt = _mm256_movemask_epi8(mask_less);
                            working_word_lt |= (static_cast<WordUnit>(mmask_lt) << offset3);
                            }
                            break;
                        case Comparator::kGreater:
                        case Comparator::kGreaterEqual:
                            {
                            uint32_t mmask_gt = _mm256_movemask_epi8(mask_greater);
                            working_word_gt |= (static_cast<WordUnit>(mmask_gt) << offset3);
                            }
                            break;
                    }

                }   //end loop 32 tuples
                //Merge the 64-bit word
                //special treatment of equal mask
                if(!(0 == kNumBitSlices && 
                            (CMP == Comparator::kLess || CMP == Comparator::kGreater))){
                    m_eq_ptr64[sub_word_id] &= working_word_eq;
                }
                switch(CMP){
                    case Comparator::kEqual:
                    case Comparator::kInequal:
                        break;
                    case Comparator::kLess:
                    case Comparator::kLessEqual:
                        m_lt_ptr64[sub_word_id] = working_word_lt;
                        break;
                    case Comparator::kGreater:
                    case Comparator::kGreaterEqual:
                        m_gt_ptr64[sub_word_id] = working_word_gt;
                        break;
                }
            }   //end loop 64 tuples
        }   //end byte-slice component

        //The bit-slice component
        if(kNumBitSlices > 0){
            if((0 == kNumByteSlices && Bitwise::kSet == OPT) || !avx_iszero(m_eq)){
            for(size_t bit_id = 0; bit_id < kNumBitSlices; bit_id++){
                __builtin_prefetch(bit_data_[bit_id] + bv_word_id + kPrefetchDistanceBitSlice);
                AvxUnit data = _mm256_lddqu_si256(
                        reinterpret_cast<__m256i*>(bit_data_[bit_id] + bv_word_id));
                ScanKernelBitSlice<CMP>(data, literal_bitslice[bit_id], m_lt, m_gt, m_eq);
            }        
            }
            //for(size_t group_id = 0; group_id < CEIL(kNumBitSlices, kNumBitSlicesPerGroup); group_id++){
            //    for(size_t bit_id = group_id*kNumBitSlicesPerGroup;
            //        bit_id < std::min(group_id*kNumBitSlicesPerGroup+kNumBitSlicesPerGroup, kNumBitSlices);
            //        bit_id++){
            //        AvxUnit data = _mm256_lddqu_si256(
            //                reinterpret_cast<__m256i*>(bit_data_[bit_id] + bv_word_id));
            //        ScanKernelBitSlice<CMP>(data, literal_bitslice[bit_id], m_lt, m_gt, m_eq);
            //    }
            //    if(avx_iszero(m_eq)){
            //        break;
            //    }
            //}
        }   //end bit-slice compnent

        //Merge the result back to bit vector

        AvxUnit y;

        switch(CMP){
            case Comparator::kEqual:
                //y = m_eq;
                y = _mm256_load_si256(&m_eq);
                break;
            case Comparator::kInequal:
                y = avx_not(m_eq);
                break;
            case Comparator::kLess:
                //y = m_lt;
                y = _mm256_load_si256(&m_lt);
                break;
            case Comparator::kGreater:
                //y = m_gt;
                y = _mm256_load_si256(&m_gt);
                break;
            case Comparator::kLessEqual:
                y = avx_or(m_lt, m_eq);
                break;
            case Comparator::kGreaterEqual:
                y = avx_or(m_gt, m_eq);
                break;
        }

        switch(OPT){
            case Bitwise::kSet:
                break;
            case Bitwise::kAnd:
                y = avx_and(y, bvblock->GetAvxUnit(bv_word_id));
                break;
            case Bitwise::kOr:
                y = avx_or(y, bvblock->GetAvxUnit(bv_word_id));
                break;
        }

        //assert(!avx_iszero(y));

        bvblock->SetAvxUnit(y, bv_word_id);

    }   //end loop all

    bvblock->ClearTail();
}

//Scan other block
template <size_t BIT_WIDTH>
void HybridSliceColumnBlock<BIT_WIDTH>::Scan(Comparator comparator,
                                             const ColumnBlock* other_block,
                                             BitVectorBlock* bvblock, 
                                             Bitwise bit_opt) const{
    assert(bvblock->num() == num_tuples_);
    assert(other_block->num_tuples() == num_tuples_);
    assert(other_block->type() == type_);
    assert(other_block->bit_width() == bit_width_);

    const HybridSliceColumnBlock<BIT_WIDTH>* block2 = 
        static_cast<const HybridSliceColumnBlock<BIT_WIDTH>*>(other_block);
    
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

template <size_t BIT_WIDTH>
template <Comparator CMP>
void HybridSliceColumnBlock<BIT_WIDTH>::ScanHelper1(
                    const HybridSliceColumnBlock<BIT_WIDTH>* other_block,
                    BitVectorBlock* bvblock,
                    Bitwise bit_opt) const{
    switch(bit_opt){
        case Bitwise::kSet:
            return ScanHelper2<CMP, Bitwise::kSet>(other_block, bvblock);
        case Bitwise::kAnd:
            return ScanHelper2<CMP, Bitwise::kAnd>(other_block, bvblock);
        case Bitwise::kOr:
            return ScanHelper2<CMP, Bitwise::kOr>(other_block, bvblock);
    }
}

template <size_t BIT_WIDTH>
template <Comparator CMP, Bitwise OPT>
void HybridSliceColumnBlock<BIT_WIDTH>::ScanHelper2(
                    const HybridSliceColumnBlock<BIT_WIDTH>* other_block,
                    BitVectorBlock* bvblock) const{

    //For every 256 tuples
    for(size_t offset = 0; offset < num_tuples_; offset += kNumAvxBits){

        size_t bv_word_id = offset / kNumWordBits;

        AvxUnit m_lt, m_gt, m_eq;
        m_lt = avx_zero();
        m_gt = avx_zero();
        switch(OPT){
            case Bitwise::kSet:
                m_eq = avx_ones();
                break;
            case Bitwise::kAnd:
                m_eq = bvblock->GetAvxUnit(bv_word_id);
                break;
            case Bitwise::kOr:
                m_eq = avx_not(bvblock->GetAvxUnit(bv_word_id));
                break;
        }

        //The byte-slice component
        if(kNumByteSlices > 0){
            WordUnit* m_eq_ptr64 = reinterpret_cast<WordUnit*>(&m_eq);
            WordUnit* m_lt_ptr64 = reinterpret_cast<WordUnit*>(&m_lt);
            WordUnit* m_gt_ptr64 = reinterpret_cast<WordUnit*>(&m_gt);

            //For every 64 tuples
            for(size_t offset2 = 0; offset2 < kNumAvxBits; offset2 += kNumWordBits){
                WordUnit working_word_lt = 0ULL;
                WordUnit working_word_gt = 0ULL;
                WordUnit working_word_eq = 0ULL;
                size_t sub_word_id = offset2 / kNumWordBits;

                //For every 32 tuples
                for(size_t offset3 = 0; offset3 < kNumWordBits; offset3 += sizeof(AvxUnit)){
                    
                    AvxUnit mask_less, mask_greater, mask_equal;
                    mask_less = avx_zero();
                    mask_greater = avx_zero();
                    mask_equal = avx_ones();

                    int input_mask;
                    (void) input_mask;
                    input_mask = static_cast<int>(m_eq_ptr64[sub_word_id] >> offset3);

                    if((OPT==Bitwise::kSet) ||  0 != input_mask){
                        __builtin_prefetch(
                            data_[0] + offset + offset2 + offset3 + kPrefetchDistanceByteSlice);
                        __builtin_prefetch(
                            other_block->data_[0] + offset + offset2 + offset3 + kPrefetchDistanceByteSlice);
                        ScanKernelByteSlice<CMP, 0>(
                                _mm256_lddqu_si256(reinterpret_cast<__m256i*>(data_[0]+offset+offset2+offset3)),
                                _mm256_lddqu_si256(reinterpret_cast<__m256i*>(other_block->data_[0]+offset+offset2+offset3)),
                                mask_less,
                                mask_greater,
                                mask_equal);
                        if(kNumByteSlices > 1 &&
                                ((OPT==Bitwise::kSet && !avx_iszero(mask_equal)) 
                                || (OPT!=Bitwise::kSet && 0!=(input_mask & _mm256_movemask_epi8(mask_equal))))){
                            __builtin_prefetch(data_[1] + offset + offset2 + kPrefetchDistanceByteSlice);
                            __builtin_prefetch(other_block->data_[1] + offset + offset2 + kPrefetchDistanceByteSlice);
                            ScanKernelByteSlice<CMP, 1>(
                                    _mm256_lddqu_si256(reinterpret_cast<__m256i*>(data_[1]+offset+offset2+offset3)),
                                    _mm256_lddqu_si256(reinterpret_cast<__m256i*>(other_block->data_[1]+offset+offset2+offset3)),
                                    mask_less,
                                    mask_greater,
                                    mask_equal);
                            if(kNumByteSlices > 2 && 
                                    ((OPT==Bitwise::kSet && !avx_iszero(mask_equal)) 
                                    || (OPT!=Bitwise::kSet && 0!=(input_mask & _mm256_movemask_epi8(mask_equal))))){
                                ScanKernelByteSlice<CMP, 2>(
                                        _mm256_lddqu_si256(reinterpret_cast<__m256i*>(data_[2]+offset+offset2+offset3)),
                                        _mm256_lddqu_si256(reinterpret_cast<__m256i*>(other_block->data_[2]+offset+offset2+offset3)),
                                        mask_less,
                                        mask_greater,
                                        mask_equal);
                                if(kNumByteSlices > 3 && 
                                        ((OPT==Bitwise::kSet && !avx_iszero(mask_equal)) 
                                        || (OPT!=Bitwise::kSet && 0!=(input_mask & _mm256_movemask_epi8(mask_equal))))){
                                    ScanKernelByteSlice<CMP, 3>(
                                            _mm256_lddqu_si256(reinterpret_cast<__m256i*>(data_[3]+offset+offset2+offset3)),
                                            _mm256_lddqu_si256(reinterpret_cast<__m256i*>(other_block->data_[3]+offset+offset2+offset3)),
                                            mask_less,
                                            mask_greater,
                                            mask_equal);
                                }
                            }
                        }
                    }

                    //Condense 8-bit masks to 1-bit masks
                    //special treatment of equal mask
                    if(!(0 == kNumBitSlices && 
                                (CMP == Comparator::kLess || CMP == Comparator::kGreater))){
                        uint32_t mmask_eq = _mm256_movemask_epi8(mask_equal);
                        working_word_eq |= (static_cast<WordUnit>(mmask_eq) << offset3);
                    }
                    switch(CMP){
                        case Comparator::kEqual:
                        case Comparator::kInequal:
                            break;
                        case Comparator::kLess:
                        case Comparator::kLessEqual:
                            {
                            uint32_t mmask_lt = _mm256_movemask_epi8(mask_less);
                            working_word_lt |= (static_cast<WordUnit>(mmask_lt) << offset3);
                            }
                            break;
                        case Comparator::kGreater:
                        case Comparator::kGreaterEqual:
                            {
                            uint32_t mmask_gt = _mm256_movemask_epi8(mask_greater);
                            working_word_gt |= (static_cast<WordUnit>(mmask_gt) << offset3);
                            }
                            break;
                    }

                }   //end loop 32 tuples
                //Merge the 64-bit word
                //special treatment of equal mask
                if(!(0 == kNumBitSlices && 
                            (CMP == Comparator::kLess || CMP == Comparator::kGreater))){
                    m_eq_ptr64[sub_word_id] &= working_word_eq;
                }
                switch(CMP){
                    case Comparator::kEqual:
                    case Comparator::kInequal:
                        break;
                    case Comparator::kLess:
                    case Comparator::kLessEqual:
                        m_lt_ptr64[sub_word_id] = working_word_lt;
                        break;
                    case Comparator::kGreater:
                    case Comparator::kGreaterEqual:
                        m_gt_ptr64[sub_word_id] = working_word_gt;
                        break;
                }
            }   //end loop 64 tuples
        }   //end byte-slice component

        //The bit-slice component
        if((kNumBitSlices > 0) /*&& !avx_iszero(m_eq)*/){
            for(size_t bit_id = 0; bit_id < kNumBitSlices; bit_id++){
                __builtin_prefetch(bit_data_[bit_id] + bv_word_id + kPrefetchDistanceBitSlice);
                __builtin_prefetch(other_block->bit_data_[bit_id] + bv_word_id + kPrefetchDistanceBitSlice);
                AvxUnit data = _mm256_lddqu_si256(
                        reinterpret_cast<__m256i*>(bit_data_[bit_id] + bv_word_id));
                AvxUnit other_data = _mm256_lddqu_si256(
                        reinterpret_cast<__m256i*>(other_block->bit_data_[bit_id] + bv_word_id));
                ScanKernelBitSlice<CMP>(data, other_data, m_lt, m_gt, m_eq);
            }        
            //for(size_t group_id = 0; group_id < CEIL(kNumBitSlices, kNumBitSlicesPerGroup); group_id++){
            //    if(avx_iszero(m_eq)){
            //        break;
            //    }
            //    for(size_t bit_id = group_id*kNumBitSlicesPerGroup;
            //        bit_id < std::min(group_id*kNumBitSlicesPerGroup+kNumBitSlicesPerGroup, kNumBitSlices);
            //        bit_id++){
            //        AvxUnit data = _mm256_lddqu_si256(
            //                reinterpret_cast<__m256i*>(bit_data_[bit_id] + bv_word_id));
            //        ScanKernelBitSlice<CMP>(data, literal_bitslice[bit_id], m_lt, m_gt, m_eq);
            //    }
            //}
        }   //end bit-slice compnent

        //Merge the result back to bit vector

        AvxUnit y;

        switch(CMP){
            case Comparator::kEqual:
                y = m_eq;
                break;
            case Comparator::kInequal:
                y = avx_not(m_eq);
                break;
            case Comparator::kLess:
                //y = m_lt;
                y = _mm256_load_si256(&m_lt);
                break;
            case Comparator::kGreater:
                //y = m_gt;
                y = _mm256_load_si256(&m_gt);
                break;
            case Comparator::kLessEqual:
                y = avx_or(m_lt, m_eq);
                break;
            case Comparator::kGreaterEqual:
                y = avx_or(m_gt, m_eq);
                break;
        }

        switch(OPT){
            case Bitwise::kSet:
                break;
            case Bitwise::kAnd:
                y = avx_and(y, bvblock->GetAvxUnit(bv_word_id));
                break;
            case Bitwise::kOr:
                y = avx_or(y, bvblock->GetAvxUnit(bv_word_id));
                break;
        }

        //assert(!avx_iszero(y));

        bvblock->SetAvxUnit(y, bv_word_id);

    }   //end loop all

    bvblock->ClearTail();

}

template <size_t BIT_WIDTH>
template <Comparator CMP, size_t BYTE_ID>
inline void HybridSliceColumnBlock<BIT_WIDTH>::ScanKernelByteSlice(const AvxUnit &byteslice1,
                                                                   const AvxUnit &byteslice2,
                                                                   AvxUnit &mask_less,
                                                                   AvxUnit &mask_greater,
                                                                   AvxUnit &mask_equal) const{
    //internal ByteSlice --- not last BS OR there's bit-slice compnents   
    if(kNumBitSlices > 0 || BYTE_ID < kNumByteSlices - 1){ 
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
    else if(0 == kNumBitSlices && BYTE_ID == kNumByteSlices - 1){   
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

template <size_t BIT_WIDTH>
template <Comparator CMP>
inline void HybridSliceColumnBlock<BIT_WIDTH>::ScanKernelBitSlice(const AvxUnit &data,
                                                                  const AvxUnit &other,
                                                                  AvxUnit &m_lt,
                                                                  AvxUnit &m_gt,
                                                                  AvxUnit &m_eq) const{
    switch(CMP){
        case Comparator::kEqual:
        case Comparator::kInequal:
            m_eq = avx_andnot(avx_xor(data, other), m_eq);
            break;
        case Comparator::kLess:
        case Comparator::kLessEqual:
            m_lt = avx_or(m_lt, avx_and(m_eq, avx_andnot(data, other)));
            m_eq = avx_andnot(avx_xor(data, other), m_eq);
            break;
        case Comparator::kGreater:
        case Comparator::kGreaterEqual:
            m_gt = avx_or(m_gt, avx_and(m_eq,avx_andnot(other, data)));
            m_eq = avx_andnot(avx_xor(data, other), m_eq);
            break;
    }
    return;
}



template <size_t BIT_WIDTH>
void HybridSliceColumnBlock<BIT_WIDTH>::BulkLoadArray(const WordUnit* codes,
                                                      size_t num,
                                                      size_t start_pos){
    assert(start_pos + num <= num_tuples_);
    for(size_t i = 0; i < num; i++){
        SetTuple(start_pos+i, codes[i]);
    }
}

//explicit specialization
template class HybridSliceColumnBlock<1>;
template class HybridSliceColumnBlock<2>;
template class HybridSliceColumnBlock<3>;
template class HybridSliceColumnBlock<4>;
template class HybridSliceColumnBlock<5>;
template class HybridSliceColumnBlock<6>;
template class HybridSliceColumnBlock<7>;
template class HybridSliceColumnBlock<8>;
template class HybridSliceColumnBlock<9>;
template class HybridSliceColumnBlock<10>;
template class HybridSliceColumnBlock<11>;
template class HybridSliceColumnBlock<12>;
template class HybridSliceColumnBlock<13>;
template class HybridSliceColumnBlock<14>;
template class HybridSliceColumnBlock<15>;
template class HybridSliceColumnBlock<16>;
template class HybridSliceColumnBlock<17>;
template class HybridSliceColumnBlock<18>;
template class HybridSliceColumnBlock<19>;
template class HybridSliceColumnBlock<20>;
template class HybridSliceColumnBlock<21>;
template class HybridSliceColumnBlock<22>;
template class HybridSliceColumnBlock<23>;
template class HybridSliceColumnBlock<24>;
template class HybridSliceColumnBlock<25>;
template class HybridSliceColumnBlock<26>;
template class HybridSliceColumnBlock<27>;
template class HybridSliceColumnBlock<28>;
template class HybridSliceColumnBlock<29>;
template class HybridSliceColumnBlock<30>;
template class HybridSliceColumnBlock<31>;
template class HybridSliceColumnBlock<32>;


}   //namespace
