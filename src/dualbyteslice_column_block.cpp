#include    "include/dualbyteslice_column_block.h"
#include    <cstdlib>
#include    <cstring>
#include    <include/avx-utility.h>

namespace byteslice{
    
#ifdef      NEARLYSTOP
#warning    "Early-stop is disabled in DualByteSliceColumnBlock!"
#endif

static constexpr size_t kPrefetchDistance = 512;

template <size_t BIT_WIDTH, Direction PDIRECTION>
DualByteSliceColumnBlock<BIT_WIDTH, PDIRECTION>::DualByteSliceColumnBlock(size_t num):
    ColumnBlock(
            PDIRECTION==Direction::kLeft ? 
                ColumnType::kDualByteSlicePadLeft:ColumnType::kDualByteSlicePadRight, 
            BIT_WIDTH, 
            num)    
{
    //allocate memory space
    assert(num <= kNumTuplesPerBlock);
    for(size_t i=0; i < kNumDualBytesPerCode; i++){
        size_t ret = posix_memalign((void**)&data_[i], 32, kMemSizePerDualByteSlice);                    
        ret = ret;
        memset(data_[i], 0x0, kMemSizePerDualByteSlice);
    }

}

template <size_t BIT_WIDTH, Direction PDIRECTION>
DualByteSliceColumnBlock<BIT_WIDTH, PDIRECTION>::~DualByteSliceColumnBlock(){
    for(size_t i=0; i < kNumDualBytesPerCode; i++){
        free(data_[i]);
    }
}

template <size_t BIT_WIDTH, Direction PDIRECTION>
bool DualByteSliceColumnBlock<BIT_WIDTH, PDIRECTION>::Resize(size_t num){
    num_tuples_ = num;
    return true;
}

template <size_t BIT_WIDTH, Direction PDIRECTION>
void DualByteSliceColumnBlock<BIT_WIDTH, PDIRECTION>::
                    SerToFile(SequentialWriteBinaryFile &file) const{
    file.Append(&num_tuples_, sizeof(num_tuples_));
    for(size_t dualbyte_id = 0; dualbyte_id < kNumDualBytesPerCode; dualbyte_id++){
        file.Append(data_[dualbyte_id], kMemSizePerDualByteSlice);
    }
}

template <size_t BIT_WIDTH, Direction PDIRECTION>
void DualByteSliceColumnBlock<BIT_WIDTH, PDIRECTION>::
                    DeserFromFile(const SequentialReadBinaryFile &file){
    file.Read(&num_tuples_, sizeof(num_tuples_));
    for(size_t dualbyte_id = 0; dualbyte_id < kNumDualBytesPerCode; dualbyte_id++){
        file.Read(data_[dualbyte_id], kMemSizePerDualByteSlice);
    }
}


//Scan against literal
template <size_t BIT_WIDTH, Direction PDIRECTION>
void DualByteSliceColumnBlock<BIT_WIDTH, PDIRECTION>::Scan(Comparator comparator,
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
void DualByteSliceColumnBlock<BIT_WIDTH, PDIRECTION>::ScanHelper1(WordUnit literal,
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
void DualByteSliceColumnBlock<BIT_WIDTH, PDIRECTION>::ScanHelper2(WordUnit literal,
                                            BitVectorBlock* bvblock) const {
    //Prepare dual-byte-slices of literal
    AvxUnit mask_literal[kNumDualBytesPerCode];
    literal &= kCodeMask;
    if(Direction::kRight == PDIRECTION){
        literal <<= kNumPaddingBits;
    }
    for(size_t dualbyte_id=0; dualbyte_id < kNumDualBytesPerCode; dualbyte_id++){
         DualByteUnit dualbyte = FLIP(static_cast<DualByteUnit>(literal >> 16*(kNumDualBytesPerCode - 1 - dualbyte_id)));
         mask_literal[dualbyte_id] = avx_set1<DualByteUnit>(dualbyte);
    }
    
    //for every kNumWordBits (64) tuples
    for(size_t offset = 0, bv_word_id = 0; offset < num_tuples_; offset += kNumWordBits, bv_word_id++){
        WordUnit bitvector_word = WordUnit(0);
        //need several iteration of AVX scan
        for(size_t i=0; i < kNumWordBits; i += sizeof(AvxUnit)/sizeof(DualByteUnit)){
            AvxUnit m_less = avx_zero();
            AvxUnit m_greater = avx_zero();
            AvxUnit m_equal; 
            int input_mask;
            (void) input_mask;

            switch(OPT){
                case Bitwise::kSet:
                    m_equal = avx_ones();
                    break;
                case Bitwise::kAnd:
                    input_mask = static_cast<int>(bvblock->GetWordUnit(bv_word_id) >> i);
                    m_equal = avx_ones();
                    break;
                case Bitwise::kOr:
                    input_mask = ~static_cast<int>(bvblock->GetWordUnit(bv_word_id) >> i);
                    m_equal = avx_ones();
                    break;
            }

            if(
#ifndef         NEARLYSTOP
                (OPT==Bitwise::kSet) ||  0 != input_mask
#else           
                true
#endif
              ){
                __builtin_prefetch(data_[0] + offset + i + kPrefetchDistance);
                ScanKernel2<CMP, 0>(
                        _mm256_lddqu_si256(reinterpret_cast<__m256i*>(data_[0]+offset+i)),
                        mask_literal[0],
                        m_less,
                        m_greater,
                        m_equal);
                if(kNumDualBytesPerCode > 1
#ifndef                 NEARLYSTOP
                        && ((OPT==Bitwise::kSet && !avx_iszero(m_equal))
                            || (OPT!=Bitwise::kSet && 0!=(input_mask & _mm256_movemask_epi8(m_equal))))
#endif
                  ){
//                     __builtin_prefetch(data_[1] + offset + i + kPrefetchDistance);
                    ScanKernel2<CMP, 1>(
                            _mm256_lddqu_si256(reinterpret_cast<__m256i*>(data_[1]+offset+i)),
                            mask_literal[1],
                            m_less,
                            m_greater,
                            m_equal);
//                     if(kNumDualBytesPerCode > 2
// #ifndef                     NEARLYSTOP
//                             && ((OPT==Bitwise::kSet && !avx_iszero(m_equal)) 
//                                 || (OPT!=Bitwise::kSet && 0!=(input_mask & _mm256_movemask_epi8(m_equal))))
// #endif
//                       ){
//                         ScanKernel2<CMP, 2>(
//                                 _mm256_lddqu_si256(reinterpret_cast<__m256i*>(data_[2]+offset+i)),
//                                 mask_literal[2],
//                                 m_less,
//                                 m_greater,
//                                 m_equal);
//                         if(kNumDualBytesPerCode > 3
// #ifndef                         NEARLYSTOP
//                                 && ((OPT==Bitwise::kSet && !avx_iszero(m_equal)) 
//                                     || (OPT!=Bitwise::kSet && 0!=(input_mask & _mm256_movemask_epi8(m_equal))))
// #endif
//                           ){
//                             ScanKernel2<CMP, 3>(
//                                     _mm256_lddqu_si256(reinterpret_cast<__m256i*>(data_[3]+offset+i)),
//                                     mask_literal[3],
//                                     m_less,
//                                     m_greater,
//                                     m_equal);
//                         }
//                     }
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
            //move mask
            uint32_t mmask = movemask_epi16(m_result);
            //save in temporary bit vector
            bitvector_word |= (static_cast<WordUnit>(mmask) << i);
        }
        //put result bitvector into bitvector block
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
void DualByteSliceColumnBlock<BIT_WIDTH, PDIRECTION>::Scan(Comparator comparator,
        const ColumnBlock* other_block, BitVectorBlock* bvblock, Bitwise bit_opt) const{

    assert(bvblock->num() == num_tuples_);
    assert(other_block->num_tuples() == num_tuples_);
    assert(other_block->type() == type_);
    assert(other_block->bit_width() == bit_width_);

    const DualByteSliceColumnBlock<BIT_WIDTH, PDIRECTION>* block2 =
        static_cast<const DualByteSliceColumnBlock<BIT_WIDTH, PDIRECTION>*>(other_block);
    
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
void DualByteSliceColumnBlock<BIT_WIDTH, PDIRECTION>::ScanHelper1(
                            const DualByteSliceColumnBlock<BIT_WIDTH, PDIRECTION>* other_block,
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
void DualByteSliceColumnBlock<BIT_WIDTH, PDIRECTION>::ScanHelper2(
                            const DualByteSliceColumnBlock<BIT_WIDTH, PDIRECTION>* other_block,
                            BitVectorBlock* bvblock) const {

    //for every kNumWordBits (64) tuples
    for(size_t offset = 0, bv_word_id = 0; offset < num_tuples_; offset += kNumWordBits, bv_word_id++){
        WordUnit bitvector_word = WordUnit(0);
        //need several iteration of AVX scan
        for(size_t i=0; i < kNumWordBits; i += sizeof(AvxUnit)/sizeof(DualByteUnit)){
            AvxUnit m_less = avx_zero();
            AvxUnit m_greater = avx_zero();
            AvxUnit m_equal; 
            int input_mask = static_cast<int>(-1ULL);
            (void) input_mask;

            switch(OPT){
                case Bitwise::kSet:
                    m_equal = avx_ones();
                    break;
                case Bitwise::kAnd:
                    input_mask = static_cast<int>(bvblock->GetWordUnit(bv_word_id) >> i);
                    m_equal = avx_ones();
                    break;
                case Bitwise::kOr:
                    input_mask = ~static_cast<int>(bvblock->GetWordUnit(bv_word_id) >> i);
                    m_equal = avx_ones();
                    break;
            }

            if((OPT==Bitwise::kSet) ||  0 != input_mask){
                __builtin_prefetch(data_[0] + offset + i + 1024);
                __builtin_prefetch(other_block->data_[0] + offset + i + 1024);
                ScanKernel2<CMP, 0>(
                        _mm256_lddqu_si256(reinterpret_cast<__m256i*>(data_[0]+offset+i)),
                        _mm256_lddqu_si256(reinterpret_cast<__m256i*>(other_block->data_[0]+offset+i)),
                        m_less,
                        m_greater,
                        m_equal);
                if(kNumDualBytesPerCode > 1 && 
                        ((OPT==Bitwise::kSet && !avx_iszero(m_equal)) 
                        || (OPT!=Bitwise::kSet && 0!=(input_mask & _mm256_movemask_epi8(m_equal))))){
//                     __builtin_prefetch(data_[1] + offset + i + 1024);
//                     __builtin_prefetch(other_block->data_[1] + offset + i + 1024);
                    ScanKernel2<CMP, 1>(
                            _mm256_lddqu_si256(reinterpret_cast<__m256i*>(data_[1]+offset+i)),
                            _mm256_lddqu_si256(reinterpret_cast<__m256i*>(other_block->data_[1]+offset+i)),
                            m_less,
                            m_greater,
                            m_equal);
//                     if(kNumDualBytesPerCode > 2 && 
//                             ((OPT==Bitwise::kSet && !avx_iszero(m_equal)) 
//                             || (OPT!=Bitwise::kSet && 0!=(input_mask & _mm256_movemask_epi8(m_equal))))){
//                         ScanKernel2<CMP, 2>(
//                                 _mm256_lddqu_si256(reinterpret_cast<__m256i*>(data_[2]+offset+i)),
//                                 _mm256_lddqu_si256(reinterpret_cast<__m256i*>(other_block->data_[2]+offset+i)),
//                                 m_less,
//                                 m_greater,
//                                 m_equal);
//                         if(kNumDualBytesPerCode > 3 && 
//                                 ((OPT==Bitwise::kSet && !avx_iszero(m_equal)) 
//                                 || (OPT!=Bitwise::kSet && 0!=(input_mask & _mm256_movemask_epi8(m_equal))))){
//                             ScanKernel2<CMP, 3>(
//                                     _mm256_lddqu_si256(reinterpret_cast<__m256i*>(data_[3]+offset+i)),
//                                     _mm256_lddqu_si256(reinterpret_cast<__m256i*>(other_block->data_[3]+offset+i)),
//                                     m_less,
//                                     m_greater,
//                                     m_equal);
//                         }
//                     }
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
            //move mask
            uint32_t mmask = movemask_epi16(m_result);
            //save in temporary bit vector
            bitvector_word |= (static_cast<WordUnit>(mmask) << i);
        }
        //put result bitvector into bitvector block
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
inline void DualByteSliceColumnBlock<BIT_WIDTH, PDIRECTION>::ScanKernel
                                                        (const AvxUnit &byteslice1,
                                                         const AvxUnit &byteslice2,
                                                         AvxUnit &mask_less,
                                                         AvxUnit &mask_greater,
                                                         AvxUnit &mask_equal) const {
    switch(CMP){
        case Comparator::kEqual:
        case Comparator::kInequal:
            mask_equal = 
                avx_and(mask_equal, avx_cmpeq<DualByteUnit>(byteslice1, byteslice2));
            break;
        case Comparator::kLess:
        case Comparator::kLessEqual:
            mask_less = 
                avx_or(mask_less, avx_and(mask_equal, avx_cmplt<DualByteUnit>(byteslice1, byteslice2)));
            mask_equal = 
                avx_and(mask_equal, avx_cmpeq<DualByteUnit>(byteslice1, byteslice2));
            break;
        case Comparator::kGreater:
        case Comparator::kGreaterEqual:
            mask_greater =
                avx_or(mask_greater, avx_and(mask_equal, avx_cmpgt<DualByteUnit>(byteslice1, byteslice2)));
            mask_equal = 
                avx_and(mask_equal, avx_cmpeq<DualByteUnit>(byteslice1, byteslice2));
            break;
    }
}

//Scan Kernel2 --- Optimized on Scan Kernel
//to remove unnecessary equal comparison for last dual byte slice
template <size_t BIT_WIDTH, Direction PDIRECTION>
template <Comparator CMP, size_t DualBYTE_ID>
inline void DualByteSliceColumnBlock<BIT_WIDTH, PDIRECTION>::ScanKernel2
                                                        (const AvxUnit &byteslice1,
                                                         const AvxUnit &byteslice2,
                                                         AvxUnit &mask_less,
                                                         AvxUnit &mask_greater,
                                                         AvxUnit &mask_equal) const {

    //internal DualByteSlice --- not last BS                                                        
    if(DualBYTE_ID < kNumDualBytesPerCode - 1){ 
        switch(CMP){
            case Comparator::kEqual:
            case Comparator::kInequal:
                mask_equal = 
                    avx_and(mask_equal, avx_cmpeq<DualByteUnit>(byteslice1, byteslice2));
                break;
            case Comparator::kLess:
            case Comparator::kLessEqual:
                mask_less = 
                    avx_or(mask_less, avx_and(mask_equal, avx_cmplt<DualByteUnit>(byteslice1, byteslice2)));
                mask_equal = 
                    avx_and(mask_equal, avx_cmpeq<DualByteUnit>(byteslice1, byteslice2));
                break;
            case Comparator::kGreater:
            case Comparator::kGreaterEqual:
                mask_greater =
                    avx_or(mask_greater, avx_and(mask_equal, avx_cmpgt<DualByteUnit>(byteslice1, byteslice2)));
                mask_equal = 
                    avx_and(mask_equal, avx_cmpeq<DualByteUnit>(byteslice1, byteslice2));
                break;
        }
    }
    //last BS: no need to compute mask_equal for some comparisons
    else if(DualBYTE_ID == kNumDualBytesPerCode - 1){   
        switch(CMP){
            case Comparator::kEqual:
            case Comparator::kInequal:
                mask_equal = 
                    avx_and(mask_equal, avx_cmpeq<DualByteUnit>(byteslice1, byteslice2));
                break;
            case Comparator::kLessEqual:
                mask_less = 
                    avx_or(mask_less, avx_and(mask_equal, avx_cmplt<DualByteUnit>(byteslice1, byteslice2)));
                mask_equal = 
                    avx_and(mask_equal, avx_cmpeq<DualByteUnit>(byteslice1, byteslice2));
                break;
            case Comparator::kLess:
                mask_less = 
                    avx_or(mask_less, avx_and(mask_equal, avx_cmplt<DualByteUnit>(byteslice1, byteslice2)));
                break;
            case Comparator::kGreaterEqual:
                mask_greater =
                    avx_or(mask_greater, avx_and(mask_equal, avx_cmpgt<DualByteUnit>(byteslice1, byteslice2)));
                mask_equal = 
                    avx_and(mask_equal, avx_cmpeq<DualByteUnit>(byteslice1, byteslice2));
                break;
            case Comparator::kGreater:
                mask_greater =
                    avx_or(mask_greater, avx_and(mask_equal, avx_cmpgt<DualByteUnit>(byteslice1, byteslice2)));
                break;
        }
    }
    //otherwise, do nothing

}


template <size_t BIT_WIDTH, Direction PDIRECTION>
void DualByteSliceColumnBlock<BIT_WIDTH, PDIRECTION>::BulkLoadArray(const WordUnit* codes,
                                                        size_t num, size_t start_pos){
    assert(start_pos + num <= num_tuples_);
    for(size_t i = 0; i < num; i++){
        SetTuple(start_pos+i, codes[i]);
    }
}



//explicit specialization
//default padding: right
template class DualByteSliceColumnBlock<1>;
template class DualByteSliceColumnBlock<2>;
template class DualByteSliceColumnBlock<3>;
template class DualByteSliceColumnBlock<4>;
template class DualByteSliceColumnBlock<5>;
template class DualByteSliceColumnBlock<6>;
template class DualByteSliceColumnBlock<7>;
template class DualByteSliceColumnBlock<8>;
template class DualByteSliceColumnBlock<9>;
template class DualByteSliceColumnBlock<10>;
template class DualByteSliceColumnBlock<11>;
template class DualByteSliceColumnBlock<12>;
template class DualByteSliceColumnBlock<13>;
template class DualByteSliceColumnBlock<14>;
template class DualByteSliceColumnBlock<15>;
template class DualByteSliceColumnBlock<16>;
template class DualByteSliceColumnBlock<17>;
template class DualByteSliceColumnBlock<18>;
template class DualByteSliceColumnBlock<19>;
template class DualByteSliceColumnBlock<20>;
template class DualByteSliceColumnBlock<21>;
template class DualByteSliceColumnBlock<22>;
template class DualByteSliceColumnBlock<23>;
template class DualByteSliceColumnBlock<24>;
template class DualByteSliceColumnBlock<25>;
template class DualByteSliceColumnBlock<26>;
template class DualByteSliceColumnBlock<27>;
template class DualByteSliceColumnBlock<28>;
template class DualByteSliceColumnBlock<29>;
template class DualByteSliceColumnBlock<30>;
template class DualByteSliceColumnBlock<31>;
template class DualByteSliceColumnBlock<32>;

}   //namespace
