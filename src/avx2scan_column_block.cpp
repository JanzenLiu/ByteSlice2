#include    "include/avx2scan_column_block.h"
#include    <cstdlib>
#include    <cstring>
#include    <immintrin.h>

namespace byteslice{

static constexpr size_t kPrefetchDistance = 16*64;

template <size_t BIT_WIDTH>
Avx2ScanColumnBlock<BIT_WIDTH>::Avx2ScanColumnBlock(size_t num):
    ColumnBlock(ColumnType::kAvx2Scan,
                BIT_WIDTH,
                num){
    assert(num <= kNumTuplesPerBlock);
    //allocate memory space
    //allocate 32 more bytes at the end to make sure we can load the last segment
    size_t size = CEIL(kNumTuplesPerBlock, kNumTuplePerSegment) * kNumBytesPerSegment + 32;
    int ret = posix_memalign((void**)&data_, 32, size);
    ret = ret;
    memset(data_, 0x0, size);

    //precompute helpers
    for(size_t i=0; i < kNumTuplePerSegment; i++){
        helpers_[i].byte_id = (i*BIT_WIDTH) / 8;
        helpers_[i].shift = (i*BIT_WIDTH) % 8;
    }

}

template <size_t BIT_WIDTH>
void Avx2ScanColumnBlock<BIT_WIDTH>::SetUpMasks(AvxUnit& shiftv,
                                                AvxUnit& shufflev,
                                                AvxUnit& code_maskv) const{
    //precompute masks
    //All code mask
    code_maskv = _mm256_set1_epi32(static_cast<uint32_t>(kCodeMask));
    //Independent shuffle mask:
    shiftv = _mm256_set_epi32(helpers_[7].shift,
                               helpers_[6].shift,
                               helpers_[5].shift,
                               helpers_[4].shift,
                               helpers_[3].shift,
                               helpers_[2].shift,
                               helpers_[1].shift,
                               helpers_[0].shift);
    //Byte-level shuffle masks:
    //Note: two 128-lanes are independent
    size_t base = helpers_[4].byte_id;
    shufflev = _mm256_set_epi8(
            helpers_[7].byte_id-base+3, helpers_[7].byte_id-base+2,
            helpers_[7].byte_id-base+1, helpers_[7].byte_id-base,
            helpers_[6].byte_id-base+3, helpers_[6].byte_id-base+2,
            helpers_[6].byte_id-base+1, helpers_[6].byte_id-base,
            helpers_[5].byte_id-base+3, helpers_[5].byte_id-base+2,
            helpers_[5].byte_id-base+1, helpers_[5].byte_id-base,
            helpers_[4].byte_id-base+3, helpers_[4].byte_id-base+2,
            helpers_[4].byte_id-base+1, helpers_[4].byte_id-base,
            helpers_[3].byte_id+3, helpers_[3].byte_id+2,
            helpers_[3].byte_id+1, helpers_[3].byte_id,
            helpers_[2].byte_id+3, helpers_[2].byte_id+2,
            helpers_[2].byte_id+1, helpers_[2].byte_id,
            helpers_[1].byte_id+3, helpers_[1].byte_id+2,
            helpers_[1].byte_id+1, helpers_[1].byte_id,
            helpers_[0].byte_id+3, helpers_[0].byte_id+2,
            helpers_[0].byte_id+1, helpers_[0].byte_id);

}

template <size_t BIT_WIDTH>
Avx2ScanColumnBlock<BIT_WIDTH>::~Avx2ScanColumnBlock(){
    free(data_);
}

template <size_t BIT_WIDTH>
bool Avx2ScanColumnBlock<BIT_WIDTH>::Resize(size_t num){
    num_tuples_ = num;
    return true;
}

template <size_t BIT_WIDTH>
void Avx2ScanColumnBlock<BIT_WIDTH>::SerToFile(SequentialWriteBinaryFile &file) const{

}

template <size_t BIT_WIDTH>
void Avx2ScanColumnBlock<BIT_WIDTH>::DeserFromFile(const SequentialReadBinaryFile &file){

}

template <size_t BIT_WIDTH>
inline AvxUnit Avx2ScanColumnBlock<BIT_WIDTH>::Unpack8(size_t index, 
                                                       const AvxUnit& shiftv, 
                                                       const AvxUnit& shufflev, 
                                                       const AvxUnit& code_maskv) const{
    assert(0 == index % kNumBytesPerSegment);
    //Shit happens for 27, 29 30 and 31 bit. Ignore them for now.
    __m128i hi = _mm_lddqu_si128(reinterpret_cast<__m128i*>(data_ + index + helpers_[4].byte_id));
    __m128i lo = _mm_lddqu_si128(reinterpret_cast<__m128i*>(data_ + index));

    //AvxUnit src = _mm256_loadu2_m128i(
    //        reinterpret_cast<__m128i*>(data_ + index + helpers_[4].byte_id),
    //        reinterpret_cast<__m128i*>(data_ + index));

    //AvxUnit src = _mm256_set_m128i(hi, lo);

    AvxUnit src = _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);


    AvxUnit ret = _mm256_shuffle_epi8(src, shufflev);
    ret = _mm256_srlv_epi32(ret, shiftv);
    ret = _mm256_and_si256(ret, code_maskv);
    return ret;
}

template <size_t BIT_WIDTH>
template <Comparator CMP>
inline AvxUnit Avx2ScanColumnBlock<BIT_WIDTH>::ScanKernel(AvxUnit a, AvxUnit b) const{
    AvxUnit mask_less, mask_greater, mask_equal;
    switch(CMP){
        case Comparator::kLessEqual:
            mask_equal = avx_cmpeq<uint32_t>(a, b);
        case Comparator::kLess:
            mask_less = avx_cmplt<uint32_t>(a, b);
            break;
        case Comparator::kGreaterEqual:
            mask_equal = avx_cmpeq<uint32_t>(a, b);
        case Comparator::kGreater:
            mask_greater = avx_cmpgt<uint32_t>(a, b);
            break;
        case Comparator::kEqual:
        case Comparator::kInequal:
            mask_equal = avx_cmpeq<uint32_t>(a, b);
            break;
    }

    AvxUnit ret;
    switch(CMP){
        case Comparator::kLessEqual:
            ret = avx_or(mask_less, mask_equal);
            break;
        case Comparator::kLess:
            ret = mask_less;
            break;
        case Comparator::kGreaterEqual:
            ret = avx_or(mask_greater, mask_equal);
            break;
        case Comparator::kGreater:
            ret = mask_greater;
            break;
        case Comparator::kEqual:
            ret = mask_equal;
            break;
        case Comparator::kInequal:
            ret = avx_not(mask_equal);
            break;
    }

    return ret;
}

//Scan literal
template <size_t BIT_WIDTH>
void Avx2ScanColumnBlock<BIT_WIDTH>::Scan(Comparator    comparator,
                                           WordUnit     literal,
                                           BitVectorBlock*  bvblock,
                                           Bitwise      bit_opt) const{
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
void Avx2ScanColumnBlock<BIT_WIDTH>::ScanHelper1(WordUnit   literal,
                                                 BitVectorBlock*    bvblock,
                                                 Bitwise    bit_opt) const{
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
void Avx2ScanColumnBlock<BIT_WIDTH>::ScanHelper2(WordUnit   literal,
                                                 BitVectorBlock*    bvblock) const{

    AvxUnit shiftv;
    AvxUnit shufflev;
    AvxUnit code_maskv;

    SetUpMasks(shiftv, shufflev, code_maskv);

    AvxUnit signbit_maskv = avx_set1<uint32_t>(1 << 31);

    //Do real work here
    AvxUnit literalv = avx_set1<uint32_t>(literal);

    if(32 == BIT_WIDTH){
        literalv = avx_xor(literalv, signbit_maskv);
    }

    //For each 64 tuples that correspond to a bitvector word
    for(size_t offset=0; offset < num_tuples_; offset += kNumWordBits){

        WordUnit bv_word = 0ULL;
        size_t bv_word_id = offset / kNumWordBits;

        for(size_t offset2=0; offset2 < kNumWordBits; offset2 += kNumTuplePerSegment){
            size_t segment_id = (offset + offset2) / kNumTuplePerSegment;
            size_t byte_index = segment_id * kNumBytesPerSegment;
            AvxUnit datav = Unpack8(byte_index, shiftv, shufflev, code_maskv);

#           ifndef NPREFETCH
#           warning "Prefetch Enabled in Avx2ScanColumnBlock"
            __builtin_prefetch(data_ + byte_index + kPrefetchDistance);
#           endif

            if(32 == BIT_WIDTH){
                datav = avx_xor(datav, signbit_maskv);
            }

            AvxUnit resultv = ScanKernel<CMP>(datav, literalv);
            uint32_t mmask = _mm256_movemask_ps(reinterpret_cast<__m256>(resultv));
            bv_word |= (static_cast<WordUnit>(mmask) << offset2);
        }

        //Merge into bit vector
        WordUnit x = bv_word;
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

//Scan other block
template <size_t BIT_WIDTH>
void Avx2ScanColumnBlock<BIT_WIDTH>::Scan(Comparator comparator,
                                          const ColumnBlock* other_block,
                                          BitVectorBlock* bvblock,
                                          Bitwise bit_opt) const{
    assert(bvblock->num() == num_tuples_);
    assert(other_block->num_tuples() == num_tuples_);
    assert(other_block->type() == type_);
    assert(other_block->bit_width() == bit_width_);

    const Avx2ScanColumnBlock<BIT_WIDTH>* block2 =
        static_cast<const Avx2ScanColumnBlock<BIT_WIDTH>*>(other_block);
    
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
void Avx2ScanColumnBlock<BIT_WIDTH>::ScanHelper1(
                        const Avx2ScanColumnBlock<BIT_WIDTH>* other_block,
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
void Avx2ScanColumnBlock<BIT_WIDTH>::ScanHelper2(
                        const Avx2ScanColumnBlock<BIT_WIDTH>* other_block,
                        BitVectorBlock* bvblock) const{
    //Do real work here
    AvxUnit shiftv;
    AvxUnit shufflev;
    AvxUnit code_maskv;

    SetUpMasks(shiftv, shufflev, code_maskv);

    AvxUnit signbit_maskv = avx_set1<uint32_t>(1 << 31);

    //For each 64 tuples that correspond to a bitvector word
    for(size_t offset=0; offset < num_tuples_; offset += kNumWordBits){
        WordUnit bv_word = 0ULL;
        size_t bv_word_id = offset / kNumWordBits;

        for(size_t offset2=0; offset2 < kNumWordBits; offset2 += kNumTuplePerSegment){
            size_t segment_id = (offset + offset2) / kNumTuplePerSegment;
            size_t byte_index = segment_id * kNumBytesPerSegment;
            AvxUnit datav = Unpack8(byte_index, shiftv, shufflev, code_maskv);
            AvxUnit datav2 = other_block->Unpack8(byte_index, shiftv, shufflev, code_maskv);

            if(32 == BIT_WIDTH){
                datav = avx_xor(datav, signbit_maskv);
                datav2 = avx_xor(datav2, signbit_maskv);
            }

            AvxUnit resultv = ScanKernel<CMP>(datav, datav2);
            uint32_t mmask = _mm256_movemask_ps(reinterpret_cast<__m256>(resultv));
            bv_word |= (static_cast<WordUnit>(mmask) << offset2);
        }

        //Merge into bit vector
        WordUnit x = bv_word;
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

template <size_t BIT_WIDTH>
void Avx2ScanColumnBlock<BIT_WIDTH>::BulkLoadArray(const WordUnit* codes,
                                                   size_t num,
                                                   size_t start_pos){
    assert(start_pos + num <= num_tuples_);
    for(size_t i = 0; i < num; i++){
        SetTuple(start_pos+i, codes[i]);
    }
}


//explicit specialization
template class Avx2ScanColumnBlock<1>;
template class Avx2ScanColumnBlock<2>;
template class Avx2ScanColumnBlock<3>;
template class Avx2ScanColumnBlock<4>;
template class Avx2ScanColumnBlock<5>;
template class Avx2ScanColumnBlock<6>;
template class Avx2ScanColumnBlock<7>;
template class Avx2ScanColumnBlock<8>;
template class Avx2ScanColumnBlock<9>;
template class Avx2ScanColumnBlock<10>;
template class Avx2ScanColumnBlock<11>;
template class Avx2ScanColumnBlock<12>;
template class Avx2ScanColumnBlock<13>;
template class Avx2ScanColumnBlock<14>;
template class Avx2ScanColumnBlock<15>;
template class Avx2ScanColumnBlock<16>;
template class Avx2ScanColumnBlock<17>;
template class Avx2ScanColumnBlock<18>;
template class Avx2ScanColumnBlock<19>;
template class Avx2ScanColumnBlock<20>;
template class Avx2ScanColumnBlock<21>;
template class Avx2ScanColumnBlock<22>;
template class Avx2ScanColumnBlock<23>;
template class Avx2ScanColumnBlock<24>;
template class Avx2ScanColumnBlock<25>;
template class Avx2ScanColumnBlock<26>;
template class Avx2ScanColumnBlock<27>;
template class Avx2ScanColumnBlock<28>;
template class Avx2ScanColumnBlock<29>;
template class Avx2ScanColumnBlock<30>;
template class Avx2ScanColumnBlock<31>;
template class Avx2ScanColumnBlock<32>;

}   //namespace
