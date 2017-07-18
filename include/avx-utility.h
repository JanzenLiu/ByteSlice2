#ifndef AVX_UTILITY_H
#define AVX_UTILITY_H

#include    <x86intrin.h>

namespace byteslice{


/**
  T should be uint8/16/32/64_t
*/
template <typename T>
inline T FLIP(T value){
    constexpr T offset = (static_cast<T>(1) << (sizeof(T)*8 - 1));
    return static_cast<T>(value ^ offset);
}

//Shift right logical
template <typename T>
inline __m256i avx_srli(const __m256i &a, int imm){
    switch(sizeof(T)){
        case 2:
            return _mm256_srli_epi16(a, imm);
        case 4:
            return _mm256_srli_epi32(a, imm);
        case 8:
            return _mm256_srli_epi64(a, imm);
    }
}

template <typename T>
inline __m256i avx_cmplt(const __m256i &a, const __m256i &b){
    switch(sizeof(T)){
        case 1:
            return _mm256_cmpgt_epi8(b, a);
        case 2:
            return _mm256_cmpgt_epi16(b, a);
        case 4:
            return _mm256_cmpgt_epi32(b, a);
        case 8:
            return _mm256_cmpgt_epi64(b, a);
    }
}

template <typename T>
inline __m256i avx_cmpgt(const __m256i &a, const __m256i &b){
    switch(sizeof(T)){
        case 1:
            return _mm256_cmpgt_epi8(a, b);
        case 2:
            return _mm256_cmpgt_epi16(a, b);
        case 4:
            return _mm256_cmpgt_epi32(a, b);
        case 8:
            return _mm256_cmpgt_epi64(a, b);
    }
}

template <typename T>
inline __m256i avx_cmpeq(const __m256i &a, const __m256i &b){
    switch(sizeof(T)){
        case 1:
            return _mm256_cmpeq_epi8(b, a);
        case 2:
            return _mm256_cmpeq_epi16(b, a);
        case 4:
            return _mm256_cmpeq_epi32(b, a);
        case 8:
            return _mm256_cmpeq_epi64(b, a);
    }
}

template <typename T>
inline __m256i avx_set1(T a){
    switch(sizeof(T)){
        case 1:
            return _mm256_set1_epi8(static_cast<int8_t>(a));
        case 2:
            return _mm256_set1_epi16(static_cast<int16_t>(a));
        case 4:
            return _mm256_set1_epi32(static_cast<int32_t>(a));
        case 8:
            return _mm256_set1_epi64x(static_cast<int64_t>(a));
    }
}

inline __m256i avx_zero(){
    return _mm256_setzero_si256();
}

inline __m256i avx_ones(){
    return _mm256_set1_epi64x(-1ULL);
    //return _mm256_set1_epi8(0xff);
}

//some alias
inline __m256i avx_and(const __m256i &a, const __m256i &b){
    return _mm256_and_si256(a, b);
}

inline __m256i avx_or(const __m256i &a, const __m256i &b){
    return _mm256_or_si256(a, b);
}

inline __m256i avx_xor(const __m256i &a, const __m256i &b){
    return _mm256_xor_si256(a, b);
}

inline __m256i avx_not(const __m256i &a){
    return _mm256_xor_si256(a, _mm256_set1_epi8(0xff));
}
//notice: returns (NOT a) AND b
inline __m256i avx_andnot(const __m256i &a, const __m256i &b){
    return _mm256_andnot_si256(a, b);
}

inline bool avx_iszero(const __m256i &a){
    //return _mm256_testc_si256(_mm256_setzero_si256(), a);
    return _mm256_testz_si256(a, a);
}

//AVX2 doesn't provide movemask of 16-bit
//Let's implement it!
inline int movemask_epi16(const __m256i &a){
//    int t = _mm256_movemask_epi8(a);
//    int ret = 0;
//    for(size_t i = 0; i < 16; i++){
//        ret |= ((t & 0x1) << i);
//        t >>= 2;
//    }
//    return ret;
    
    int ret = 0;
    ret |= _mm256_movemask_ps(reinterpret_cast<__m256>(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(a, 0))));
    ret |= _mm256_movemask_ps(reinterpret_cast<__m256>(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(a, 1)))) << 8;
    return ret;

}

inline __m256i inverse_movemask_epi8(uint32_t mmask){
    uint8_t byte0 = static_cast<uint8_t>(mmask);
    uint8_t byte1 = static_cast<uint8_t>(mmask >> 8);
    uint8_t byte2 = static_cast<uint8_t>(mmask >> 16);
    uint8_t byte3 = static_cast<uint8_t>(mmask >> 24);

    __m256i t = _mm256_set_epi8(
            byte3, byte3, byte3, byte3, byte3, byte3, byte3, byte3,
            byte2, byte2, byte2, byte2, byte2, byte2, byte2, byte2,
            byte1, byte1, byte1, byte1, byte1, byte1, byte1, byte1,
            byte0, byte0, byte0, byte0, byte0, byte0, byte0, byte0);

    __m256i mask = _mm256_set_epi8(
            0x80, 0x40, 0x20, 0x10, 0x8, 0x4, 0x2, 0x1,
            0x80, 0x40, 0x20, 0x10, 0x8, 0x4, 0x2, 0x1,
            0x80, 0x40, 0x20, 0x10, 0x8, 0x4, 0x2, 0x1,
            0x80, 0x40, 0x20, 0x10, 0x8, 0x4, 0x2, 0x1);

    __m256i ret = _mm256_cmpeq_epi8(mask, _mm256_and_si256(mask, t));
    
    return ret;

}

}   //namespace

#endif  //AVX_UTILITY_H
