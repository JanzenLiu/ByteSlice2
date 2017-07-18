#include    "gtest/gtest.h"
#include    "include/avx-utility.h"
#include    <cstdlib>
#include    <cstdint>
#include    <cstdio>

namespace byteslice{

class AvxUtilityTest: public ::testing::Test{
public:
    virtual void SetUp(){
        std::srand(std::time(0));
    }

    virtual void TearDown(){
    }
};

TEST_F(AvxUtilityTest, InverseMovemask8){
    int mmask = 0x0f030107;
    __m256i expected_mask = _mm256_set_epi8(
            0, 0, 0, 0, -1, -1, -1, -1,
            0, 0, 0, 0, 0, 0, -1, -1,
            0, 0, 0, 0, 0, 0, 0, -1,
            0, 0, 0, 0, 0, -1, -1, -1);
    __m256i ret_mask = inverse_movemask_epi8(mmask);
    __m256i compare = _mm256_xor_si256(expected_mask, ret_mask);
    EXPECT_TRUE(avx_iszero(compare));
}


TEST_F(AvxUtilityTest, Movemask16){
    __m256i d = _mm256_set_epi16(0x1,0x2,0x3,0x4,0x5,0x6,0x7,0x8,0x1,0x2,0x3,0x4,0x5,0x6,0x7,0x8);
    __m256i l = _mm256_set1_epi16(0x5);
    __m256i mask = avx_cmplt<uint16_t>(d, l);
    int mmask = movemask_epi16(mask);
    EXPECT_EQ(0xf0f0, mmask);
}

TEST_F(AvxUtilityTest, DISABLED_Endian){
    __m256i d = _mm256_set_epi64x(0x12345678, 0xaabbccdd, 0xffffffff, 0xefefefef);
    unsigned char *p = (unsigned char*)(&d);
    for(int i=0; i<32; i++){
        printf("%x ", p[i]);
        if((i+1)%8 == 0)
            printf("\n");
    }
    int64_t *q = (int64_t*)(&d);
    for(int i=0; i<4; i++){
        printf("%x ", q[i]);
    }
    printf("\n");

}

TEST_F(AvxUtilityTest, Flip){
    //byte
    for(size_t i = 0; i < 20; i++){
        unsigned char a = std::rand()%256;
        unsigned char b = std::rand()%256;
        signed char x = static_cast<signed char>(FLIP<unsigned char>(a));
        signed char y = static_cast<signed char>(FLIP<unsigned char>(b));
        EXPECT_TRUE((a<b) == (x<y));
        EXPECT_TRUE((a==b) == (a==b));
    }
}

TEST_F(AvxUtilityTest, Set1){
    //byte
    {
        unsigned char a1 = 0x3e;
        __m256i m1 = avx_set1<unsigned char>(a1);
        int *p = (int*)(&m1);
        EXPECT_EQ(0x3e3e3e3e, *p);
    }

    //double byte
    {
        uint16_t a1 = 0x12ae;
        __m256i m1 = avx_set1<uint16_t>(a1);
        uint16_t *p = (uint16_t*)(&m1);
        EXPECT_EQ(0x12ae, p[1]);
        uint8_t *q = (uint8_t*)(&m1);
        EXPECT_EQ(0xae, q[0]);
        EXPECT_EQ(0x12, q[1]);
    }

}

TEST_F(AvxUtilityTest, Compare){
    //int32
    __m256i a = _mm256_set_epi32(0xef, 0xcd, 0xab, 0x89, 0x67, 0x45, 0x23, 0x01);
    __m256i b = avx_set1<int>(0x89);
    int mmask;

    __m256i m_lt = avx_cmplt<int>(a, b);
    mmask = _mm256_movemask_ps((__m256)m_lt);
    EXPECT_EQ(0x0f, mmask);

    __m256i m_gt = avx_cmpgt<int>(a, b);
    mmask = _mm256_movemask_ps(__m256(m_gt));
    EXPECT_EQ(0xe0, mmask);
    
    __m256i m_eq = avx_cmpeq<int>(a, b);
    mmask = _mm256_movemask_ps(__m256(m_eq));
    EXPECT_EQ(0x10, mmask);

}

}   //namespace
