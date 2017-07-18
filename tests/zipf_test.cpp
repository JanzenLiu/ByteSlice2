#include    "include/zipf.h"
#include    "gtest/gtest.h"
#include    <iostream>
#include    <cstdlib>
#include    <cstdint>
#include    <cstdio>
#include    <vector>

namespace byteslice {
    
class ZipfTest: public ::testing::Test{
public:
    virtual void SetUp(){
        data_.assign(num_, 0);
    }
protected:
    typedef uint32_t T;
    size_t num_ = 20;
    std::vector<T> data_;
};

TEST_F(ZipfTest, Simple){
    size_t N = 8*data_.size();
    double z= 1.0;
    T x = 0;
    auto next = [&x](){ return x++;};
    fill_zipf(data_.begin(), data_.end(),
              N, z,
              next);
    std::copy(data_.begin(), data_.end(), std::ostream_iterator<T>(std::cout, " "));
    std::cout << std::endl;
    std::random_shuffle(data_.begin(), data_.end());
    std::copy(data_.begin(), data_.end(), std::ostream_iterator<T>(std::cout, " "));
    std::cout << std::endl;
}

}   // namespace