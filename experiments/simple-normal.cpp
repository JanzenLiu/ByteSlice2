#include    <iostream>
#include    <unistd.h>
#include    <cstring>
#include    <cstdlib>
#include    <ctime>
#include    <cassert>
#include    <omp.h>
#include    <algorithm>
#include    <random>
#include    <functional>
#include    <fstream>
#include    <string>

#include    "util/system_perf_monitor.h"
#include    "include/hybrid_timer.h"

#include    "include/types.h"
#include    "include/column.h"
#include    "include/bitvector.h"
#include    "include/zipf.h"

using namespace byteslice;

int main(int argc, char* argv[]){

    //default options
    ColumnType type = ColumnType::kByteSlicePadRight;
    size_t num_rows = 512*1024*1024;
    size_t code_length = 16;
    double selectivity = 0.1;
    Comparator comparator = Comparator::kLess;
    size_t repeat = 20;
    double lfraction = -1.0;
    WordUnit stddev = 100;
    WordUnit mean = (1ULL << code_length)/2;
    std::string filename = std::string("");

    //get options:
    //t - column type; s - column size; b - bit width
    //f - selectivity; r - repeat; p - predicate
    int c;
    while((c = getopt(argc, argv, "i:l:t:s:b:f:r:p:z:h")) != -1){
        switch(c){
            case 'h':
                std::cout << "Usage: " << argv[0]
                        << " [-t <column type = avx2|bits|bytes|dbs>]"
                        << " [-s <size>]"
                        << " [-b <bit width>]"
                        << " [-f <selectivity>]"
                        << " [-r <repetition>]"
                        << " [-p <predicate = lt|le|gt|ge|eq|ne>]"
                        << " [-z <stddev>]"
                        << " [-l <literal fraction (of domain)]"
                        << " [-i <input file>]"
                        << std::endl;
                exit(0);
            case 'i':
                filename = std::string(optarg);
                break;
            case 'z':
                stddev = atoi(optarg);
                break;
            case 'l':
                lfraction = atof(optarg);
                break;
            case 't':
                if(0 == strcmp(optarg, "n"))
                    type = ColumnType::kNaive;
                else if(0 == strcmp(optarg, "navx"))
                    type = ColumnType::kNaiveAvx;
                else if(0 == strcmp(optarg, "hbp"))
                    type = ColumnType::kHbp;
                else if(0 == strcmp(optarg, "bits"))
                    type = ColumnType::kBitSlice;
                else if(0 == strcmp(optarg, "bytes"))
                    type = ColumnType::kByteSlicePadRight;
                else if(0 == strcmp(optarg, "hs"))
                    type = ColumnType::kHybridSlice;
                else if(0 == strcmp(optarg, "nnavx"))
                    type = ColumnType::kNumpsAvx;
                else if(0 == strcmp(optarg, "shs"))
                    type = ColumnType::kSmartHybridSlice;
                else if(0 == strcmp(optarg, "avx2"))
                    type = ColumnType::kAvx2Scan;
                else if(0 == strcmp(optarg, "vbp"))
                    type = ColumnType::kVbp;
                else if(0 == strcmp(optarg, "dbs"))
                    type = ColumnType::kDualByteSlicePadRight;
                else{
                    std::cerr << "Unknown column type:" << optarg << std::endl;
                    exit(1);
                }
                break;
            case 'p':
                if(0 == strcmp(optarg, "lt"))
                    comparator = Comparator::kLess;
                else if(0 == strcmp(optarg, "le"))
                    comparator = Comparator::kLessEqual;
                else if(0 == strcmp(optarg, "gt"))
                    comparator = Comparator::kGreater;
                else if(0 == strcmp(optarg, "ge"))
                    comparator = Comparator::kGreaterEqual;
                else if(0 == strcmp(optarg, "eq"))
                    comparator = Comparator::kEqual;
                else if(0 == strcmp(optarg, "ne"))
                    comparator = Comparator::kInequal;
                else{
                    std::cerr << "Unknown predicate: " << optarg << std::endl;
                    exit(1);
                }
                break;
            case 's':
                num_rows = atoi(optarg);
                break;
            case 'b':
                code_length = atoi(optarg);
                break;
            case 'f':
                selectivity = atof(optarg);
                break;
            case 'r':
                repeat = atoi(optarg);
                break;
        }
    }

    const WordUnit mask = (1ULL << code_length) - 1;
    WordUnit literal;
    mean = mask/2;

    uint64_t sum_l2_misses = 0;
    uint64_t sum_l3_misses = 0;
    double sum_ipc = 0;
    uint64_t sum_insn = 0;
    double sum_timeofday = 0;
    uint64_t sum_rdtsc = 0;
    uint64_t sum_bytes_read = 0;
    uint64_t sum_bytes_write = 0;

    //Init PCM
    SystemPerfMonitor pm;
    pm.Init();

    HybridTimer t1;

    //Prepare the material
    Column* column = new Column(type, code_length, num_rows);
    BitVector* bitvector = new BitVector(column);

    
    // populate column with normal distributed random data
    // and obtain literal
    WordUnit* raw = new WordUnit[num_rows];
    // if an input file is provided, read from file
    // otherwise generate data during runtime
    if(filename != ""){
        std::cout << "Reading from file ... " << filename << std::endl;
        std::ifstream infile(filename, std::ifstream::in);
        for(size_t i=0; i < num_rows; i++){
            WordUnit val;
            infile >> val;
            raw[i] = val & mask;
        }
        infile.close();
    }
    else{
        auto dice = std::bind(std::normal_distribution<double>(0,1),
                          std::default_random_engine(std::time(nullptr)));
        for(size_t i=0; i < num_rows; i++){
            double val = mean + dice() * stddev;
            val = (val < 0)? 0:val;
            val = (val > mask)? mask: val;
            raw[i] = static_cast<WordUnit>(val) & mask;
        }
    }

    if(lfraction >= 0.0 && lfraction <= 1.0){
        literal = static_cast<WordUnit>(lfraction * mask);
    }
    else{
        std::sort(raw, raw + num_rows);
        literal = raw[static_cast<size_t>(num_rows * selectivity)];
        std::random_shuffle(raw, raw + num_rows);
    }
    
    column->BulkLoadArray(raw, num_rows);
    delete [] raw;
    
    std::cout << "Literal = " << std::hex << literal << std::dec << std::endl;
 
    //print options
    std::cout << "ColumnType=" << type 
            << " num_rows=" << num_rows 
            << " code_length=" << code_length
            << " selectivity=" << selectivity
            << " predicate=" << comparator
            << " repeat= " << repeat
            << " lfraction=" << lfraction
            << " stddev=" << stddev
            << " num_threads=" << omp_get_max_threads() << std::endl;
    
    std::cout << "#timeofday\t rdtsc/tuple\t insn/tuple\t";
    std::cout << "L2Miss/tuple\t L3Miss/tuple\t IPC";
    std::cout << "Bandwidth(R)\t Bandwidth(W) \t" 
              << "Bandwidth(GB/s)" << std::endl;
   
    
    pm.Start();
    t1.Start();
    for(size_t run=0; run < repeat; run++){
    	//std::cout << "Kick start!" << std::endl;
        column->Scan(comparator, literal, bitvector, Bitwise::kSet);
    }
    pm.Stop();
    t1.Stop();

    sum_timeofday += t1.GetSeconds();
    sum_rdtsc += t1.GetNumCycles();
    sum_l2_misses += pm.GetL2CacheMisses();
    sum_l3_misses += pm.GetL3CacheMisses();
    sum_ipc += pm.GetIPC();
    sum_insn += pm.GetRetiredInstructions();
    sum_bytes_read += pm.GetBytesReadFromMC();
    sum_bytes_write += pm.GetBytesWrittenToMC();


    //assert(expected_result == bitvector->CountOnes());

    delete bitvector;
    delete column;


    std::cout << std::fixed << std::setprecision(3) << std::left;
    std::cout 
            << sum_timeofday / repeat << "\t"
            << double(sum_rdtsc/repeat) / num_rows << "\t"
            << double(sum_insn/repeat) / num_rows << "\t"
            << double(sum_l2_misses/repeat) / num_rows << "\t"
            << double(sum_l3_misses/repeat) / num_rows << "\t"
            << sum_ipc << "\t"
            << sum_bytes_read / sum_timeofday << "\t"
            << sum_bytes_write / sum_timeofday << "\t"
            << ((sum_bytes_read + sum_bytes_write) / sum_timeofday) / (1e9)
            << std::endl;

    pm.Destroy();
}

