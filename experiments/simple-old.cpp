#include    <iostream>
#include    <unistd.h>
#include    <cstring>
#include    <cstdlib>
#include    <ctime>
#include    <cassert>
#include    <omp.h>
#include    <algorithm>

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
    double zipf = 0.0;
    double lfraction = -1.0;

    //get options:
    //t - column type; s - column size; b - bit width
    //f - selectivity; r - repeat; p - predicate
    int c;
    while((c = getopt(argc, argv, "l:t:s:b:f:r:p:z:h")) != -1){
        switch(c){
            case 'h':
                std::cout << "Usage: " << argv[0]
                        << " [-t <column type = avx2|bits|bytes|dbs>]"
                        << " [-s <size>]"
                        << " [-b <bit width>]"
                        << " [-f <selectivity>]"
                        << " [-r <repetition>]"
                        << " [-p <predicate = lt|le|gt|ge|eq|ne>]"
                        << " [-z <zipf>]"
                        << " [-l <literal fraction (of domain)]"
                        << std::endl;
                exit(0);
            case 'z':
                zipf = atof(optarg);
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
    WordUnit literal = static_cast<WordUnit>(mask * selectivity);

    std::srand(std::time(0));

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

    size_t expected_result = 0;
    
    // populate column with (zipfian) random data
    // and obtain literal
    WordUnit* raw = new WordUnit[num_rows];
    WordUnit x = 0ULL;
    auto next = [&x, mask]()->WordUnit{ return mask & x++;};
    fill_zipf(raw, raw + num_rows, mask+1ULL, zipf, next);
    literal = raw[static_cast<size_t>(num_rows * selectivity)];
    std::random_shuffle(raw, raw + num_rows);
    column->BulkLoadArray(raw, num_rows);
    delete [] raw;

    if(lfraction >= 0.0 && lfraction <= 1.0){
        literal = static_cast<WordUnit>(lfraction * mask);
    }
    
    std::cout << "Literal = " << std::hex << literal << std::dec << std::endl;
 
    //print options
    std::cout << "ColumnType=" << type 
            << " num_rows=" << num_rows 
            << " code_length=" << code_length
            << " selectivity=" << selectivity
            << " predicate=" << comparator
            << " repeat= " << repeat
            << " zipf=" << zipf
            << " lfraction=" << lfraction
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


    assert(expected_result == bitvector->CountOnes());

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

