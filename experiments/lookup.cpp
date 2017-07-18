#include    <iostream>
#include    <iomanip>
#include    <fstream>
#include    <unistd.h>
#include    <cstring>
#include    <cstdlib>
#include    <ctime>

#include    "util/system_perf_monitor.h"
#include    "include/hybrid_timer.h"

#include    "include/types.h"
#include    "include/column.h"
#include    "include/bitvector.h"

using namespace byteslice;

int main(int argc, char* argv[]){

    //default options
    ColumnType type = ColumnType::kByteSlicePadRight;
    size_t num_rows = 512*1024*1024;
    size_t code_length = 16;
    double selectivity = 0.1;
    Comparator comparator = Comparator::kLess;
    size_t repeat = 1024*1024*100;
    const char* filename = "lookup.data";
    std::ofstream of;


    //get options:
    //t - column type; s - column size; b - bit width
    //f - selectivity; r - repeat; p - predicate
    //o - output file
    int c;
    while((c = getopt(argc, argv, "t:s:f:r:o:p:")) != -1){
        switch(c){
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
            case 'o':
                filename = optarg;
                break;
        }
    }

    of.open(filename, std::ofstream::out);

    //Init PCM
    SystemPerfMonitor pm;
    pm.Init();
    
    HybridTimer t1;

    //print options
    of << "# "
        << "ColumnType= " << type 
        << "Predicate= " << comparator
        << " num_rows= " << num_rows 
        << " selectivity= " << selectivity
        << " repeat= " << repeat << std::endl;

    of << "#bit_width\t timeofday\t rdtsc/tuple\t insn/tuple\t";
    of << "L2Miss/tuple\t L3Miss/tuple\t IPC\t";
    of << "Bandwidth(R)\t Bandwidth(W)" << std::endl;


    std::srand(std::time(0));
    /*-------------------------------------------------------------*/
    //Iterate thourgh different code_length's

    for(code_length = 2; code_length <= 32; code_length += 2){

        const WordUnit mask = (1ULL << code_length) - 1;
    
        double sum_timeofday = 0;
        uint64_t sum_rdtsc = 0;
        uint64_t sum_l2_misses = 0;
        uint64_t sum_l3_misses = 0;
        double sum_ipc = 0;
        uint64_t sum_insn = 0;
        uint64_t sum_bytes_read = 0;
        uint64_t sum_bytes_write = 0;

        //Prepare the material
        Column* column = new Column(type, code_length, num_rows);
        BitVector* bitvector = new BitVector(column);
        WordUnit* data = new WordUnit[num_rows];

        //populate the column with random data
        for(size_t i = 0; i < num_rows; i++){
            data[i] = std::rand() & mask;
        }
 
        pm.Start();
        t1.Start();

        WordUnit dummy = 0;
        for(size_t run=0; run < repeat; run++){
   
            size_t id = std::rand() % num_rows;
            dummy += column->GetTuple(id);
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

        delete data;
        delete bitvector;
        delete column;

        of << std::fixed;
        of << std::setprecision(8);
        of << std::left;
        of << code_length << "\t"
           << sum_timeofday / repeat << "\t"
           << double(sum_rdtsc) / repeat << "\t"
           << double(sum_insn) / repeat << "\t"
           << double(sum_l2_misses) / repeat << "\t"
           << double(sum_l3_misses) / repeat << "\t"
           << sum_ipc/repeat << "\t"
           << sum_bytes_read / sum_timeofday << "\t"
           << sum_bytes_write / sum_timeofday << "\t"
           << std::endl;
        of.flush();
    }
    /*-------------------------------------------------------------*/

    pm.Destroy();
    of.close();
}

