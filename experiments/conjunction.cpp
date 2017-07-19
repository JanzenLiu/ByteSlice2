#include    <iostream>
#include    <iomanip>
#include    <fstream>
#include    <unistd.h>
#include    <cstring>
#include    <cstdlib>
#include    <ctime>
#include    <vector>

#include    "include/hybrid_timer.h"

#include    "include/types.h"
#include    "include/column.h"
#include    "include/bitvector.h"
#include    "include/pipeline_scan.h"

using namespace byteslice;

int main(int argc, char* argv[]){

    //default options
    ColumnType type = ColumnType::kByteSlicePadRight;
    size_t num_rows = 100*1024*1024;
    size_t code_length = 16;
    double selectivity = 0.1;
    Comparator comparator = Comparator::kLess;
    size_t repeat = 1;
    const char* filename = "conjunction.data";
    std::ofstream of;

    //get options:
    //t - column type; s - column size; b - bit width
    //f - selectivity; r - repeat; p - predicate
    //o - output file
    int c;
    while((c = getopt(argc, argv, "s:f:r:o:p:")) != -1){
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
    
    HybridTimer t1;

    size_t code_length1 = 14;
    size_t code_length2 = 16;

    //print options
    of << "# "
        << "ColumnType= " << type 
        << "Predicate= " << comparator
        << " num_rows= " << num_rows 
        << " selectivity= " << selectivity
        << " repeat= " << repeat << std::endl;

    of << "# selectivity  BS-pipeline  BS-columnar  "
       << "BS(1)  BS(2)  " << std::endl;


    std::srand(std::time(0));
    /*-------------------------------------------------------------*/
    //Iterate thourgh different selectivity
    std::vector<double> selectivity_vec = {1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001};
    //for(selectivity = 1.0; selectivity >= 0.001; selectivity /= 2){
    for(double selectivity : selectivity_vec){

        uint64_t cycles_pipeline = 0, cycles_columnar1 = 0, cycles_columnar2 = 0;
        uint64_t cycles_bwv1 = 0, cycles_bwv2 = 0;

        const WordUnit mask1 = (1ULL << code_length1) - 1;
        const WordUnit mask2 = (1ULL << code_length2) - 1;
        WordUnit literal1 = static_cast<WordUnit>(mask1 * selectivity);
        WordUnit literal2 = static_cast<WordUnit>(mask2 * 0.25);

        //Prepare the material
        Column* column1 = new Column(ColumnType::kByteSlicePadRight, code_length1, num_rows);
        Column* column2 = new Column(ColumnType::kByteSlicePadRight, code_length2, num_rows);
        //Column* bwv1 = new Column(ColumnType::kBitSlice, code_length, num_rows);
        //Column* bwv2 = new Column(ColumnType::kBitSlice, code_length, num_rows);
        BitVector* bitvector = new BitVector(num_rows);

        //populate the column with random data
        for(size_t i = 0; i < num_rows; i++){
            WordUnit code1 = std::rand() & mask1;
            column1->SetTuple(i, code1);
            //bwv1->SetTuple(i, code);
        }
        for(size_t i = 0; i < num_rows; i++){
            WordUnit code2 = std::rand() & mask2;
            column2->SetTuple(i, code2);
            //bwv2->SetTuple(i, code);
        }


        for(size_t run=0; run < repeat; run++){
            /*--Pipeline---------------------------*/
            bitvector->SetOnes();
            PipelineScan scan;
            scan.AddPredicate(AtomPredicate(column1, comparator, literal1));
            scan.AddPredicate(AtomPredicate(column2, comparator, literal2));

            t1.Start();
            scan.ExecuteBlockwise(bitvector);
            t1.Stop();

            cycles_pipeline += t1.GetNumCycles();
            //std::cout << bitvector->CountOnes() << "\t";
            /*------------------------------------*/

            /*--Columnar--------------------------*/
            bitvector->SetOnes();
            
            t1.Start();
            column1->Scan(comparator, literal1, bitvector, Bitwise::kSet);
            t1.Stop();

            cycles_columnar1 += t1.GetNumCycles();
            //std::cout << bitvector->CountOnes() << "\t";

            t1.Start();
            column2->Scan(comparator, literal2, bitvector, Bitwise::kAnd);
            t1.Stop();

            cycles_columnar2 += t1.GetNumCycles();
            //std::cout << bitvector->CountOnes() << std::endl;
            /*--------------------------------*/

            /*--BwV: Columnar--------------------*/
            /*
            bitvector->SetOnes();

            pm.Start();
            t1.Start();
            bwv1->Scan(comparator, literal1, bitvector, Bitwise::kSet);
            pm.Stop();
            t1.Stop();

            cycles_bwv1 += t1.GetNumCycles();
            std::cout << bitvector->CountOnes() << "\t";

            pm.Start();
            t1.Start();
            bwv2->Scan(comparator, literal2, bitvector, Bitwise::kAnd);
            pm.Stop();
            t1.Stop();

            cycles_bwv2 += t1.GetNumCycles();
            std::cout << bitvector->CountOnes() << std::endl;
            */
 
        }
    
        delete bitvector;
        delete column1;
        delete column2;
        //delete bwv1;
        //delete bwv2;

        of << std::fixed;
        of << std::setprecision(8);
        of << std::left;
        of << selectivity << "\t"
            << double(cycles_pipeline / repeat) / num_rows << "\t"
            << double((cycles_columnar1 + cycles_columnar2) / repeat) / num_rows << "\t"
            << double(cycles_columnar1 / repeat) / num_rows << "\t"
            << double(cycles_columnar2 / repeat) / num_rows << "\t"
            //<< double((cycles_bwv1 + cycles_bwv2) / repeat) / num_rows << "\t"
            //<< double(cycles_bwv1 / repeat) / num_rows << "\t"
            //<< double(cycles_bwv2 / repeat) / num_rows << "\t"
            << std::endl;
        of.flush();
    }
    /*-------------------------------------------------------------*/

    of.close();
}

