#include <iostream>
#include <string>
#include <vector>
#include <stdlib.h>

int main()
{
    std::vector<std::string> params;
    std::string o1 = "1.1";
    std::string o2 = "1.2";
    std::string o3 = "2.1";
    std::string o4 = "2.2";
    
    params.push_back(o1);
    params.push_back(o2);
    params.push_back(o3);
    params.push_back(o4);
    
    // test the convertion from vector of string to column and byte
    for(size_t i = 0; i < params.size(); i++){
        std::string param = params[i];
        size_t pos;
        for(pos = 0; pos < param.length(); pos++)
            if(param[pos] == '.')
                break;
        size_t col = atoi(param.substr(0,pos).c_str());
        size_t byte = atoi(param.substr(pos + 1, param.length()-1-pos).c_str());
        std::cout << "column#" << col << " "
                << "byte#" << byte << std::endl;
    } 
}