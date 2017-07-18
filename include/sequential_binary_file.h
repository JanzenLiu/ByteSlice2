#ifndef	SEQUENTIAL_BINARY_FILE_H
#define	SEQUENTIAL_BINARY_FILE_H

#include    <cstdio>
#include    <string>

namespace byteslice{

class SequentialReadBinaryFile{
public:
    bool Open(const std::string filename);
    bool Close();
    size_t Read(void* buf, size_t size) const;
    bool IsEnd();

private:
    FILE* file_ = NULL;
    std::string filename_;

};

class SequentialWriteBinaryFile{
public:
    bool Open(const std::string filename);
    bool Close();
    size_t Append(const void* data, size_t size);
    bool Flush();

private:
    FILE* file_ = NULL;
    std::string filename_;

};

}   //namespace

#endif	//SEQUENTIAL_BINARY_FILE_H
