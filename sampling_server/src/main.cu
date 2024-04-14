#include "server.h"
#include <stdlib.h>
#include <iostream>

int main(int argc, char** argv){

    std::cout<<"Start Sampling Server\n";
    Server* server = NewGPUServer();
    std::vector<int> fanout;
    fanout.push_back(25);
    fanout.push_back(10);
    std::cout << "Dynamic? " << atoi(argv[3]) << "\n";
    server->Initialize(atoi(argv[1]), fanout, 1, atoi(argv[3]));//gpu number, default 1; in memory, default true
    server->PreSc(atoi(argv[2]));//cache aggregate mode, default 0
    server->Run();
    server->Finalize();
    
}