#include "../../src/communication/infiniband/connection.h"

#include <iostream>
#include <fstream>

// std::this_thread::sleep_for(std::chrono::seconds(1));
#include <chrono>
#include <thread>

using namespace bbts::ib;

int main(int argc, char **argv) {
  std::string usage = "usage: " + std::string(argv[0]) + " <rank> <device name> <hosts file>";
  if(argc != 4) {
    std::cout << usage << std::endl;
    return 1;
  }

  int rank = std::stoi(argv[1]);

  std::vector<std::string> ips;
  {
    std::ifstream s = std::ifstream(argv[3]);
    if(!s.is_open()) {
      std::cout << "Could not open '" << argv[3] << "'" << std::endl;
      std::cout << usage << std::endl;
      return 1;
    }
    std::string l;
    while(std::getline(s, l)) {
      ips.push_back(l);
    }
    for(auto i: ips){
      std::cout << "IP: " << i << std::endl;
    }
  }

  size_t size = 10; // 104857600;
  size_t num = 100;

  std::vector<bytes_t> bs;
  if(rank != 2) {
    for(int i = 0; i != num; ++i) {
      float* data = new float[size];
      std::fill(data, data + size, 1.0*i);
      bs.push_back({ .data = (void*)data, .size = size});
    }
  }

  connection_t c(argv[2], rank, ips);
  std::cout << " CONNECTED " << std::endl;

  auto start = std::chrono::high_resolution_clock::now();
  std::vector<std::future<bool>> sends;
  std::vector<std::future<bytes_t>> recvs;
  if(rank == 0) {
    // send tag [100, 100+num)
    // recv tag [100+num,100+2*num)
    for(int i = 0; i != num; ++i) {
      sends.push_back(c.send_bytes(1, 100 + i, bs[i]));
      sends.push_back(c.send_bytes(2, 100 + i, bs[i]));
      recvs.push_back(c.recv_bytes(100 + num + i));
    }
  } else if(rank == 1) {
    // send tag [100, 100+num)
    // recv tag [100+num,100+2*num)
    for(int i = 0; i != num; ++i) {
      sends.push_back(c.send_bytes(0, 100 +num + i, bs[i]));
      sends.push_back(c.send_bytes(2, 100 +num + i, bs[i]));
      recvs.push_back(c.recv_bytes(100 + i));
    }
  } else if(rank == 2) {
    // everything sends here
    for(int i = 0; i != num; ++i) {
      recvs.push_back(c.recv_bytes(100 + i));
      recvs.push_back(c.recv_bytes(100 + num + i));
    }
  }

  for(auto& fut: recvs) {
    bs.push_back(fut.get());
  }
  for(auto& fut: sends) {
    fut.get();
  }
  //auto stop = std::chrono::high_resolution_clock::now();
  //size_t total = size*num*32*2;
  //auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  //double gbs = (1.0e6*total)/(1.0e9*duration.count());
  //std::cout << std::endl << gbs << " GBs" << std::endl;

  for(bytes_t b: bs) {
    float* data = (float*)b.data;
    std::cout << " @ " << data[0];
    delete[] data;
  }
  std::cout << std::endl;
}
