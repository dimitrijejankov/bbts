#include "../../src/communication/infiniband/connection.h"

#include <iostream>
#include <fstream>

// std::this_thread::sleep_for(std::chrono::seconds(1));
#include <chrono>
#include <thread>

using namespace bbts::ib;

//template <typename GenConnection>
//void three_node(int rank, size_t size, size_t num, GenConnection g) {
//  std::vector<bytes_t> bs;
//  for(int i = 0; i != 2*num; ++i) {
//    float* data = new float[size];
//    if(i < num && rank == 0) {
//      std::fill(data, data + size, 1000 + i);
//    }
//    if(i >= num && rank == 1) {
//      std::fill(data, data + size, 1000 + i);
//    }
//    bs.push_back({data, size});
//  }
//
//  std::vector<std::future<bool>> futs;
//
//  connection_t c = g();
//  std::cout << " CONNECTED " << std::endl;
//
//  if(rank == 0) {
//    for(int i = 0; i != num; ++i) {
//      futs.push_back(c.send_bytes_wait(1, 100+i, bs[i]));
//      futs.push_back(c.send_bytes_wait(2, 100+i, bs[i]));
//      auto idx = futs.size();
//      futs[idx-1].wait();
//      futs[idx-2].wait();
//    }
//    for(int i = num; i != 2*num; ++i) {
//      futs.push_back(c.recv_bytes_wait(100+i, bs[i]));
//      auto idx = futs.size();
//      futs[idx-1].wait();
//    }
//  } else if(rank == 1) {
//    for(int i = 0; i != num; ++i) {
//      futs.push_back(c.recv_bytes_wait(100+i, bs[i]));
//      auto idx = futs.size();
//      futs[idx-1].wait();
//    }
//    for(int i = num; i != 2*num; ++i) {
//      futs.push_back(c.send_bytes_wait(0, 100+i, bs[i]));
//      futs.push_back(c.send_bytes_wait(2, 100+i, bs[i]));
//      auto idx = futs.size();
//      futs[idx-1].wait();
//      futs[idx-2].wait();
//    }
//  } else if(rank == 2) {
//    // everything sends here
//    for(int i = 0; i != 2*num; ++i) {
//      futs.push_back(c.recv_bytes_wait(100 + i, bs[i]));
//      auto idx = futs.size();
//      futs[idx-1].wait();
//    }
//  }
//
//  for(auto& fut: futs) {
//    fut.get();
//  }
//
//  //for(bytes_t b: bs) {
//  //  float* data = (float*)b.data;
//  //  std::cout << " @ " << data[0];
//  //  delete[] data;
//  //}
//  //std::cout << std::endl;
//
//}

//                   use_send_mr
//  no bytes
//  use_recv_bytes
//  use_recv_memory
template <typename GenConnection>
void two_node_general(
  int rank,
  size_t size,
  size_t num_per_channel,
  size_t num_channel,
  GenConnection g,
  bool use_send_mr ,
  int use_recv_mr,
  bool print_results)
{
  connection_t c = g();
  std::cout << "CONNECTED" << std::endl;
  if(rank != 0 && rank != 1) {
    return;
  }
  auto other_rank = rank == 0 ? 1 : 0;
  auto num = num_per_channel*num_channel;

  int32_t* data_send = new int32_t[size*num];
  int32_t* data_recv = new int32_t[size*num];
  ibv_mr* mr_send = nullptr;
  ibv_mr* mr_recv = nullptr;

  if(use_send_mr) {
    std::cout << "Pinning " << (size*num*sizeof(int32_t)*1.0e-6) << "MB" << std::endl;
    mr_send = ibv_reg_mr(
      c.get_protection_domain(),
      data_send, size*num*sizeof(int32_t),
      IBV_ACCESS_LOCAL_WRITE);
    if(!mr_send) {
      throw std::runtime_error("two node couldn't register data_send");
    }
  }

  if(use_recv_mr) {
    std::cout << "Pinning " << (size*num*sizeof(int32_t)*1.0e-6) << "MB" << std::endl;
    mr_recv = ibv_reg_mr(
      c.get_protection_domain(),
      data_recv, size*num*sizeof(int32_t),
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if(!mr_recv) {
      throw std::runtime_error("two node couldn't register data_recv");
    }
  }

  std::vector<bytes_t> send_bs;
  std::vector<bytes_t> recv_bs;
  for(int i = 0; i != num; ++i) {
    int32_t* send_ptr = data_send + i*size;
    int32_t* recv_ptr = data_recv + i*size;

    std::fill(send_ptr, send_ptr + size, 1000 + i);
    std::fill(recv_ptr, recv_ptr + size, -1);

    send_bs.push_back(to_bytes_t(send_ptr, size));
    recv_bs.push_back(to_bytes_t(recv_ptr, size));
  }

  // Send and recv a dummy 1 byte object.
  // This is the "barrier"
  auto barrier_tag = 1;
  {
    char msg;
    auto f_send = c.send(
            other_rank,
            barrier_tag,
            to_bytes_t(&msg, 1));
    auto f_recv = c.recv_from(
            other_rank,
            barrier_tag);
    bool success_send = f_send.get();
    auto [success_recv, _0] = f_recv.get();
    if(!success_send || !success_recv) {
      throw std::runtime_error("couldn't barrier");
    }
  }
  std::cout << "BARRIER" << std::endl;

  auto start = std::chrono::high_resolution_clock::now();

  std::vector<std::future<bool>> futs;
  for(int which_channel = 0; which_channel != num_channel; ++which_channel) {
    for(int i = 0; i != num_per_channel; ++i) {
      auto which = which_channel * num_per_channel + i;
      auto tag = 1 + which_channel;
      // send it
      if(use_send_mr) {
        futs.push_back(c.send(other_rank, tag, send_bs[which], mr_send));
      } else {
        futs.push_back(c.send(other_rank, tag, send_bs[which]));
      }

      // recv it
      if(use_recv_mr) {
        futs.push_back(c.recv_from_with_bytes(other_rank, tag, recv_bs[which], mr_recv));
      } else {
        futs.push_back(c.recv_from_with_bytes(other_rank, tag, recv_bs[which]));
      }
    }
  }

  int num_success = 0;
  for(auto& fut: futs) {
    num_success += fut.get();
  }
  auto stop = std::chrono::high_resolution_clock::now();
  size_t total = size*num_success*sizeof(int32_t);
  // in microseconds
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  if(print_results) {
    for(auto& b: recv_bs) {
      int32_t* ptr = (int32_t*)b.data;
      std::cout << "{ " << ptr[0] << std::boolalpha << " " <<
        std::all_of(ptr + 1, ptr + size, [&](int32_t v){
          if(v != ptr[0]) { std::cout << "[" << ptr[0] << "|" << v << "] "; }
          return v == ptr[0]; }) << " }" << std::endl;

    }
    std::cout << std::endl;
  }

  std::cout << num_success << " / " << (2*num) << std::endl;
  double gigabyte_per_second = (1.0e6 * total) / (duration.count() * 1.0e9);
  std::cout << std::endl << gigabyte_per_second << " GB/s" << std::endl;

  if(use_recv_mr) {
    ibv_dereg_mr(mr_recv);
  }
  if(use_send_mr) {
    ibv_dereg_mr(mr_send);
  }

  delete[] data_send;
  delete[] data_recv;
}

//template <typename GenConnection>
//void two_node(int rank, size_t size, size_t num, GenConnection g) {
//  std::vector<bytes_t> bs;
//  for(int i = 0; i != num; ++i) {
//    float* data = new float[size];
//    std::fill(data, data + size, 1000 + i);
//    bs.push_back({data, sizeof(float)*size});
//  }
//
//  connection_t c = g();
//  std::cout << "CONNECTED" << std::endl;
//
//  auto start = std::chrono::high_resolution_clock::now();
//
//  std::vector<std::future<bool>> sends;
//  std::vector<std::future<std::tuple<bool, own_bytes_t>>> recvs;
//  if(rank == 0) {
//    // send tag [100, 100+num)
//    // recv tag [100+num,100+2*num)
//    for(int i = 0; i != num; ++i) {
//      sends.push_back(c.send(1, 100 + i, bs[i]));
//      recvs.push_back(c.recv_from(1, 100 + num + i));
//      sends.back().wait();
//      recvs.back().wait();
//    }
//  } else if(rank == 1) {
//    // send tag [100, 100+num)
//    // recv tag [100+num,100+2*num)
//    for(int i = 0; i != num; ++i) {
//      sends.push_back(c.send(0, 100 +num + i, bs[i]));
//      recvs.push_back(c.recv_from(0, 100 + i));
//      sends.back().wait();
//      recvs.back().wait();
//    }
//  }
//
//  int num_success = 0;
//  for(auto& fut: recvs) {
//    auto [success, r] = fut.get();
//    num_success += success;
//    bs.push_back({
//      .data = r.ptr.release(),
//      .size = r.size
//    });
//  }
//  for(auto& fut: sends) {
//    num_success += fut.get();
//  }
//
//  std::cout << num_success << " / " << (2*num) << std::endl;
//  auto stop = std::chrono::high_resolution_clock::now();
//  // in bytes
//  size_t total = size*num_success*sizeof(float);
//  // in microseconds
//  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
//
//  if(rank < 2) {
//    double gigabyte_per_second = (1.0e6 * total) / (duration.count() * 1.0e9);
//    std::cout << std::endl << gigabyte_per_second << " GB/s" << std::endl;
//  }
//
//  for(bytes_t b: bs) {
//    float* data = (float*)b.data;
//    //if(rank < 2) {
//    //  std::cout << " @ " << data[0];
//    //}
//    delete[] data;
//  }
//  std::cout << std::endl;
//}

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

  //size_t size = 1024*1024*100;
  size_t size = 1024;
  size_t num = 1000;

  //size_t size = 1024;
  //size_t num = 1000;


  auto g = [&](){ return connection_t(argv[2], rank, 0, ips); };
  //three_node(rank, size, num, g);
  //two_node(rank, size, num, g);

  two_node_general(rank, size, 1, num, g, true, true, true);
  //two_node_general(rank, size, num, 1, g, true, true, true);
  //two_node_general(rank, size, 1, num, g, false, false, true);

}
