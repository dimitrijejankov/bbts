#include "gpu_scheduler_impl.h"
#include <cstddef>
#include <thread>

#ifdef ENABLE_GPU

namespace bbts {

gpu_scheduler_impl_t::gpu_scheduler_impl_t(
    const bbts::tensor_factory_ptr_t &fact)
    : _factory(fact) {

  // get the number of devices
  // cuDeviceGetCount(&num_devices);
  num_devices = 4;

  devices.resize(num_devices);
  for (auto dev = 0; dev < num_devices; ++dev) {

    // set the device
    cudaSetDevice(dev);

    // get the buffers
    auto &FRONT = devices[dev].FRONT;
    auto &MID = devices[dev].MID;
    auto &BACK = devices[dev].BACK;

    // create the front stream (at the current step fetch the tensor)
    cudaStreamCreate(&devices[dev]._streams[FRONT]);
    cudaEventCreate(&devices[dev]._events[FRONT]);
    cublasCreate(&devices[dev]._handles[FRONT]);
    cublasSetStream(devices[dev]._handles[FRONT], devices[dev]._streams[FRONT]);

    // create the mid stream (at the current step do the kernel)
    cudaStreamCreate(&devices[dev]._streams[MID]);
    cudaEventCreate(&devices[dev]._events[MID]);
    cublasCreate(&devices[dev]._handles[MID]);
    cublasSetStream(devices[dev]._handles[MID], devices[dev]._streams[MID]);

    // create the back stream (at the current step does unloading)
    cudaStreamCreate(&devices[dev]._streams[BACK]);
    cudaEventCreate(&devices[dev]._events[BACK]);
    cublasCreate(&devices[dev]._handles[BACK]);
    cublasSetStream(devices[dev]._handles[BACK], devices[dev]._streams[BACK]);
  }
}

gpu_scheduler_impl_t::~gpu_scheduler_impl_t() {

  // destroy the handle
  for (auto dev = 0; dev < num_devices; ++dev) {

    auto &FRONT = devices[dev].FRONT;
    auto &MID = devices[dev].MID;
    auto &BACK = devices[dev].BACK;

    cublasDestroy(devices[dev]._handles[FRONT]);
    cublasDestroy(devices[dev]._handles[MID]);
    cublasDestroy(devices[dev]._handles[BACK]);
  }
}

void gpu_scheduler_impl_t::run() {
  std::vector<std::thread> threads;
  for (auto dev = 0; dev < num_devices; ++dev) {
    threads.push_back(std::thread([&, dev]() { _run(dev); }));
  }
  for (auto &t : threads) {
    t.join();
  }
}

std::future<bool> gpu_scheduler_impl_t::execute_kernel(
    bbts::ud_impl_t *fun, const bbts::ud_impl_t::tensor_params_t *params,
    const bbts::ud_impl_t::tensor_args_t *inputs,
    bbts::ud_impl_t::tensor_args_t *outputs) {

  std::future<bool> wait;
  {
    std::promise<bool> success;
    wait = success.get_future();

    // lock this thing
    std::unique_lock<std::mutex> lk(_m);

    // schedule
    _q.push(kernel_spec_t{.fun = fun,
                          .params = *params,
                          .inputs = inputs,
                          .outputs = outputs,
                          .success = std::move(success)});
    _cv.notify_one();
  }

  return wait;
}

void gpu_scheduler_impl_t::shutdown() {

  std::unique_lock<std::mutex> lk(_m);

  // shutdown and notify
  _shutdown = true;
  _cv.notify_one();
}

void gpu_scheduler_impl_t::_run(int device) {

  // set the device
  cudaSetDevice(device);
  std::cout << "Running on " << device << '\n';

  // get all the stuff we need
  auto &events = devices[device]._events;
  auto &has_something = devices[device]._has_something;
  auto &streams = devices[device]._streams;
  auto &specs = devices[device]._specs;
  auto &handles = devices[device]._handles;

  auto &FRONT = devices[device].FRONT;
  auto &MID = devices[device].MID;
  auto &BACK = devices[device].BACK;

  while (true) {

    // sync the phases
    cudaEventSynchronize(events[FRONT]);
    cudaEventSynchronize(events[MID]);
    cudaEventSynchronize(events[BACK]);

    // wait until we have something here
    std::unique_lock<std::mutex> lk(_m);
    _cv.wait(lk,
             [&] { return _left_to_process != 0 || !_q.empty() || _shutdown; });

    // are we done?
    if (_shutdown && _left_to_process == 0) {
      break;
    }

    // run the kernel first, since this is more imporant
    if (has_something[MID]) {

      // set the stream and cublas handle
      specs[MID].params.stream = streams[MID];
      specs[MID].params.cublas_handle = handles[MID];

      // call the kernel
      specs[MID].fun->call_gpu_ud(specs[MID].params, *specs[MID].inputs,
                                  *specs[MID].outputs);
      cudaEventRecord(events[MID], streams[MID]);
    }

    // do we have something to put into the pipeline
    if (!_q.empty()) {

      // get it from the queue
      specs[FRONT] = std::move(_q.front());
      _q.pop();

      // prefetch the input tensors
      for (std::size_t i = 0; i < specs[FRONT].inputs->num_args(); ++i) {

        // get the tensor and the number of bytes
        auto ts = specs[FRONT].inputs->get_by_idx(i).get_data_ptr<void>();
        auto num_bytes = _factory->get_tensor_size(
                             specs[FRONT].inputs->get_by_idx(i).get_meta<tensor_meta_t>()) -
                         sizeof(bbts::tensor_meta_t);
        cudaMemPrefetchAsync(ts, num_bytes, 0, streams[FRONT]);
      }
      cudaEventRecord(events[FRONT], streams[FRONT]);
      has_something[FRONT] = true;

      // we have one more to process
      _left_to_process++;
    }

    // do the unloading if necessary
    if (has_something[BACK]) {

      // go through all the outputs and unload
      for (std::size_t i = 0; i < specs[BACK].outputs->num_args(); ++i) {

        // get the tensor and the number of bytes
        auto ts = specs[BACK].outputs->get_by_idx(i).get_data_ptr<void>();
        auto num_bytes = _factory->get_tensor_size(
                         specs[BACK].outputs->get_by_idx(i).get_meta<tensor_meta_t>()) -
                         sizeof(bbts::tensor_meta_t);

        // load out on back
        cudaMemPrefetchAsync(&ts, num_bytes, cudaCpuDeviceId, streams[BACK]);
      }
      cudaEventRecord(events[BACK], streams[BACK]);
      has_something[BACK] = false;

      // we finished
      specs[BACK].success.set_value(true);

      // we processed one
      _left_to_process--;
    }

    // rotate the streams and events
    _rotate(FRONT, MID, BACK);
  }
}

void gpu_scheduler_impl_t::_rotate(size_t &FRONT, size_t &MID, size_t &BACK) {

  auto tmp = BACK;
  BACK = MID;
  MID = FRONT;
  FRONT = tmp;
}

} // namespace bbts

#endif