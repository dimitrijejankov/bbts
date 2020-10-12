#include "../src/operations/reduce_op.h"

int main(int argc, char **argv) {

  // we set the root node to three as we want the results to got there
  const bbts::node_id_t root_node = 2;

  // make the configuration
  auto config = std::make_shared<bbts::node_config_t>(bbts::node_config_t{.argc=argc, .argv = argv});

  // create the tensor factory
  bbts::tensor_factory_ptr_t factory = std::make_shared<bbts::tensor_factory_t>();

  // init the communicator with the configuration
  bbts::communicator_t comm(config);

  // check the number of nodes
  if(root_node >= comm.get_num_nodes()) {
    std::cerr << "Can not run add more nodes or change the root node\n";
    return -1;
  }

  // create the storage
  bbts::storage_t storage;

  // get the id
  auto id = factory->get_tensor_ftm("dense");

  // crate the udf manager
  bbts::udf_manager manager(factory);

  // return me that matcher for matrix addition
  auto matcher = manager.get_matcher_for("matrix_add");

  // get the ud object
  auto ud = matcher->findMatch({"dense", "dense"}, {"dense"}, false, 0);

  // make the meta
  bbts::dense_tensor_meta_t dm{id, 100, 200};
  auto &m = dm.as<bbts::tensor_meta_t>();

  // get how much we need to allocate
  auto size = factory->get_tensor_size(m);
  std::unique_ptr<char[]> a_mem(new char[size]);

  // initi the tensor
  auto &a = factory->init_tensor((bbts::tensor_t*) a_mem.get(), m).as<bbts::dense_tensor_t>();

  // get a reference to the metadata
  auto &am = a.meta().m();

  // init the matrix
  for(int i = 0; i < am.num_rows * am.num_cols; ++i) {
    a.data()[i] = 1 + comm.get_rank() + i;
  }

  // we are only involving the even ranks in teh computation
  bool success = true;
  if(comm.get_rank() % 2 == 0) {

    // pick all the nodes with an even rank, for testing purpouses 
    std::vector<bbts::node_id_t> nodes;
    for(bbts::node_id_t i = 0; i < comm.get_num_nodes(); i += 2) {
      nodes.push_back(i);
    }
    std::swap(nodes[0], *std::find(nodes.begin(), nodes.end(), root_node));

    // craete the reduce
    auto reduce_op = bbts::reduce_op(comm, *factory, storage, nodes, 111, a, *ud);
    auto b = reduce_op.apply();

    if(comm.get_rank() == root_node) {
    
      auto &bb = b->as<bbts::dense_tensor_t>();
      for(int i = 0; i < am.num_rows * am.num_cols; ++i) {
          
          int32_t val = 0;
          for(int rank = 0; rank < comm.get_num_nodes(); rank += 2) {
            val += 1 + rank + i;
          }
          if(bb.data()[i] != val) {
            success = false;
            std::cout << "not ok " << val << " " << bb.data()[i] << '\n';
          }
      }
    }
  }

  // wait for all
  comm.barrier();

  // if works
  if(success) {
    std::cerr << "all good\n";
  }

  return !success;
}