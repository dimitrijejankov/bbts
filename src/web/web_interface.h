#include "../../third_party/cpp-httplib/httplib.h"

#include "../server/node_config.h"
#include <cstdint>
#include <stdexcept>

namespace bbts {

class web_interface {
public:

  web_interface(const bbts::node_config_ptr_t config) {
    
    // set the directories and the port
    web_root = config->web_root;
    log_dir = config->log_dir;
    port = config->web_port;

    // setup the root
    auto ret = svr.set_mount_point("/", web_root);
    if (!ret) { throw std::runtime_error("Could not find the web root."); }

    // init the API
    init_api();
  }

  void run() {

    // kick off the server
    svr.listen("0.0.0.0", port);
  }

  void shutdown() { svr.stop(); }

private:

  void init_api() {
    
  }

  // the http server
  httplib::Server svr;
  
  // the port we listen to for requests
  int32_t port;

  // the root of the web stuff (html, css and js files)
  std::string web_root = "./web";

  // the directory where all the logs are stored
  std::string log_dir = "./log";

};

}