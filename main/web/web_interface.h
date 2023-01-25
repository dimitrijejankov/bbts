#include "../../third_party/cpp-httplib/httplib.h"
#include "../server/node_config.h"
#include "gpu_profiler.pb.h"
#include <google/protobuf/util/json_util.h>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <filesystem>

namespace fs = std::filesystem;

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

    svr.Get("/api/logs/all", [&](const httplib::Request& req, httplib::Response& res) {

      std::stringstream ss;
      ss << '[';
      bool has_something = false;
      auto path = std::filesystem::current_path();
      for (const auto &entry : fs::directory_iterator(path / log_dir)) {
        ss  << entry.path().filename() << ","; has_something = true;
      }
      if(has_something) { ss.seekp(-1, std::ios_base::end); }
      ss << ']';
    
      res.set_content(ss.str(), "text/json");
    });

    svr.Get("/api/logs/current", [&](const httplib::Request& req, httplib::Response& res) {
      res.set_content("not implemented", "text/json");
    });

    svr.Get(R"(/api/logs/(.+))", [&](const httplib::Request& req, httplib::Response& res) {
      
      // get the log
      auto selected_log = req.matches[1];
      std::stringstream ss; ss << selected_log;

      // open the file
      auto gpu_log_path = std::filesystem::current_path() / log_dir / ss.str() / "gpu.proto";
      std::fstream input(gpu_log_path, std::ios::in | std::ios::binary);
      
      gpu_profiler_log_t log;
      log.ParseFromIstream(&input);

      std::string json_string;
      google::protobuf::util::JsonPrintOptions options;
      options.add_whitespace = true;
      options.always_print_primitive_fields = true;
      options.preserve_proto_field_names = true;
      MessageToJsonString(log, &json_string, options);
  
      res.set_content(std::move(json_string), "text/json");
    });
  }

  // the http server
  httplib::Server svr;
  
  // the port we listen to for requests
  int32_t port;

  // the root of the web stuff (html, css and js files)
  std::string web_root = "web";

  // the directory where all the logs are stored
  std::string log_dir = "logs";

};

}