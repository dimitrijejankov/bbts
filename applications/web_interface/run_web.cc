#include "../../main/web/web_interface.h"

int main() {

  auto config = std::make_shared<bbts::node_config_t>(0, nullptr);
  bbts::web_interface web(config);

  web.run();
}