#include <net_export.h>
#include <iostream>

int main(int argc, char* argv[])
{
  if (argc != 3)
  {
    std::cout << "usage: " << argv[0] << " trained_network_file" << " network_export_dir_name" << "\n";
    return 0;
  }

  std::string trained_network_file  = argv[1];
  std::string network_export_name       = argv[2];

  std::string export_path_dir = network_export_name + "/";

  NetExport net_export(trained_network_file);
  net_export.process(export_path_dir, network_export_name);

  return 0;
}
