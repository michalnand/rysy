#include <net_export.h>
#include <iostream>

int main(int argc, char* argv[])
{
  if (argc != 4)
  {
    std::cout << "usage: " << argv[0] << " trained_network_file" << " export_path_dir" << " export_network_name/\n";
    return 0;
  }

  std::string trained_network_file  = argv[1];
  std::string export_path_dir       = argv[2];
  std::string export_network_name   = argv[3];

  NetExport net_export(trained_network_file);
  net_export.process(export_path_dir, export_network_name);

  return 0;
}
