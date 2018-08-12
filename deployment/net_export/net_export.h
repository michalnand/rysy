#ifndef _NET_EXPORT_H_
#define _NET_EXPORT_H_

#include <string>

#include <json_config.h>
#include <log.h>

class NetExport
{
  private:
    Json::Value json_parameters;

  public:
    NetExport(std::string trained_config_file_name);
    virtual ~NetExport();

    int process(std::string export_path, std::string network_prefix);
};


#endif
