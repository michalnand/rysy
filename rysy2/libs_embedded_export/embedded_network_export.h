#ifndef _EMBEDDED_NETWORK_EXPORT_H_
#define _EMBEDDED_NETWORK_EXPORT_H_

#include <string>

#include <json_config.h>
#include <log.h>

#include <shape.h>

class EmbeddedNetworkExport
{
  private:
    Json::Value json_parameters;

  public:
    EmbeddedNetworkExport(std::string trained_config_file_name);
    virtual ~EmbeddedNetworkExport();

    int process(std::string export_path, std::string network_prefix);
};


#endif
