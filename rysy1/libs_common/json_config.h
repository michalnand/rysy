#ifndef _JSON_CONFIG_H_
#define _JSON_CONFIG_H_

#include <string>

#include <jsoncpp/json/json.h>
#include <jsoncpp/json/value.h>

#include <jsoncpp/json/reader.h>
#include <jsoncpp/json/writer.h>

class JsonConfig
{
  public:
    Json::Value result;

    JsonConfig();
    JsonConfig(std::string json_file_name);
    ~JsonConfig();

    void load(std::string json_file_name);
    void save(std::string file_name);

    Json::Value get_result();
};


#endif
