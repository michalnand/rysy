#ifndef _LOG_H_
#define _LOG_H_

#include <string>
#include <json_config.h>

class Log
{
  protected:
    std::string file_name;

  public:
    Log();
    Log(std::string file_name);

    virtual ~Log();

    void set_output_file_name(std::string file_name);
    void put_string(std::string str);

  public:
    Log& operator <<(std::string &rhs);
    Log& operator <<(char *rhs);
    Log& operator <<(const char *rhs);


    Log& operator <<(char rhs);
    Log& operator <<(unsigned char rhs);

    Log& operator <<(int rhs);
    Log& operator <<(unsigned int rhs);

    Log& operator <<(long int rhs);
    Log& operator <<(unsigned long int rhs);

    Log& operator <<(float rhs);
    Log& operator <<(double rhs);

    Log& operator <<(Json::Value &rhs);
    Log& operator <<(std::vector<float> &rhs);
};


#endif
