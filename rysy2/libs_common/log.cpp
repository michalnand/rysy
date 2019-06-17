#include "log.h"

#include <sstream>
#include <iomanip>
#include <iostream>
#include <fstream>

Log::Log()
{
  file_name = "std_out_file_name";
}

Log::Log(std::string file_name)
{
  set_output_file_name(file_name);
}

Log::~Log()
{ 

}

void Log::set_output_file_name(std::string file_name)
{
  this->file_name = file_name;

  std::ofstream result_file(this->file_name);
  result_file.close();
}


void Log::put_string(std::string str)
{
  if (file_name == "std_out_file_name")
  {
    std::cout << str;
  }
  else
  {
    std::ofstream result_file(file_name, std::ofstream::out | std::ofstream::app);
    result_file << str;
    result_file.close();
  }
}

Log& Log::operator <<(std::string &rhs)
{
  put_string(rhs);

  return *this;
}

Log& Log::operator <<(char *rhs)
{
  std::string tmp(rhs);

  put_string(tmp);

  return *this;
}

Log& Log::operator <<(const char *rhs)
{
  std::string tmp(rhs);

  put_string(tmp);

  return *this;
}


Log& Log::operator <<(char rhs)
{
  std::string tmp;
  tmp = std::to_string(rhs);
  put_string(tmp);

  return *this;
}


Log& Log::operator <<(unsigned char rhs)
{
  std::string tmp;
  tmp = std::to_string(rhs);
  put_string(tmp);

  return *this;
}


Log& Log::operator <<(int rhs)
{
  std::string tmp;
  tmp = std::to_string(rhs);
  put_string(tmp);

  return *this;
}

Log& Log::operator <<(unsigned int rhs)
{
  std::string tmp;
  tmp = std::to_string(rhs);
  put_string(tmp);

  return *this;
}

Log& Log::operator <<(long int rhs)
{
  std::string tmp;
  tmp = std::to_string(rhs);
  put_string(tmp);

  return *this;
}

Log& Log::operator <<(unsigned long int rhs)
{
  std::string tmp;
  tmp = std::to_string(rhs);
  put_string(tmp);

  return *this;
}


Log& Log::operator <<(float rhs)
{
  std::string tmp;
  tmp = std::to_string(rhs);
  put_string(tmp);

  return *this;
}

Log& Log::operator <<(double rhs)
{
  std::string tmp;
  tmp = std::to_string(rhs);
  put_string(tmp);

  return *this;
}

Log& Log::operator <<(Json::Value &rhs)
{
  Json::FastWriter fastWriter;

  std::string output = fastWriter.write(rhs);

  put_string(output);

  return *this;
}


Log& Log::operator <<(std::vector<float> &rhs)
{
  std::string tmp;

  for (unsigned int i = 0; i < rhs.size(); i++)
    tmp+= std::to_string(rhs[i]) + "\t";

  put_string(tmp);

  return *this;
}
