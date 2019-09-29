#include "cs_to_class_id.h"


CSToClassId::CSToClassId()
{
  count = -1;
}

CSToClassId::CSToClassId(std::string file_name)
{
  load(file_name);
}

CSToClassId::CSToClassId(CSToClassId& other)
{
  copy(other);
}

CSToClassId::CSToClassId(const CSToClassId& other)
{
  copy(other);
}

CSToClassId::~CSToClassId()
{

}

CSToClassId& CSToClassId::operator= (CSToClassId& other)
{
  copy(other);

  return *this;
}

CSToClassId& CSToClassId::operator= (const CSToClassId& other)
{
  copy(other);

  return *this;
}

void CSToClassId::copy(CSToClassId& other)
{
  mapped = other.mapped;
  count  = other.count;
}

void CSToClassId::copy(const CSToClassId& other)
{
  mapped = other.mapped;
  count  = other.count;
}




void CSToClassId::load(std::string file_name)
{
  JsonConfig json(file_name);

  mapped.clear();

  count = -1;

  for (unsigned int i = 0; i < json.result["labels"].size(); i++)
  {
    std::string name = json.result["labels"][i]["string"].asString();
    int id = json.result["labels"][i]["number"].asInt();

    mapped[name] = id;

    if (id > count)
      count = id;

  }
}

int CSToClassId::get(std::string class_name)
{
  if (mapped.find(class_name) == mapped.end())
    return -1;

  return mapped[class_name];
}

int CSToClassId::get_count()
{
  return count;
}
