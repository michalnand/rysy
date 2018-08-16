#ifndef _CS_TO_CLASS_ID_H_
#define _CS_TO_CLASS_ID_H_

#include <map>
#include <json_config.h>

class CSToClassId
{
  private:
    int count;
    std::map<std::string, int> mapped;

  public:
    // Default constructor
    CSToClassId();
    CSToClassId(std::string file_name);


    // Copy constructor
    CSToClassId(CSToClassId& other);

    // Copy constructor
    CSToClassId(const CSToClassId& other);

    // Destructor
    virtual ~CSToClassId();

    // Copy assignment operator
    CSToClassId& operator= (CSToClassId& other);

    // Copy assignment operator
    CSToClassId& operator= (const CSToClassId& other);


  protected:
    void copy(CSToClassId& other);
    void copy(const CSToClassId& other);

  public:
    void load(std::string file_name);

    int get(std::string class_name);
    int get_count();
};

#endif
