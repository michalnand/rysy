#ifndef _RL_STATE_H_
#define _RL_STATE_H_

#include <vector>
#include <string>

class RLState
{
  protected:
    unsigned int m_w, m_h, m_d, m_size;
    std::vector<float> m_state;

  public:
    RLState();
    RLState(RLState& other);
    RLState(const RLState& other);

    RLState(unsigned int w, unsigned int h = 1, unsigned int d = 1);


    virtual ~RLState();
    RLState& operator= (RLState& other);
    RLState& operator= (const RLState& other);

    void init(unsigned int w, unsigned h = 1, unsigned d = 1);

  protected:
    void copy(RLState& other);
    void copy(const RLState& other);

  public:
    std::vector<float>& get();
    void set(std::vector<float> &state);
    void set_element(float value, unsigned int w, unsigned int h = 0, unsigned int d = 0);
    void clear();

  public:

    virtual std::string as_string(unsigned int precision = 3);
    void print(unsigned int precision = 3);

    bool is_valid();

  public:
    void random(float range = 1.0);
    void add_noise(float level);

    unsigned int argmax();
    unsigned int argmin();

    float dot(RLState& other);
    float distance(RLState& other);

    void normalise();

  public:
    unsigned int w()
    {
      return m_w;
    }

    unsigned int h()
    {
      return m_h;
    }

    unsigned int d()
    {
      return m_d;
    }

    unsigned int size()
    {
      return m_size;
    }

  private:
    float randf();



};

#endif
