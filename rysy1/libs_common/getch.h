#ifndef _GETCH_H_
#define _GETCH_H_


class Getchar
{
  public:
    Getchar();
    ~Getchar();

    int get();

    operator int()
    {
      return get();
    }
    
  private:
    int kbhit();
};


#endif
