#include "array_test.h"


#define ARR_TEST_SIZE     ((unsigned int)4000)
#define ARR_TEST_SIZE_B   ((unsigned int)100)


int array_test_value_generator(int x, int y = 0)
{
  return x*3141 + y*271 + 1;
}

int array_test()
{
  //basic constructor test
  Array<int, ARR_TEST_SIZE> arr_a;

  if (arr_a.size() != ARR_TEST_SIZE)
    return -1;

  //operator [] test
  for (unsigned int i = 0; i < arr_a.size(); i++)
    arr_a[i] = array_test_value_generator(i);

  for (unsigned int i = 0; i < arr_a.size(); i++)
    if (arr_a[i] != array_test_value_generator(i))
      return -2;

  //copy constructor test
  Array<int, ARR_TEST_SIZE> arr_b(arr_a);

  if (arr_b.size() != ARR_TEST_SIZE)
    return -3;

  for (unsigned int i = 0; i < arr_b.size(); i++)
    if (arr_b[i] != array_test_value_generator(i))
      return -4;

  //operator = test
  Array<int, ARR_TEST_SIZE> arr_c;
  arr_c = arr_a;

  if (arr_c.size() != ARR_TEST_SIZE)
    return -5;

  for (unsigned int i = 0; i < arr_c.size(); i++)
    if (arr_c[i] != array_test_value_generator(i))
      return -6;

  //operator != test
  if (arr_a != arr_b)
    return -7;

  //operator == test
  if (!(arr_a == arr_b))
    return -8;

  //method set test
  arr_a.set(123456);
  for (unsigned int i = 0; i < arr_a.size(); i++)
    if (arr_a[i] != 123456)
      return -9;

  // matrix init test
  Array<Array<int, ARR_TEST_SIZE_B>, ARR_TEST_SIZE> matrix;
  for (unsigned int i = 0; i < matrix.size(); i++)
    if (matrix[i].size() != ARR_TEST_SIZE_B)
      return -10;

  //operator [] test
  for (unsigned int y = 0; y < matrix.size(); y++)
    for (unsigned int x = 0; x < matrix[y].size(); x++)
      matrix[y][x] = array_test_value_generator(x, y);

  for (unsigned int y = 0; y < matrix.size(); y++)
    for (unsigned int x = 0; x < matrix[y].size(); x++)
      if (matrix[y][x] != array_test_value_generator(x, y))
        return -11;

  //copy constructor test
  Array<Array<int, ARR_TEST_SIZE_B>, ARR_TEST_SIZE> matrix_b(matrix);

  for (unsigned int y = 0; y < matrix.size(); y++)
    for (unsigned int x = 0; x < matrix[y].size(); x++)
      if (matrix[y][x] != matrix_b[y][x])
        return -12;


  //operator = test
  Array<Array<int, ARR_TEST_SIZE_B>, ARR_TEST_SIZE> matrix_c;
  matrix_c = matrix;

  for (unsigned int y = 0; y < matrix.size(); y++)
    for (unsigned int x = 0; x < matrix[y].size(); x++)
      if (matrix[y][x] != matrix_c[y][x])
        return -13;

  return 0;
}





int array_dynamic_test()
{
  //basic constructor test
  ArrayDynamic<int> arr_a(ARR_TEST_SIZE);

  if (arr_a.size() != ARR_TEST_SIZE)
    return -1;

  //operator [] test
  for (unsigned int i = 0; i < arr_a.size(); i++)
    arr_a[i] = array_test_value_generator(i);

  for (unsigned int i = 0; i < arr_a.size(); i++)
    if (arr_a[i] != array_test_value_generator(i))
      return -2;

  //copy constructor test
  ArrayDynamic<int> arr_b(arr_a);

  if (arr_b.size() != ARR_TEST_SIZE)
    return -3;

  for (unsigned int i = 0; i < arr_b.size(); i++)
    if (arr_b[i] != array_test_value_generator(i))
      return -4;

  //operator = and init test
  ArrayDynamic<int> arr_c;
  arr_c.init(arr_a.size());
  arr_c = arr_a;

  if (arr_c.size() != ARR_TEST_SIZE)
    return -5;

  for (unsigned int i = 0; i < arr_c.size(); i++)
    if (arr_c[i] != array_test_value_generator(i))
      return -6;

  //operator != test
  if (arr_a != arr_b)
    return -7;

  //operator == test
  if (!(arr_a == arr_b))
    return -8;

  //method set test
  arr_a.set(123456);
  for (unsigned int i = 0; i < arr_a.size(); i++)
    if (arr_a[i] != 123456)
      return -9;

  return 0;
}
