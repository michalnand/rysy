#include <iostream>
#include <tensor.h>

#include <cuda_float_allocator.cuh>
#include <cuda_tensor.cuh>


int shape_test()
{
    std::vector<unsigned int> width  = {1, 100, 101, 1024, 1023,   1,   1,    1,    1,  1,   1,    1,    1,   32,  31,  1,   1,    32,  31,  19,  19,  19,  13,  19,  64,   63,  96, 64, 96, 64, 60, 61, 20, 21};
    std::vector<unsigned int> height = {1,   1,   1,    1,    1, 100, 101, 1024, 1023,  1,   1,    1,    1,   32,  31,  1,   1,    32,  31,  19,  19,  17,  17,  13,  64,   63,  96, 64, 96, 64, 50, 51, 50, 51};
    std::vector<unsigned int> depth  = {1,   1,   1,    1,    1,   1,   1,    1,    1,  100, 101, 1024, 1023,  1,   1,  100, 101,  32,  31,  32,  32,  13,  19,  17,  32,   31,  3,  3, 32, 32, 20, 21, 60, 61};

    for (unsigned int i = 0; i < width.size(); i++)
    {
        unsigned int w = width[i];
        unsigned int h = height[i];
        unsigned int d = depth[i];

        //std::cout << "SIZE = " << w << " " << h << " " << d << "\n";

        Shape shape(w, h, d);

        Tensor ta(shape);

        if (ta.size() != w*h*d)
            return -1;

        float const_value = ((rand()%100000)/100000.0 - 0.5)*2.0;
        float eps = 0.00001;

        ta.set_const(const_value);

        for (unsigned int d = 0; d < ta.d(); d++)
        for (unsigned int h = 0; h < ta.h(); h++)
        for (unsigned int w = 0; w < ta.w(); w++)
        {
            float v = ta.get(w, h, d, 0);
            float err = v - const_value;
            if (err < 0.0)
                err = -err;

            if (err > eps)
                return -2;
        }

        std::vector<float> ref(w*h*d);

        for (unsigned int j = 0; j < ref.size(); j++)
            ref[j] = ((rand()%100000)/100000.0 - 0.5)*2.0;

        ta.set_from_host(ref);

        std::vector<float> ta_out(w*h*d);
        std::vector<float> tb_out(w*h*d);
        std::vector<float> tc_out(w*h*d);

        Tensor tb(ta);

        Tensor tc;
        tc = ta;


        ta.set_to_host(ta_out);
        tb.set_to_host(tb_out);
        tc.set_to_host(tc_out);

        for (unsigned int j = 0; j < ref.size(); j++)
        {
            float err;

            err = ref[j] - ta_out[j];
            if (err < 0.0)
                err = -err;
            if (err > eps)
                return -3;


            err = ref[j] - tb_out[j];
            if (err < 0.0)
                err = -err;
            if (err > eps)
                return -4;


            err = ref[j] - tc_out[j];
            if (err < 0.0)
                err = -err;
            if (err > eps)
                return -5;
        }

    }

    return 0;
}

int main()
{
    int test_shape_result = shape_test();
    std::cout << "tensor shape test result " << test_shape_result << "\n";

    /*
    Shape shape(8, 7, 3);
    Tensor t1(shape);

    t1.set_const(1.0);
    t1.set_random(1.0);

    t1.print();
    */
    std::cout << "program done\n";
    return 0;
}
