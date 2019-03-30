#ifndef _OPENCV_DETECTOR_H_
#define _OPENCV_DETECTOR_H_

#include <iostream>

#include <detector.h>
#include "opencv2/opencv.hpp"

#include <timer.h>


class OpenCVDetector
{
    public:
        OpenCVDetector(std::string json_file_name);
        virtual ~OpenCVDetector();

        int process_frame();
        sDetectorResult& get_result();

    private:
        unsigned int padding(unsigned int value, unsigned int padding);
        float round_to_two(float var);

    private:
        Timer timer;

        cv::VideoCapture *video_capture;
        Detector         *detector;
        cv::VideoWriter  *video_writer;

        bool print_results_enabled, visualisation_enabled, saving_enabled;

        float fps_filtered;
        unsigned int real_width, real_height;
};

#endif
