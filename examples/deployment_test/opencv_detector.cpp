#include <opencv_detector.h>


OpenCVDetector::OpenCVDetector(std::string json_file_name)
{
    JsonConfig detector_json(json_file_name);

    std::string input_stream = detector_json.result["input_stream"].asString();
    unsigned int width = detector_json.result["width"].asInt();
    unsigned int height = detector_json.result["height"].asInt();

    float confidence = detector_json.result["confidence"].asFloat();

    std::string network = detector_json.result["network"].asString();

    print_results_enabled = detector_json.result["print_results_enabled"].asBool();
    visualisation_enabled = detector_json.result["visualisation_enabled"].asBool();
    saving_enabled = detector_json.result["saving_enabled"].asBool();

    width  = padding(width, 16);
    height = padding(height, 16);

    video_capture = nullptr;
    detector = nullptr;
    video_writer = nullptr;

    video_capture = new cv::VideoCapture(input_stream);

    if(video_capture->isOpened())
    {
        video_capture->set(CV_CAP_PROP_FRAME_WIDTH, width);
        video_capture->set(CV_CAP_PROP_FRAME_HEIGHT,height);

        real_width = int(video_capture->get(CV_CAP_PROP_FRAME_WIDTH));
        real_height = int(video_capture->get(CV_CAP_PROP_FRAME_HEIGHT));

        std::cout << "input size : " << real_width << " " << real_height << "\n";
        std::cout << "confidence : " << confidence << "\n";

        detector = new Detector(network, real_width, real_height, confidence);

        fps_filtered = 0.0;

        if (visualisation_enabled)
            cv::namedWindow("camera", 1);

        if (saving_enabled)
        {
            std::string output_file_name = detector_json.result["output_file_name"].asString();
            video_writer = new cv::VideoWriter(output_file_name,CV_FOURCC('M','J','P','G'),25, cv::Size(real_width,real_height));
        }

    }
    else
    {
        std::cout << "error opening input stream " << input_stream << "\n";
    }
}

OpenCVDetector::~OpenCVDetector()
{
    if (detector != nullptr)
        delete detector;

    if (video_capture != nullptr)
        delete video_capture;

    if (video_writer != nullptr)
        delete video_writer;
}


sDetectorResult& OpenCVDetector::get_result()
{
    return detector->get_result();
}


unsigned int OpenCVDetector::padding(unsigned int value, unsigned int padding)
{
	unsigned int result;
	result = (value/padding)*padding;
	return result;
}

float OpenCVDetector::round_to_two(float var)
{
    float value = (int)(var * 100 + .5);
    return (float)value / 100;
}




int OpenCVDetector::process_frame()
{
    cv::Mat frame;
    *video_capture >> frame;

    if (frame.empty())
        return -1;

    timer.start();
    detector->process(frame);
    timer.stop();

    float fps = 1.0/(0.001*timer.get_duration() + 0.000000001);
	fps_filtered = 0.95*fps_filtered + 0.05*fps;

    if (visualisation_enabled)
    {
    	detector->inpaint_class_result(frame, 0.5);

    	std::string str_fps_a = "resolution = [" + std::to_string(real_width) + " " + std::to_string(real_height) + "]";
    	std::string str_fps_b = "fps = " + std::to_string((int)fps_filtered);
    	cv::putText(frame, str_fps_a, cv::Point(30, 30), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(255, 0, 0), 2);
    	cv::putText(frame, str_fps_b, cv::Point(30, 60), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(255, 0, 0), 2);


		cv::imshow("camera", frame);
	}

    if (saving_enabled)
    {
        video_writer->write(frame);
    }

    if (print_results_enabled)
    {
        std::cout << get_result().json_string << "\n\n";
    }

    if( cv::waitKey(10) == 27 )
        return 1;

    return 0;
}
