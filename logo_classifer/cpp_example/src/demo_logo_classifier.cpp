#include "keras_infer.h"

int main()
{
    CNNInfer cnn_infer("../../saved_models/logo_classifier.nnet", 28, 28);
    cv::Mat img = cv::imread("../../data/pos_raw/a.jpg", cv::IMREAD_GRAYSCALE);
    int class_idx;
    float confidence;
    cnn_infer.compute(img,class_idx,confidence);
    printf("The class is: %d, confidence:%f\n", class_idx, confidence);
}
