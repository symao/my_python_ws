#ifndef __KERAS_INFER_H__
#define __KERAS_INFER_H__

#include "keras_model.h"
#include <opencv2/opencv.hpp>

class CNNInfer{
public:
    CNNInfer(const char* model_file, int rows, int cols):m_model(model_file, true){
        m_dc.m_depth = 1;
        m_dc.m_rows = rows;
        m_dc.m_cols = cols;
        m_dc.data = std::vector<std::vector<std::vector<float> > >(1,
            std::vector<std::vector<float> >(rows, std::vector<float>(cols)));
    }

    void compute(const cv::Mat& img, int& class_idx, float& confidence)
    {
        cv::Mat t;
        cv::resize(img, t, cv::Size(m_dc.m_cols, m_dc.m_rows));
        for(int i=0;i<m_dc.m_rows;i++)
        {
            uchar* ptr = t.ptr<uchar>(i);
            for(int j=0;j<m_dc.m_cols;j++)
                m_dc.data[0][i][j] = *ptr++/255.0f;
        }
        std::vector<float> out = m_model.compute_output(&m_dc);
        // for(auto i:out) printf("%f ", i); printf("\n");
        if(out.empty()) class_idx = -1;
        class_idx = 0;
        confidence = out[0];
        for(int i=1;i<out.size();i++)
        {
            if(out[i]>confidence)
            {
                confidence = out[i];
                class_idx = i;
            }
        }
        return;
    }

    int compute(const cv::Mat& img)
    {
        int class_idx;
        float confidence;
        compute(img,class_idx,confidence);
        return class_idx;
    }

private:
    keras::KerasModel m_model;
    keras::DataChunk2D m_dc;
};

#endif//__KERAS_INFER_H__