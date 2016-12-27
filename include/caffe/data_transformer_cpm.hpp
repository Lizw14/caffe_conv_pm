//
// Created by fanliwen on 2016/12/27.
//

#ifndef CAFFE_DATA_TRANSFORMER_CPM_HPP_H
#define CAFFE_DATA_TRANSFORMER_CPM_HPP_H

#include "caffe/data_transformer.hpp"

namespace caffe {

    template <typename Dtype>
    class DataTransformerCPM : public DataTransformer<Dtype> {
    public:
        explicit DataTransformerCPM(const TransformationParameter& param, Phase phase);
        virtual ~DataTransformerCPM() {}

        //image and label
        void Transform_nv(const Datum& datum, Blob<Dtype>* transformed_blob, Blob<Dtype>* transformed_label_blob, int cnt);

        struct AugmentSelection {
            bool flip;
            float degree;
            cv::Size crop;
            float scale;
        };

        struct Joints {
            vector<cv::Point2f> joints;
            vector<int> isVisible;
        };

        struct MetaData {
            string dataset;
            cv::Size img_size;
            bool isValidation;
            int numOtherPeople;
            int people_index;
            int annolist_index;
            int write_number;
            int total_write_number;
            int epoch;
            cv::Point2f objpos; //objpos_x(float), objpos_y (float)
            float scale_self;
            Joints joint_self; //(3*16)

            vector<cv::Point2f> objpos_other; //length is numOtherPeople
            vector<float> scale_other; //length is numOtherPeople
            vector<Joints> joint_others; //length is numOtherPeople
        };

        void generateLabelMap(Dtype*, cv::Mat&, MetaData meta);
        void visualize(cv::Mat& img, MetaData meta, AugmentSelection as);

        bool augmentation_flip(cv::Mat& img, cv::Mat& img_aug, MetaData& meta);
        float augmentation_rotate(cv::Mat& img_src, cv::Mat& img_aug, MetaData& meta);
        float augmentation_scale(cv::Mat& img, cv::Mat& img_temp, MetaData& meta);
        cv::Size augmentation_croppad(cv::Mat& img_temp, cv::Mat& img_aug, MetaData& meta);
        void RotatePoint(cv::Point2f& p, cv::Mat R);
        bool onPlane(cv::Point p, cv::Size img_size);
        void swapLeftRight(Joints& j);
        void SetAugTable(int numData);

        int np_in_lmdb;
        int np;
        bool is_table_set;
        vector<vector<float> > aug_degs;
        vector<vector<int> > aug_flips;

    protected:
        void Transform_nv(const Datum& datum, Dtype* transformed_data, Dtype* transformed_label, int cnt);
        void ReadMetaData(MetaData& meta, const string& data, size_t offset3, size_t offset1);
        void TransformMetaJoints(MetaData& meta);
        void TransformJoints(Joints& joints);
        void clahe(cv::Mat& img, int, int);
        void putGaussianMaps(Dtype* entry, cv::Point2f center, int stride, int grid_x, int grid_y, float sigma);
    };
}

#endif //CAFFE_DATA_TRANSFORMER_CPM_HPP_H
