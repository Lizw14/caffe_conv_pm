//
// Created by fanliwen on 2016/12/19.
//

#ifndef CAFFE_CPM_DATA_LAYER_H
#define CAFFE_CPM_DATA_LAYER_H

#include <vector>
#include "caffe/data_reader.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/data_transformer_cpm.hpp"

namespace caffe {

    template <typename Dtype>
    class CPMDataLayer : public BasePrefetchingDataLayer<Dtype> {
    public:
        explicit CPMDataLayer(const LayerParameter& param);
        virtual ~CPMDataLayer();
        virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
        virtual inline bool ShareInParallel() const { return false; }
        virtual inline const char* type() const { return "CPMData"; }
        virtual inline int ExactNumBottomBlobs() const { return 0; }
        virtual inline int MinTopBlobs() const { return 1; }
        virtual inline int MaxTopBlobs() const { return 2; }

    protected:
        virtual void load_batch(Batch<Dtype>* batch);

        DataReader reader_;
        Blob<Dtype> transformed_label_; // add another blob
    };

}
#endif //CAFFE_CPM_DATA_LAYER_H
