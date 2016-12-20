#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/highgui/highgui.hpp>
#endif  // USE_OPENCV

#include <iostream>
#include <algorithm>
#include <fstream>
using namespace std;

#include <string>
#include <sstream>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template<typename Dtype>
void DecodeFloats(const string& data, size_t idx, Dtype* pf, size_t len) {
  memcpy(pf, const_cast<char*>(&data[idx]), len * sizeof(Dtype));
}

string DecodeString(const string& data, size_t idx) {
  string result = "";
  int i = 0;
  while(data[idx+i] != 0){
    result.push_back(char(data[idx+i]));
    i++;
  }
  return result;
}

template<typename Dtype>
void DataTransformer<Dtype>::ReadMetaData(MetaData& meta, const string& data, size_t offset3, size_t offset1) { //very specific to genLMDB.py
  // ------------------- Dataset name ----------------------
  meta.dataset = DecodeString(data, offset3);
  // ------------------- Image Dimension -------------------
  float height, width;
  DecodeFloats(data, offset3+offset1, &height, 1);
  DecodeFloats(data, offset3+offset1+4, &width, 1);
  meta.img_size = cv::Size(width, height);
  // ----------- Validation, nop, counters -----------------
  meta.isValidation = data[offset3 + 2 * offset1] != 0;
  meta.numOtherPeople = (int)data[offset3+2*offset1+1];
  meta.people_index = (int)data[offset3+2*offset1+2];
  float annolist_index;
  DecodeFloats(data, offset3+2*offset1+3, &annolist_index, 1);
  meta.annolist_index = (int)annolist_index;
  float write_number;
  DecodeFloats(data, offset3+2*offset1+7, &write_number, 1);
  meta.write_number = (int)write_number;
  float total_write_number;
  DecodeFloats(data, offset3+2*offset1+11, &total_write_number, 1);
  meta.total_write_number = (int)total_write_number;

  // count epochs according to counters
  static int cur_epoch = -1;
  if(meta.write_number == 0){
    cur_epoch++;
  }
  meta.epoch = cur_epoch;
  if(meta.write_number % 1000 == 0){
    LOG(INFO) << "dataset: " << meta.dataset <<"; img_size: " << meta.img_size
        << "; meta.annolist_index: " << meta.annolist_index << "; meta.write_number: " << meta.write_number
        << "; meta.total_write_number: " << meta.total_write_number << "; meta.epoch: " << meta.epoch;
  }
  if(param_.aug_way() == "table" && !is_table_set){
    SetAugTable(meta.total_write_number);
    is_table_set = true;
  }

  // ------------------- objpos -----------------------
  DecodeFloats(data, offset3+3*offset1, &meta.objpos.x, 1);
  DecodeFloats(data, offset3+3*offset1+4, &meta.objpos.y, 1);
  meta.objpos -= cv::Point2f(1,1);
  // ------------ scale_self, joint_self --------------
  DecodeFloats(data, offset3+4*offset1, &meta.scale_self, 1);
  meta.joint_self.joints.resize(np_in_lmdb);
  meta.joint_self.isVisible.resize(np_in_lmdb);
  for(int i=0; i<np_in_lmdb; i++){
    DecodeFloats(data, offset3+5*offset1+4*i, &meta.joint_self.joints[i].x, 1);
    DecodeFloats(data, offset3+6*offset1+4*i, &meta.joint_self.joints[i].y, 1);
    meta.joint_self.joints[i] -= cv::Point2f(1,1); //from matlab 1-index to c++ 0-index
    float isVisible;
    DecodeFloats(data, offset3+7*offset1+4*i, &isVisible, 1);
    meta.joint_self.isVisible[i] = (isVisible == 0) ? 0 : 1;
    if(meta.joint_self.joints[i].x < 0 || meta.joint_self.joints[i].y < 0 ||
       meta.joint_self.joints[i].x >= meta.img_size.width || meta.joint_self.joints[i].y >= meta.img_size.height){
      meta.joint_self.isVisible[i] = 2; // 2 means cropped, 0 means occluded by still on image
    }
  }
  
  //others (7 lines loaded)
  meta.objpos_other.resize(meta.numOtherPeople);
  meta.scale_other.resize(meta.numOtherPeople);
  meta.joint_others.resize(meta.numOtherPeople);
  for(int p=0; p<meta.numOtherPeople; p++){
    DecodeFloats(data, offset3+(8+p)*offset1, &meta.objpos_other[p].x, 1);
    DecodeFloats(data, offset3+(8+p)*offset1+4, &meta.objpos_other[p].y, 1);
    meta.objpos_other[p] -= cv::Point2f(1,1);
    DecodeFloats(data, offset3+(8+meta.numOtherPeople)*offset1+4*p, &meta.scale_other[p], 1);
  }
  //8 + numOtherPeople lines loaded
  for(int p=0; p<meta.numOtherPeople; p++){
    meta.joint_others[p].joints.resize(np_in_lmdb);
    meta.joint_others[p].isVisible.resize(np_in_lmdb);
    for(int i=0; i<np_in_lmdb; i++){
      DecodeFloats(data, offset3+(9+meta.numOtherPeople+3*p)*offset1+4*i, &meta.joint_others[p].joints[i].x, 1);
      DecodeFloats(data, offset3+(9+meta.numOtherPeople+3*p+1)*offset1+4*i, &meta.joint_others[p].joints[i].y, 1);
      meta.joint_others[p].joints[i] -= cv::Point2f(1,1);
      float isVisible;
      DecodeFloats(data, offset3+(9+meta.numOtherPeople+3*p+2)*offset1+4*i, &isVisible, 1);
      meta.joint_others[p].isVisible[i] = (isVisible == 0) ? 0 : 1;
      if(meta.joint_others[p].joints[i].x < 0 || meta.joint_others[p].joints[i].y < 0 ||
         meta.joint_others[p].joints[i].x >= meta.img_size.width || meta.joint_others[p].joints[i].y >= meta.img_size.height){
        meta.joint_others[p].isVisible[i] = 2; // 2 means cropped, 1 means occluded by still on image
      }
    }
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::SetAugTable(int numData){
  aug_degs.resize(numData);     
  aug_flips.resize(numData);  
  for(int i = 0; i < numData; i++){
    aug_degs[i].resize(param_.num_total_augs());
    aug_flips[i].resize(param_.num_total_augs());
  }
  //load table files
  char filename[100];
  sprintf(filename, "../../rotate_%d_%d.txt", param_.num_total_augs(), numData);
  ifstream rot_file(filename);
  char filename2[100];
  sprintf(filename2, "../../flip_%d_%d.txt", param_.num_total_augs(), numData);
  ifstream flip_file(filename2);

  for(int i = 0; i < numData; i++){
    for(int j = 0; j < param_.num_total_augs(); j++){
      rot_file >> aug_degs[i][j];
      flip_file >> aug_flips[i][j];
    }
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::TransformMetaJoints(MetaData& meta) {
  //transform joints in meta from np_in_lmdb (specified in prototxt) to np (specified in prototxt)
  TransformJoints(meta.joint_self);
  for(int i=0;i<meta.joint_others.size();i++){
    TransformJoints(meta.joint_others[i]);
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::TransformJoints(Joints& j) {
  Joints jo = j;
  if(np == 14){
    int MPI_to_ours[14] = {9, 8, 12, 11, 10, 13, 14, 15, 2, 1, 0, 3, 4, 5};
    jo.joints.resize(np);
    jo.isVisible.resize(np);
    for(int i=0;i<np;i++){
      jo.joints[i] = j.joints[MPI_to_ours[i]];
      jo.isVisible[i] = j.isVisible[MPI_to_ours[i]];
    }
  }
  else if(np == 28){
    int MPI_to_ours_1[28] = {9, 8,12,11,10,13,14,15, 2, 1, 0, 3, 4, 5, 7, 6, \
                             9, 8,12,11, 8,13,14, 2, 1, 3, 4, 6};
                          //17,18,19,20,21,22,23,24,25,26,27,28
    int MPI_to_ours_2[28] = {9, 8,12,11,10,13,14,15, 2, 1, 0, 3, 4, 5, 7, 6, \
                             8,12,11,10,13,14,15, 1, 0, 4, 5, 7};
                          //17,18,19,20,21,22,23,24,25,26,27,28
    jo.joints.resize(np);
    jo.isVisible.resize(np);
    for(int i=0;i<np;i++){
      jo.joints[i] = (j.joints[MPI_to_ours_1[i]] + j.joints[MPI_to_ours_2[i]]) * 0.5;
      if(j.isVisible[MPI_to_ours_1[i]]==2 || j.isVisible[MPI_to_ours_2[i]]==2){
        jo.isVisible[i] = 2;
      }
      else {
        jo.isVisible[i] = j.isVisible[MPI_to_ours_1[i]] && j.isVisible[MPI_to_ours_2[i]];
      }
    }
  }
  j = jo;
}

template<typename Dtype>
DataTransformer<Dtype>::DataTransformer(const TransformationParameter& param,
    Phase phase)
    : param_(param), phase_(phase) {
  // check if we want to use mean_file
  if (param_.has_mean_file()) {
    CHECK_EQ(param_.mean_value_size(), 0) <<
      "Cannot specify mean_file and mean_value at the same time";
    const string& mean_file = param.mean_file();
    if (Caffe::root_solver()) {
      LOG(INFO) << "Loading mean file from: " << mean_file;
    }
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
    data_mean_.FromProto(blob_proto);
  }
  // check if we want to use mean_value
  if (param_.mean_value_size() > 0) {
    CHECK(param_.has_mean_file() == false) <<
      "Cannot specify mean_file and mean_value at the same time";
    for (int c = 0; c < param_.mean_value_size(); ++c) {
      mean_values_.push_back(param_.mean_value(c));
    }
  }

  np_in_lmdb = param_.np_in_lmdb();
  np = param_.num_parts();
  is_table_set = false;
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const Datum& datum,
                                       Dtype* transformed_data) {
  const string& data = datum.data();
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();

  const int crop_size = param_.crop_size();
  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_uint8 = data.size() > 0;
  const bool has_mean_values = mean_values_.size() > 0;

  CHECK_GT(datum_channels, 0);
  CHECK_GE(datum_height, crop_size);
  CHECK_GE(datum_width, crop_size);

  Dtype* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(datum_channels, data_mean_.channels());
    CHECK_EQ(datum_height, data_mean_.height());
    CHECK_EQ(datum_width, data_mean_.width());
    mean = data_mean_.mutable_cpu_data();
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == datum_channels) <<
     "Specify either 1 mean_value or as many as channels: " << datum_channels;
    if (datum_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < datum_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }

  int height = datum_height;
  int width = datum_width;

  int h_off = 0;
  int w_off = 0;
  if (crop_size) {
    height = crop_size;
    width = crop_size;
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = Rand(datum_height - crop_size + 1);
      w_off = Rand(datum_width - crop_size + 1);
    } else {
      h_off = (datum_height - crop_size) / 2;
      w_off = (datum_width - crop_size) / 2;
    }
  }

  Dtype datum_element;
  int top_index, data_index;
  for (int c = 0; c < datum_channels; ++c) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        data_index = (c * datum_height + h_off + h) * datum_width + w_off + w;
        if (do_mirror) {
          top_index = (c * height + h) * width + (width - 1 - w);
        } else {
          top_index = (c * height + h) * width + w;
        }
        if (has_uint8) {
          datum_element =
            static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
        } else {
          datum_element = datum.float_data(data_index);
        }
        if (has_mean_file) {
          transformed_data[top_index] =
            (datum_element - mean[data_index]) * scale;
        } else {
          if (has_mean_values) {
            transformed_data[top_index] =
              (datum_element - mean_values_[c]) * scale;
          } else {
            transformed_data[top_index] = datum_element * scale;
          }
        }
      }
    }
  }
}


template<typename Dtype>
void DataTransformer<Dtype>::Transform(const Datum& datum,
                                       Blob<Dtype>* transformed_blob) {
  // If datum is encoded, decoded and transform the cv::image.
  if (datum.encoded()) {
#ifdef USE_OPENCV
    CHECK(!(param_.force_color() && param_.force_gray()))
        << "cannot set both force_color and force_gray";
    cv::Mat cv_img;
    if (param_.force_color() || param_.force_gray()) {
    // If force_color then decode in color otherwise decode in gray.
      cv_img = DecodeDatumToCVMat(datum, param_.force_color());
    } else {
      cv_img = DecodeDatumToCVMatNative(datum);
    }
    // Transform the cv::image into blob.
    return Transform(cv_img, transformed_blob);
#else
    LOG(FATAL) << "Encoded datum requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  } else {
    if (param_.force_color() || param_.force_gray()) {
      LOG(ERROR) << "force_color and force_gray only for encoded datum";
    }
  }

  const int crop_size = param_.crop_size();
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();

  // Check dimensions.
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int num = transformed_blob->num();

  CHECK_EQ(channels, datum_channels);
  CHECK_LE(height, datum_height);
  CHECK_LE(width, datum_width);
  CHECK_GE(num, 1);

  if (crop_size) {
    CHECK_EQ(crop_size, height);
    CHECK_EQ(crop_size, width);
  } else {
    CHECK_EQ(datum_height, height);
    CHECK_EQ(datum_width, width);
  }

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();
  Transform(datum, transformed_data);
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform_nv(const Datum& datum,
                                          Blob<Dtype>* transformed_data,
                                          Blob<Dtype>* transformed_label,
                                          int cnt) {
  const int datum_channels = datum.channels();

  const int im_channels = transformed_data->channels();
  const int im_num = transformed_data->num();

  const int lb_num = transformed_label->num();

  CHECK_EQ(datum_channels, 4);
  CHECK_EQ(im_channels, 4);
  CHECK_EQ(im_num, lb_num);
  CHECK_GE(im_num, 1);

  Dtype* transformed_data_pointer = transformed_data->mutable_cpu_data();
  Dtype* transformed_label_pointer = transformed_label->mutable_cpu_data();

  this->Transform_nv(datum, transformed_data_pointer, transformed_label_pointer, cnt); //call function 1
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform_nv(const Datum& datum, Dtype* transformed_data, Dtype* transformed_label, int cnt) {
  
  //TODO: some parameter should be set in prototxt
  int clahe_tileSize = param_.clahe_tile_size();
  int clahe_clipLimit = param_.clahe_clip_limit();
  //float targetDist = 41.0/35.0;
  AugmentSelection as = {
    false,
    0.0,
    cv::Size(),
    0,
  };
  MetaData meta;
  
  const string& data = datum.data();
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();
  const bool has_uint8 = data.size() > 0;
  int crop_x = param_.crop_size_x();
  int crop_y = param_.crop_size_y();

  CHECK_GT(datum_channels, 0);

  //before any transformation, get the image from datum
  cv::Mat img = cv::Mat::zeros(datum_height, datum_width, CV_8UC3);
  int offset = img.rows * img.cols;
  int dindex;
  Dtype d_element;
  for (int i = 0; i < img.rows; ++i) {
    for (int j = 0; j < img.cols; ++j) {
      cv::Vec3b& rgb = img.at<cv::Vec3b>(i, j);
      for(int c = 0; c < 3; c++){
        dindex = c*offset + i*img.cols + j;
        if (has_uint8)
          d_element = static_cast<Dtype>(static_cast<uint8_t>(data[dindex]));
        else
          d_element = datum.float_data(dindex);
        rgb[c] = d_element;
      }
    }
  }

  //color, contract
  if(param_.do_clahe())
    this->clahe(img, clahe_tileSize, clahe_clipLimit);
  if(param_.gray() == 1){
    cv::cvtColor(img, img, CV_BGR2GRAY);
    cv::cvtColor(img, img, CV_GRAY2BGR);
  }

  int offset3 = 3 * offset;
  int offset1 = datum_width;
  this->ReadMetaData(meta, data, offset3, offset1);
  if(param_.transform_body_joint()) // we expect to transform body joints, and not to transform hand joints
    this->TransformMetaJoints(meta);

  //Start transforming
  cv::Mat img_aug = cv::Mat::zeros(crop_y, crop_x, CV_8UC3);
  cv::Mat img_temp, img_temp2, img_temp3; //size determined by scale
  // We only do random transform as augmentation when training.
  if (phase_ == TRAIN) {
    as.scale = augmentation_scale(img, img_temp, meta);
    as.degree = augmentation_rotate(img_temp, img_temp2, meta);
    as.crop = augmentation_croppad(img_temp2, img_temp3, meta);
    as.flip = augmentation_flip(img_temp3, img_aug, meta);
    if(param_.visualize()) 
      visualize(img_aug, meta, as);
  }
  else {
    img_aug = img.clone();
    as.scale = 1;
    as.crop = cv::Size();
    as.flip = 0;
    as.degree = 0;
  }

  //copy transformed img (img_aug) into transformed_data, do the mean-subtraction here
  offset = img_aug.rows * img_aug.cols;
  for (int i = 0; i < img_aug.rows; ++i) {
    for (int j = 0; j < img_aug.cols; ++j) {
      cv::Vec3b& rgb = img_aug.at<cv::Vec3b>(i, j);
      transformed_data[0*offset + i*img_aug.cols + j] = (rgb[0] - 128)/256.0;
      transformed_data[1*offset + i*img_aug.cols + j] = (rgb[1] - 128)/256.0;
      transformed_data[2*offset + i*img_aug.cols + j] = (rgb[2] - 128)/256.0;
      transformed_data[3*offset + i*img_aug.cols + j] = 0; //zero 4-th channel
    }
  }
  
  putGaussianMaps(transformed_data + 3*offset, meta.objpos, 1, img_aug.cols, img_aug.rows, param_.sigma_center());
  generateLabelMap(transformed_label, img_aug, meta);
}

template<typename Dtype>
float DataTransformer<Dtype>::augmentation_scale(cv::Mat& img_src, cv::Mat& img_temp, MetaData& meta) {
  float dice = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); //[0,1]
  float scale_multiplier;
  if(dice > param_.scale_prob()) {
    img_temp = img_src.clone();
    scale_multiplier = 1;
  }
  else {
    float dice2 = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); //[0,1]
    scale_multiplier = (param_.scale_max() - param_.scale_min()) * dice2 + param_.scale_min(); //linear shear into [scale_min, scale_max]
  }
  float scale_abs = param_.target_dist()/meta.scale_self;
  float scale = scale_abs * scale_multiplier;
  cv::resize(img_src, img_temp, cv::Size(), scale, scale, cv::INTER_CUBIC);
  //modify meta data
  meta.objpos *= scale;
  for(int i=0; i<np; i++){
    meta.joint_self.joints[i] *= scale;
  }
  for(int p=0; p<meta.numOtherPeople; p++){
    meta.objpos_other[p] *= scale;
    for(int i=0; i<np; i++){
      meta.joint_others[p].joints[i] *= scale;
    }
  }
  return scale_multiplier;
}

template<typename Dtype>
bool DataTransformer<Dtype>::onPlane(cv::Point p, cv::Size img_size) {
  if(p.x < 0 || p.y < 0) return false;
  if(p.x >= img_size.width || p.y >= img_size.height) return false;
  return true;
}

template<typename Dtype>
cv::Size DataTransformer<Dtype>::augmentation_croppad(cv::Mat& img_src, cv::Mat& img_dst, MetaData& meta) {
  float dice_x = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); //[0,1]
  float dice_y = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); //[0,1]
  int crop_x = param_.crop_size_x();
  int crop_y = param_.crop_size_y();

  float x_offset = int((dice_x - 0.5) * 2 * param_.center_perterb_max());
  float y_offset = int((dice_y - 0.5) * 2 * param_.center_perterb_max());

  cv::Point2i center = meta.objpos + cv::Point2f(x_offset, y_offset);
  int offset_left = -(center.x - (crop_x/2));
  int offset_up = -(center.y - (crop_y/2));

  img_dst = cv::Mat::zeros(crop_y, crop_x, CV_8UC3) + cv::Scalar(128,128,128);
  for(int i=0;i<crop_y;i++){
    for(int j=0;j<crop_x;j++){ //i,j on cropped
      int coord_x_on_img = center.x - crop_x/2 + j;
      int coord_y_on_img = center.y - crop_y/2 + i;
      if(onPlane(cv::Point(coord_x_on_img, coord_y_on_img), cv::Size(img_src.cols, img_src.rows))){
        img_dst.at<cv::Vec3b>(i,j) = img_src.at<cv::Vec3b>(coord_y_on_img, coord_x_on_img);
      }
    }
  }

  //modify meta data
  cv::Point2f offset(offset_left, offset_up);
  meta.objpos += offset;
  for(int i=0; i<np; i++){
    meta.joint_self.joints[i] += offset;
  }
  for(int p=0; p<meta.numOtherPeople; p++){
    meta.objpos_other[p] += offset;
    for(int i=0; i<np; i++){
      meta.joint_others[p].joints[i] += offset;
    }
  }

  return cv::Size(x_offset, y_offset);
}

template<typename Dtype>
void DataTransformer<Dtype>::swapLeftRight(Joints& j) {
  
  //MPII R leg: 0(ankle), 1(knee), 2(hip)
  //     L leg: 5(ankle), 4(knee), 3(hip)
  //     R arms: 10(wrist), 11(elbow), 12(shoulder)
  //     L arms: 15(wrist), 14(elbow), 13(shoulder)
  if(np == 9){
    int right[4] = {1,2,3,7};
    int left[4] = {4,5,6,8};
    for(int i=0; i<4; i++){
      int ri = right[i] - 1;
      int li = left[i] - 1;
      cv::Point2f temp = j.joints[ri];
      j.joints[ri] = j.joints[li];
      j.joints[li] = temp;
      int temp_v = j.isVisible[ri];
      j.isVisible[ri] = j.isVisible[li];
      j.isVisible[li] = temp_v;
    }
  }
  else if(np == 14){
    int right[6] = {3,4,5,9,10,11}; //1-index
    int left[6] = {6,7,8,12,13,14}; //1-index
    for(int i=0; i<6; i++){
      int ri = right[i] - 1;
      int li = left[i] - 1;
      cv::Point2f temp = j.joints[ri];
      j.joints[ri] = j.joints[li];
      j.joints[li] = temp;
      int temp_v = j.isVisible[ri];
      j.isVisible[ri] = j.isVisible[li];
      j.isVisible[li] = temp_v;
    }
  }
  else if(np == 28){
    int right[11] = {3,4,5,9,10,11,18,19,20,24,25}; //1-index
    int left[11] = {6,7,8,12,13,14,21,22,23,26,27}; //1-index
    for(int i=0; i<11; i++){
      int ri = right[i] - 1;
      int li = left[i] - 1;
      cv::Point2f temp = j.joints[ri];
      j.joints[ri] = j.joints[li];
      j.joints[li] = temp;
      int temp_v = j.isVisible[ri];
      j.isVisible[ri] = j.isVisible[li];
      j.isVisible[li] = temp_v;
    }
  }
}

template<typename Dtype>
bool DataTransformer<Dtype>::augmentation_flip(cv::Mat& img_src, cv::Mat& img_aug, MetaData& meta) {
  bool doflip;
  if(param_.aug_way() == "rand"){
    float dice = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    doflip = (dice <= param_.flip_prob());
  }
  else if(param_.aug_way() == "table"){
    doflip = (aug_flips[meta.write_number][meta.epoch % param_.num_total_augs()] == 1);
  }
  else {
    doflip = 0;
    LOG(INFO) << "Unhandled exception!!!!!!";
  }

  if(doflip){
    cv::flip(img_src, img_aug, 1);
    int w = img_src.cols;

    meta.objpos.x = w - 1 - meta.objpos.x;
    for(int i=0; i<np; i++){
      meta.joint_self.joints[i].x = w - 1 - meta.joint_self.joints[i].x;
    }
    if(param_.transform_body_joint())
      swapLeftRight(meta.joint_self);

    for(int p=0; p<meta.numOtherPeople; p++){
      meta.objpos_other[p].x = w - 1 - meta.objpos_other[p].x;
      for(int i=0; i<np; i++){
        meta.joint_others[p].joints[i].x = w - 1 - meta.joint_others[p].joints[i].x;
      }
      if(param_.transform_body_joint())
        swapLeftRight(meta.joint_others[p]);
    }
  }
  else {
    img_aug = img_src.clone();
  }
  return doflip;
}

template<typename Dtype>
void DataTransformer<Dtype>::RotatePoint(cv::Point2f& p, cv::Mat R){
  cv::Mat point(3,1,CV_64FC1);
  point.at<double>(0,0) = p.x;
  point.at<double>(1,0) = p.y;
  point.at<double>(2,0) = 1;
  cv::Mat new_point = R * point;
  p.x = new_point.at<double>(0,0);
  p.y = new_point.at<double>(1,0);
}

template<typename Dtype>
float DataTransformer<Dtype>::augmentation_rotate(cv::Mat& img_src, cv::Mat& img_dst, MetaData& meta) {
  
  float degree;
  if(param_.aug_way() == "rand"){
    float dice = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    degree = (dice - 0.5) * 2 * param_.max_rotate_degree();
  } else if(param_.aug_way() == "table"){
    degree = aug_degs[meta.write_number][meta.epoch % param_.num_total_augs()];
  } else {
    degree = 0;
    LOG(INFO) << "Unhandled exception!!!!!!";
  }
  
  cv::Point2f center(img_src.cols/2.0, img_src.rows/2.0);
  cv::Mat R = getRotationMatrix2D(center, degree, 1.0);
  cv::Rect bbox = cv::RotatedRect(center, img_src.size(), degree).boundingRect();
  // adjust transformation matrix
  R.at<double>(0,2) += bbox.width/2.0 - center.x;
  R.at<double>(1,2) += bbox.height/2.0 - center.y;
  warpAffine(img_src, img_dst, R, bbox.size(), cv::INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar(128,128,128));
  
  //adjust meta data
  RotatePoint(meta.objpos, R);
  for(int i=0; i<np; i++){
    RotatePoint(meta.joint_self.joints[i], R);
  }
  for(int p=0; p<meta.numOtherPeople; p++){
    RotatePoint(meta.objpos_other[p], R);
    for(int i=0; i<np; i++){
      RotatePoint(meta.joint_others[p].joints[i], R);
    }
  }
  return degree;
}

template<typename Dtype>
void DataTransformer<Dtype>::putGaussianMaps(Dtype* entry, cv::Point2f center, int stride, int grid_x, int grid_y, float sigma){
  //LOG(INFO) << "putGaussianMaps here we start for " << center.x << " " << center.y;
  float start = stride/2.0 - 0.5; //0 if stride = 1, 0.5 if stride = 2, 1.5 if stride = 4, ...
  for (int g_y = 0; g_y < grid_y; g_y++){
    for (int g_x = 0; g_x < grid_x; g_x++){
      float x = start + g_x * stride;
      float y = start + g_y * stride;
      float d2 = (x-center.x)*(x-center.x) + (y-center.y)*(y-center.y);
      float exponent = d2 / 2.0 / sigma / sigma;
      if(exponent > 4.6052){ //ln(100) = -ln(1%)
        continue;
      }
      entry[g_y*grid_x + g_x] += exp(-exponent);
      if(entry[g_y*grid_x + g_x] > 1) 
        entry[g_y*grid_x + g_x] = 1;
    }
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::generateLabelMap(Dtype* transformed_label, cv::Mat& img_aug, MetaData meta) {
  int rezX = img_aug.cols;
  int rezY = img_aug.rows;
  int stride = param_.stride();
  int grid_x = rezX / stride;
  int grid_y = rezY / stride;
  int channelOffset = grid_y * grid_x;

  // clear out transformed_label, it may remain things for last batch
  for (int g_y = 0; g_y < grid_y; g_y++){
    for (int g_x = 0; g_x < grid_x; g_x++){
      for (int i = 0; i < 2*(np+1); i++){
        transformed_label[i*channelOffset + g_y*grid_x + g_x] = 0;
      }
    }
  }
  
  for (int i = 0; i < np; i++){
    cv::Point2f center = meta.joint_self.joints[i];
    if(meta.joint_self.isVisible[i] <= 1){
      putGaussianMaps(transformed_label + i*channelOffset, center, param_.stride(), 
                      grid_x, grid_y, param_.sigma()); //self
      putGaussianMaps(transformed_label + (i+np+1)*channelOffset, center, param_.stride(), 
                      grid_x, grid_y, param_.sigma()); //self
    }
    for(int j = 0; j < meta.numOtherPeople; j++){ //for every other person
      cv::Point2f center = meta.joint_others[j].joints[i];
      if(meta.joint_others[j].isVisible[i] <= 1){
        putGaussianMaps(transformed_label + (i+np+1)*channelOffset, center, param_.stride(), 
                        grid_x, grid_y, param_.sigma());
      }
    }
  }
  
  //put background channel
  for (int g_y = 0; g_y < grid_y; g_y++){
    for (int g_x = 0; g_x < grid_x; g_x++){
      float maximum = 0;
      for (int i = 0; i < np; i++){
        maximum = (maximum > transformed_label[i*channelOffset + g_y*grid_x + g_x]) ? maximum : transformed_label[i*channelOffset + g_y*grid_x + g_x];
      }
      transformed_label[np*channelOffset + g_y*grid_x + g_x] = max(1.0-maximum, 0.0);
      //second background channel
      maximum = 0;
      for (int i = np+1; i < 2*np+1; i++){
        maximum = (maximum > transformed_label[i*channelOffset + g_y*grid_x + g_x]) ? maximum : transformed_label[i*channelOffset + g_y*grid_x + g_x];
      }
      transformed_label[(2*np+1)*channelOffset + g_y*grid_x + g_x] = max(1.0-maximum, 0.0);
    }
  }

  //visualize
  if(param_.visualize()){
    cv::Mat label_map;
    for(int i = 0; i < 2*(np+1); i++){      
      label_map = cv::Mat::zeros(grid_y, grid_x, CV_8UC1);
      for (int g_y = 0; g_y < grid_y; g_y++) {
        for (int g_x = 0; g_x < grid_x; g_x++){
          label_map.at<uchar>(g_y,g_x) = (int)(transformed_label[i*channelOffset + g_y*grid_x + g_x]*255);
        }
      }
      cv::resize(label_map, label_map, cv::Size(), stride, stride, cv::INTER_LINEAR);
      cv::applyColorMap(label_map, label_map, cv::COLORMAP_JET);
      addWeighted(label_map, 0.5, img_aug, 0.5, 0.0, label_map);

      char imagename [100];
      sprintf(imagename, "augment_%04d_label_part_%02d.jpg", meta.write_number, i);
      cv::imwrite(imagename, label_map);
    }
  }
}

void setLabel(cv::Mat& im, const std::string label, const cv::Point& org) {
    int fontface = cv::FONT_HERSHEY_SIMPLEX;
    double scale = 0.5;
    int thickness = 1;
    int baseline = 0;

    cv::Size text = cv::getTextSize(label, fontface, scale, thickness, &baseline);
    cv::rectangle(im, org + cv::Point(0, baseline), org + cv::Point(text.width, -text.height), CV_RGB(0,0,0), CV_FILLED);
    cv::putText(im, label, org, fontface, scale, CV_RGB(255,255,255), thickness, 20);
}

template<typename Dtype>
void DataTransformer<Dtype>::visualize(cv::Mat& img, MetaData meta, AugmentSelection as) {
  cv::Mat img_vis = img.clone();
  static int counter = 0;

  cv::rectangle(img_vis, meta.objpos - cv::Point2f(3,3), meta.objpos + cv::Point2f(3,3), CV_RGB(255,255,0), CV_FILLED);
  for(int i=0;i<np;i++){
    if(np == 21){ // hand case
      if(i < 4)
        cv::circle(img_vis, meta.joint_self.joints[i], 3, CV_RGB(0,0,255), -1);
      else if(i < 6 || i == 12 || i == 13)
        cv::circle(img_vis, meta.joint_self.joints[i], 3, CV_RGB(255,0,0), -1);
      else if(i < 8 || i == 14 || i == 15)
        cv::circle(img_vis, meta.joint_self.joints[i], 3, CV_RGB(255,255,0), -1);
      else if(i < 10|| i == 16 || i == 17)
        cv::circle(img_vis, meta.joint_self.joints[i], 3, CV_RGB(255,100,0), -1);
      else if(i < 12|| i == 18 || i == 19)
        cv::circle(img_vis, meta.joint_self.joints[i], 3, CV_RGB(255,100,100), -1);
      else 
        cv::circle(img_vis, meta.joint_self.joints[i], 3, CV_RGB(0,100,100), -1);
    } else if(np == 9){
      if(i==0 || i==1 || i==2 || i==6)
        cv::circle(img_vis, meta.joint_self.joints[i], 3, CV_RGB(0,0,255), -1);
      else if(i==3 || i==4 || i==5 || i==7)
        cv::circle(img_vis, meta.joint_self.joints[i], 3, CV_RGB(255,0,0), -1);
      else
        cv::circle(img_vis, meta.joint_self.joints[i], 3, CV_RGB(255,255,0), -1);
    } else if(np == 14 || np == 28) {//body case
      if(i < 14) {
        if(i==2 || i==3 || i==4 || i==8 || i==9 || i==10)
          cv::circle(img_vis, meta.joint_self.joints[i], 3, CV_RGB(0,0,255), -1);
        else if(i==5 || i==6 || i==7 || i==11 || i==12 || i==13)
          cv::circle(img_vis, meta.joint_self.joints[i], 3, CV_RGB(255,0,0), -1);
        else
          cv::circle(img_vis, meta.joint_self.joints[i], 3, CV_RGB(255,255,0), -1);
      } else if(i < 16)
        cv::circle(img_vis, meta.joint_self.joints[i], 3, CV_RGB(0,255,0), -1);
      else {
        if(i==17 || i==18 || i==19 || i==23 || i==24)
          cv::circle(img_vis, meta.joint_self.joints[i], 3, CV_RGB(255,0,0), -1);
        else if(i==20 || i==21 || i==22 || i==25 || i==26)
          cv::circle(img_vis, meta.joint_self.joints[i], 3, CV_RGB(255,100,100), -1);
        else
          cv::circle(img_vis, meta.joint_self.joints[i], 3, CV_RGB(255,200,200), -1);
      }
    }
  }
  
  cv::line(img_vis, meta.objpos + cv::Point2f(-368/2,-368/2), meta.objpos + cv::Point2f(368/2,-368/2), CV_RGB(0,255,0), 2);
  cv::line(img_vis, meta.objpos + cv::Point2f(368/2,-368/2), meta.objpos + cv::Point2f(368/2,368/2), CV_RGB(0,255,0), 2);
  cv::line(img_vis, meta.objpos + cv::Point2f(368/2,368/2), meta.objpos + cv::Point2f(-368/2,368/2), CV_RGB(0,255,0), 2);
  cv::line(img_vis, meta.objpos + cv::Point2f(-368/2,368/2), meta.objpos + cv::Point2f(-368/2,-368/2), CV_RGB(0,255,0), 2);

  for(int p=0;p<meta.numOtherPeople;p++){
    cv::rectangle(img_vis, meta.objpos_other[p] - cv::Point2f(3,3), meta.objpos_other[p] + cv::Point2f(3,3), CV_RGB(0,255,255), CV_FILLED);
    for(int i=0;i<np;i++){
      cv::circle(img_vis, meta.joint_others[p].joints[i], 2, CV_RGB(0,0,0), -1);
    }
  }
  
  // draw text
  if(phase_ == TRAIN){
    std::stringstream ss;

    ss << meta.dataset << " " << meta.write_number << " index:" << meta.annolist_index << "; p:" << meta.people_index 
       << "; o_scale: " << meta.scale_self;
    string str_info = ss.str();
    setLabel(img_vis, str_info, cv::Point(0, 20));

    stringstream ss2; 
    ss2 << "mult: " << as.scale << "; rot: " << as.degree << "; flip: " << (as.flip?"true":"ori");
    str_info = ss2.str();
    setLabel(img_vis, str_info, cv::Point(0, 40));

    cv::rectangle(img_vis, cv::Point(0, 0+img_vis.rows), cv::Point(param_.crop_size_x(), param_.crop_size_y()+img_vis.rows), cv::Scalar(255,255,255), 1);

    char imagename [100];
    sprintf(imagename, "augment_%04d_epoch_%d_writenum_%d.jpg", counter, meta.epoch, meta.write_number);
    cv::imwrite(imagename, img_vis);
  }
  else {
    string str_info = "no augmentation for testing";
    setLabel(img_vis, str_info, cv::Point(0, 20));

    char imagename [100];
    sprintf(imagename, "augment_%04d.jpg", counter);
    cv::imwrite(imagename, img_vis);
  }
  counter++;
}


template<typename Dtype>
void DataTransformer<Dtype>::Transform(const vector<Datum> & datum_vector,
                                       Blob<Dtype>* transformed_blob) {
  const int datum_num = datum_vector.size();
  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();

  CHECK_GT(datum_num, 0) << "There is no datum to add";
  CHECK_LE(datum_num, num) <<
    "The size of datum_vector must be no greater than transformed_blob->num()";
  Blob<Dtype> uni_blob(1, channels, height, width);
  for (int item_id = 0; item_id < datum_num; ++item_id) {
    int offset = transformed_blob->offset(item_id);
    uni_blob.set_cpu_data(transformed_blob->mutable_cpu_data() + offset);
    Transform(datum_vector[item_id], &uni_blob);
  }
}

#ifdef USE_OPENCV
template<typename Dtype>
void DataTransformer<Dtype>::Transform(const vector<cv::Mat> & mat_vector,
                                       Blob<Dtype>* transformed_blob) {
  const int mat_num = mat_vector.size();
  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();

  CHECK_GT(mat_num, 0) << "There is no MAT to add";
  CHECK_EQ(mat_num, num) <<
    "The size of mat_vector must be equals to transformed_blob->num()";
  Blob<Dtype> uni_blob(1, channels, height, width);
  for (int item_id = 0; item_id < mat_num; ++item_id) {
    int offset = transformed_blob->offset(item_id);
    uni_blob.set_cpu_data(transformed_blob->mutable_cpu_data() + offset);
    Transform(mat_vector[item_id], &uni_blob);
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const cv::Mat& cv_img,
                                       Blob<Dtype>* transformed_blob) {
  const int crop_size = param_.crop_size();
  const int img_channels = cv_img.channels();
  const int img_height = cv_img.rows;
  const int img_width = cv_img.cols;

  // Check dimensions.
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int num = transformed_blob->num();

  CHECK_EQ(channels, img_channels);
  CHECK_LE(height, img_height);
  CHECK_LE(width, img_width);
  CHECK_GE(num, 1);

  CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";

  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;

  CHECK_GT(img_channels, 0);
  CHECK_GE(img_height, crop_size);
  CHECK_GE(img_width, crop_size);

  Dtype* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(img_channels, data_mean_.channels());
    CHECK_EQ(img_height, data_mean_.height());
    CHECK_EQ(img_width, data_mean_.width());
    mean = data_mean_.mutable_cpu_data();
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == img_channels) <<
     "Specify either 1 mean_value or as many as channels: " << img_channels;
    if (img_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < img_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }

  int h_off = 0;
  int w_off = 0;
  cv::Mat cv_cropped_img = cv_img;
  if (crop_size) {
    CHECK_EQ(crop_size, height);
    CHECK_EQ(crop_size, width);
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = Rand(img_height - crop_size + 1);
      w_off = Rand(img_width - crop_size + 1);
    } else {
      h_off = (img_height - crop_size) / 2;
      w_off = (img_width - crop_size) / 2;
    }
    cv::Rect roi(w_off, h_off, crop_size, crop_size);
    cv_cropped_img = cv_img(roi);
  } else {
    CHECK_EQ(img_height, height);
    CHECK_EQ(img_width, width);
  }

  CHECK(cv_cropped_img.data);

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();
  int top_index;
  for (int h = 0; h < height; ++h) {
    const uchar* ptr = cv_cropped_img.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < width; ++w) {
      for (int c = 0; c < img_channels; ++c) {
        if (do_mirror) {
          top_index = (c * height + h) * width + (width - 1 - w);
        } else {
          top_index = (c * height + h) * width + w;
        }
        // int top_index = (c * height + h) * width + w;
        Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
        if (has_mean_file) {
          int mean_index = (c * img_height + h_off + h) * img_width + w_off + w;
          transformed_data[top_index] =
            (pixel - mean[mean_index]) * scale;
        } else {
          if (has_mean_values) {
            transformed_data[top_index] =
              (pixel - mean_values_[c]) * scale;
          } else {
            transformed_data[top_index] = pixel * scale;
          }
        }
      }
    }
  }
}
#endif  // USE_OPENCV

template<typename Dtype>
void DataTransformer<Dtype>::Transform(Blob<Dtype>* input_blob,
                                       Blob<Dtype>* transformed_blob) {
  const int crop_size = param_.crop_size();
  const int input_num = input_blob->num();
  const int input_channels = input_blob->channels();
  const int input_height = input_blob->height();
  const int input_width = input_blob->width();

  if (transformed_blob->count() == 0) {
    // Initialize transformed_blob with the right shape.
    if (crop_size) {
      transformed_blob->Reshape(input_num, input_channels,
                                crop_size, crop_size);
    } else {
      transformed_blob->Reshape(input_num, input_channels,
                                input_height, input_width);
    }
  }

  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int size = transformed_blob->count();

  CHECK_LE(input_num, num);
  CHECK_EQ(input_channels, channels);
  CHECK_GE(input_height, height);
  CHECK_GE(input_width, width);


  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;

  int h_off = 0;
  int w_off = 0;
  if (crop_size) {
    CHECK_EQ(crop_size, height);
    CHECK_EQ(crop_size, width);
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = Rand(input_height - crop_size + 1);
      w_off = Rand(input_width - crop_size + 1);
    } else {
      h_off = (input_height - crop_size) / 2;
      w_off = (input_width - crop_size) / 2;
    }
  } else {
    CHECK_EQ(input_height, height);
    CHECK_EQ(input_width, width);
  }

  Dtype* input_data = input_blob->mutable_cpu_data();
  if (has_mean_file) {
    CHECK_EQ(input_channels, data_mean_.channels());
    CHECK_EQ(input_height, data_mean_.height());
    CHECK_EQ(input_width, data_mean_.width());
    for (int n = 0; n < input_num; ++n) {
      int offset = input_blob->offset(n);
      caffe_sub(data_mean_.count(), input_data + offset,
            data_mean_.cpu_data(), input_data + offset);
    }
  }

  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == input_channels) <<
     "Specify either 1 mean_value or as many as channels: " << input_channels;
    if (mean_values_.size() == 1) {
      caffe_add_scalar(input_blob->count(), -(mean_values_[0]), input_data);
    } else {
      for (int n = 0; n < input_num; ++n) {
        for (int c = 0; c < input_channels; ++c) {
          int offset = input_blob->offset(n, c);
          caffe_add_scalar(input_height * input_width, -(mean_values_[c]),
            input_data + offset);
        }
      }
    }
  }

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();

  for (int n = 0; n < input_num; ++n) {
    int top_index_n = n * channels;
    int data_index_n = n * channels;
    for (int c = 0; c < channels; ++c) {
      int top_index_c = (top_index_n + c) * height;
      int data_index_c = (data_index_n + c) * input_height + h_off;
      for (int h = 0; h < height; ++h) {
        int top_index_h = (top_index_c + h) * width;
        int data_index_h = (data_index_c + h) * input_width + w_off;
        if (do_mirror) {
          int top_index_w = top_index_h + width - 1;
          for (int w = 0; w < width; ++w) {
            transformed_data[top_index_w-w] = input_data[data_index_h + w];
          }
        } else {
          for (int w = 0; w < width; ++w) {
            transformed_data[top_index_h + w] = input_data[data_index_h + w];
          }
        }
      }
    }
  }
  if (scale != Dtype(1)) {
    DLOG(INFO) << "Scale: " << scale;
    caffe_scal(size, scale, transformed_data);
  }
}

template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(const Datum& datum) {
  if (datum.encoded()) {
#ifdef USE_OPENCV
    CHECK(!(param_.force_color() && param_.force_gray()))
        << "cannot set both force_color and force_gray";
    cv::Mat cv_img;
    if (param_.force_color() || param_.force_gray()) {
    // If force_color then decode in color otherwise decode in gray.
      cv_img = DecodeDatumToCVMat(datum, param_.force_color());
    } else {
      cv_img = DecodeDatumToCVMatNative(datum);
    }
    // InferBlobShape using the cv::image.
    return InferBlobShape(cv_img);
#else
    LOG(FATAL) << "Encoded datum requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  }
  const int crop_size = param_.crop_size();
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();
  // Check dimensions.
  CHECK_GT(datum_channels, 0);
  CHECK_GE(datum_height, crop_size);
  CHECK_GE(datum_width, crop_size);
  // Build BlobShape.
  vector<int> shape(4);
  shape[0] = 1;
  shape[1] = datum_channels;
  shape[2] = (crop_size)? crop_size: datum_height;
  shape[3] = (crop_size)? crop_size: datum_width;
  return shape;
}

template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(
    const vector<Datum> & datum_vector) {
  const int num = datum_vector.size();
  CHECK_GT(num, 0) << "There is no datum to in the vector";
  // Use first datum in the vector to InferBlobShape.
  vector<int> shape = InferBlobShape(datum_vector[0]);
  // Adjust num to the size of the vector.
  shape[0] = num;
  return shape;
}

#ifdef USE_OPENCV
template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(const cv::Mat& cv_img) {
  const int crop_size = param_.crop_size();
  const int img_channels = cv_img.channels();
  const int img_height = cv_img.rows;
  const int img_width = cv_img.cols;
  // Check dimensions.
  CHECK_GT(img_channels, 0);
  CHECK_GE(img_height, crop_size);
  CHECK_GE(img_width, crop_size);
  // Build BlobShape.
  vector<int> shape(4);
  shape[0] = 1;
  shape[1] = img_channels;
  shape[2] = (crop_size)? crop_size: img_height;
  shape[3] = (crop_size)? crop_size: img_width;
  return shape;
}

template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(
    const vector<cv::Mat> & mat_vector) {
  const int num = mat_vector.size();
  CHECK_GT(num, 0) << "There is no cv_img to in the vector";
  // Use first cv_img in the vector to InferBlobShape.
  vector<int> shape = InferBlobShape(mat_vector[0]);
  // Adjust num to the size of the vector.
  shape[0] = num;
  return shape;
}
#endif  // USE_OPENCV

template <typename Dtype>
void DataTransformer<Dtype>::InitRand() {
  const bool needs_rand = param_.mirror() ||
      (phase_ == TRAIN && param_.crop_size());
  if (needs_rand) {
    const unsigned int rng_seed = caffe_rng_rand();
    rng_.reset(new Caffe::RNG(rng_seed));
  } else {
    rng_.reset();
  }
}

template <typename Dtype>
int DataTransformer<Dtype>::Rand(int n) {
  CHECK(rng_);
  CHECK_GT(n, 0);
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(rng_->generator());
  return ((*rng)() % n);
}

template <typename Dtype>
void DataTransformer<Dtype>::clahe(cv::Mat& bgr_image, int tileSize, int clipLimit) {
  cv::Mat lab_image;
  cv::cvtColor(bgr_image, lab_image, CV_BGR2Lab);

  // Extract the L channel
  vector<cv::Mat> lab_planes(3);
  split(lab_image, lab_planes);  // now we have the L image in lab_planes[0]

  // apply the CLAHE algorithm to the L channel
  cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(clipLimit, cv::Size(tileSize, tileSize));
  cv::Mat dst;
  clahe->apply(lab_planes[0], dst);

  // Merge the the color planes back into an Lab image
  dst.copyTo(lab_planes[0]);
  merge(lab_planes, lab_image);

  // convert back to RGB
  cv::Mat image_clahe;
  cv::cvtColor(lab_image, image_clahe, CV_Lab2BGR);
  bgr_image = image_clahe.clone();
}


INSTANTIATE_CLASS(DataTransformer);

}  // namespace caffe
