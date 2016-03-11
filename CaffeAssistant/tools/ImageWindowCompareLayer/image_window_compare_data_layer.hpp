#ifndef CAFFE_IMAGE_DATA_LAYER_HPP_
#define CAFFE_IMAGE_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Provides data to the Net from image files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class ImageWindowCompareDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit ImageWindowCompareDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~ImageWindowCompareDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ImageData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void ShuffleImages();
  virtual void ShuffleWindows();
  virtual void cropToWindows();
  virtual void preprocess(cv::Mat & input, cv::Mat & output);
  virtual void load_batch(Batch<Dtype>* batch);
  virtual void bg_edge_pick(cv::Mat &bin, cv::Mat & mask,
			std::vector<std::pair<int, int> > & pick_vec);
  virtual void bg_random_pick(cv::Mat &bin, cv::Mat & mask,
		  std::vector<std::pair<int, int> > & pick_vec);

  // lines_: a vector that contains the rgb file names and label file names
  vector<std::pair<std::string, std::string> > lines_;
  //windows_lines: a vector that contains a pair with file ID and center points locations
  vector<std::pair<int/*img order*/,
         std::pair<int/*label*/, cv::Point >
        > > windows_lines;
  vector<cv::Mat> rgb_mat_vec; // all rgb file will be loaded here.
  vector<cv::Mat> label_mat_vec;
  int lines_id_;
  int window_height, window_width;
};


}  // namespace caffe

#endif  // CAFFE_IMAGE_DATA_LAYER_HPP_

