//#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <boost/range/algorithm/random_shuffle.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

//#include <sstream>
#include <random>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/image_window_compare_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
ImageWindowCompareDataLayer<Dtype>::~ImageWindowCompareDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void ImageWindowCompareDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  window_height = this->layer_param_.image_window_compare_data_param().window_height();
  window_width  = this->layer_param_.image_window_compare_data_param().window_width();
  const int new_height = this->layer_param_.image_window_compare_data_param().new_height();
  const int new_width  = this->layer_param_.image_window_compare_data_param().new_width();
  const bool is_color  = this->layer_param_.image_window_compare_data_param().is_color();
  string root_folder = this->layer_param_.image_window_compare_data_param().root_folder();

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  // Read the file with rgb files and label files
  const string& source = this->layer_param_.image_window_compare_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string filename;
  string labelname;
  int count = 0;
  while (infile >> filename >> labelname) {
    lines_.push_back(std::make_pair(filename, labelname));
    cv::Mat rgb_img = ReadImageToCVMat(root_folder + filename,
                                      new_height, new_width, is_color);
    cv::Mat label_img = ReadImageToCVMat(root_folder + labelname,
                                      new_height, new_width, false);
    CHECK(rgb_img.data) << "Could not load " << filename;
    CHECK(label_img.data) << "Could not load " << labelname;
    preprocess(rgb_img, rgb_img);
    rgb_mat_vec.push_back(rgb_img);
    label_mat_vec.push_back(label_img);
    ++count;
  }
  windows_lines.reserve(int(new_height * new_width * 0.1 * count * 2));

  if (this->layer_param_.image_window_compare_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.\n";

  LOG(INFO) << "Before crop!\n";
  cropToWindows();
  LOG(INFO) << "After crop!\n";
  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
//  if (this->layer_param_.image_data_param().rand_skip()) {
//    unsigned int skip = caffe_rng_rand() %
//        this->layer_param_.image_data_param().rand_skip();
//    LOG(INFO) << "Skipping first " << skip << " data points.";
//    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
//    lines_id_ = skip;
//  }
  // Read an image, and use it to initialize the top blob.
//  cv::Mat rgb_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
//                                    new_height, new_width, is_color);
//  cv::Mat label_img = ReadImageToCVMat(root_folder + lines_[lines_id_].second,
//                                    new_height, new_width, is_color);
//  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
  // Use data_transformer to infer the expected blob shape from a cv_image.

  cv::Rect rect(windows_lines[0].second.second.x, windows_lines[0].second.second.y,
		  2 * window_height + 1, 2 * window_width + 1 );
  cv::Mat sample_img = rgb_mat_vec[windows_lines[0].first](rect);
  vector<int> top_shape = this->data_transformer_->InferBlobShape(sample_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape prefetch_data and top[0] according to the batch_size.
  const int batch_size = this->layer_param_.image_window_compare_data_param().batch_size();
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  top_shape[0] = batch_size;
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  top[0]->Reshape(top_shape);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  vector<int> label_shape(1, batch_size);
  top[1]->Reshape(label_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].label_.Reshape(label_shape);
  }
}

template <typename Dtype>
void ImageWindowCompareDataLayer<Dtype>::ShuffleImages() {
//  caffe::rng_t* prefetch_rng =
//      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
//  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

/**
 * This function is to make the images in lines_ some patches.
 * Here we just need record the image tag and location.
 */
template <typename Dtype>
void ImageWindowCompareDataLayer<Dtype>::cropToWindows(){
  assert(rgb_mat_vec.size() == 0 || label_mat_vec.size() == 0 ||
		  window_height < 0 || window_height > label_mat_vec[0].rows ||
		  window_width < 0 || window_width > label_mat_vec[0].cols);
//  LOG(INFO) << "ERROR: image border parameter is not matched.";
//  exit(-1);

  std::vector<cv::Mat> split_vec;
  int tag = 0;
  //make windows_lines
  LOG(INFO) << "Before main loop!\n";
  std::vector<std::pair<int, int> > picked_locations; //negative locations
  for(int i = 0; i < rgb_mat_vec.size(); ++i){
	picked_locations.clear();
    cv::Mat rgb_img, label_img;
    rgb_mat_vec[i].copyTo(rgb_img);
    label_mat_vec[i].copyTo(label_img);
    cv::split(rgb_img, split_vec);
    cv::Mat mask = split_vec[2] > 25; // red channel > 25
    cv::copyMakeBorder(rgb_img, rgb_img,
    		window_height, window_height,
			window_width, window_width,
			cv::BORDER_CONSTANT, 0);
    rgb_mat_vec[i] = rgb_img;  //From now on, the size of rgb_img is different from label_img
    cv::Mat bin = label_img > 128;
    cv::Mat nonzero_locations, zero_locations;
    cv::findNonZero(bin, nonzero_locations);
    picked_locations.reserve(nonzero_locations.total());

//    caffe::rng_t* order_rng =
//          static_cast<caffe::rng_t*>(prefetch_rng_->generator());

    bg_edge_pick(bin, mask, picked_locations);
    for(int i = 0; i < nonzero_locations.rows; ++i){
    	windows_lines.push_back(std::make_pair(tag,
    			std::make_pair(label_img.at<bool>(nonzero_locations.at<cv::Point>(i, 0)) > 0,
    			nonzero_locations.at<cv::Point>(i, 0) ) ) );
    	windows_lines.push_back(std::make_pair(tag,
    	    			std::make_pair(
    	    					label_img.at<bool>(picked_locations[i].first, picked_locations[i].second) > 0,
    	    			cv::Point(picked_locations[i].first, picked_locations[i].second))));
    }
    ++tag;
  }
}

template <typename Dtype>
void ImageWindowCompareDataLayer<Dtype>::bg_random_pick(cv::Mat &bin, cv::Mat & mask,
		std::vector<std::pair<int, int> > & pick_vec){
	cv::Mat zero_locations, nonzero_locations;
    cv::findNonZero((~bin) & mask, zero_locations);
    cv::findNonZero(bin, nonzero_locations);
    int num = nonzero_locations.total();
    std::vector<int> zero_ind(zero_locations.rows);
    for(int i = 0; i < zero_locations.rows; ++i){
      zero_ind[i] = i;
    }
    std::random_shuffle(zero_ind.begin(), zero_ind.end());
    for(int i = 0; i < num; ++i){
    	pick_vec.push_back(std::make_pair(zero_locations.at<cv::Point>(zero_ind[i], 0).x,
    			zero_locations.at<cv::Point>(zero_ind[i], 0).y));
    }
}

template <typename Dtype>
void ImageWindowCompareDataLayer<Dtype>::bg_edge_pick(cv::Mat &bin, cv::Mat & mask,
		std::vector<std::pair<int, int> > & pick_vec){
	cv::Mat zero_locations, nonzero_locations;
    cv::findNonZero((~bin) & mask, zero_locations);
    std::vector<int> zero_ind(zero_locations.rows);
    cv::findNonZero(bin, nonzero_locations);
    int num = nonzero_locations.total();
//    int iter = 0;
    cv::Mat dilate_struct = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2,2),
    		cv::Point(-1,-1));
    cv::Mat bin_dilate;
    cv::dilate(bin, bin_dilate, dilate_struct);
    cv::Mat edge;
    cv::Canny(bin_dilate, edge, 3, 9, 3);
#ifdef IWCD_DEBUG
    cv::imshow("edge", edge);
    cv::waitKey();
#endif
    cv::Mat edge_locations;
    cv::findNonZero(edge, edge_locations);
    int edge_num = edge_locations.total();
    assert(edge_num < num);
    for(int i = 0; i < edge_num; ++i){
    	pick_vec.push_back(std::make_pair(edge_locations.at<cv::Point>(i, 0).x,
    			edge_locations.at<cv::Point>(i, 0).y));
    }
    if(edge_num < num){
        std::vector<int> zero_ind(zero_locations.rows);
        for(int i = 0; i < zero_locations.rows; ++i){
          zero_ind[i] = i;
        }
        std::random_shuffle(zero_ind.begin(), zero_ind.end());
        for(int i = 0; i < num - edge_num; ++i){
        	pick_vec.push_back(std::make_pair(zero_locations.at<cv::Point>(zero_ind[i], 0).x,
        			zero_locations.at<cv::Point>(zero_ind[i], 0).y));
        }
    }
}

template <typename Dtype>
void ImageWindowCompareDataLayer<Dtype>::ShuffleWindows() {
//  caffe::rng_t* prefetch_rng =
//      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
   std::random_shuffle(windows_lines.begin(), windows_lines.end());
}

// This function is called on prefetch thread
template <typename Dtype>
void ImageWindowCompareDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  ImageWindowCompareDataParameter image_data_param = this->layer_param_.image_window_compare_data_param();
  const int batch_size = image_data_param.batch_size();
//  const int new_height = image_data_param.new_height();
//  const int new_width = image_data_param.new_width();
//  const bool is_color = image_data_param.is_color();
  string root_folder = image_data_param.root_folder();

  // Reshape according to the first image of each batch
  // on single input batches allows for inputs of varying dimension.
//  cv::Mat rgb_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
//      new_height, new_width, is_color);
//  cv::Mat label_img = ReadImageToCVMat(root_folder + lines_[lines_id_].second,
//      new_height, new_width, false);
//  CHECK(rgb_img.data) << "Could not load " << lines_[lines_id_].first;
//  CHECK(label_img.data) << "Could not load " << lines_[lines_id_].second;
  // Use data_transformer to infer the expected blob shape from a cv_img.

  cv::Rect rect(windows_lines[0].second.second.x, windows_lines[0].second.second.y,
		  2 * window_height + 1, 2 * window_width + 1 );
  cv::Mat sample_img = rgb_mat_vec[windows_lines[0].first](rect);



  vector<int> top_shape = this->data_transformer_->InferBlobShape(sample_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();

  // datum scales
  const int lines_size = windows_lines.size();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
	  //LOG(INFO) << "item id is： " << item_id;
    timer.Start();
    CHECK_GT(lines_size, lines_id_);
//    cv::Mat rgb_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
//        new_height, new_width, is_color);
//    cv::Mat label_img = ReadImageToCVMat(root_folder + lines_[lines_id_].second,
//            new_height, new_width, is_color);
//    CHECK(rgb_img.data) << "Could not load " << lines_[lines_id_].first;
//    CHECK(label_img.data) << "Could not load " << lines_[lines_id_].second;

    cv::Rect cv_rect(windows_lines[lines_id_].second.second.x, windows_lines[lines_id_].second.second.y,
    		  2 * window_height + 1, 2 * window_width + 1 );
    cv::Mat cv_img = rgb_mat_vec[windows_lines[lines_id_].first](cv_rect);
    assert(cv_img.data);
    read_time += timer.MicroSeconds();
    /*stringstream ss;
    ss << item_id;
    cv::putText( cv_img, ss.str(), cv::Point(30,30), 7,
                 0.5, cv::Scalar(255,255,255), 1, 8);
    cv::imshow("sample_img",cv_img);
    LOG(INFO) << "Wait key\n";
    cv::waitKey(20);
    LOG(INFO) << "Wait over\n";*/
    timer.Start();
    // Apply transformations (mirror, crop...) to the image
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(prefetch_data + offset);
    this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
    trans_time += timer.MicroSeconds();

    prefetch_label[item_id] = windows_lines[lines_id_].second.first;
    // go to the next iter
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.image_window_compare_data_param().shuffle()) {
        ShuffleWindows();
      }
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

template <typename Dtype>
void ImageWindowCompareDataLayer<Dtype>::preprocess(cv::Mat & input, cv::Mat & output){
	//TODO
}




INSTANTIATE_CLASS(ImageWindowCompareDataLayer);
REGISTER_LAYER_CLASS(ImageWindowCompareData);

}  // namespace caffe
//#endif  // USE_OPENCV

