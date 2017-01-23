#include <stddef.h>
#include <time.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
}

#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//////////////////////////////////////////////////////////////////////////////
//
// Data Layout in caffe's Blob data structure is 4-dimension array
//
//   (number, channel, height, width)
//
// so the GPU memory layout is:
// float* buffer = net->input_blobs()[0]->mutable_cpu_data();
// buffer-> +========================+
//          | image[0] red channel   |
//          +------------------------+
//          | image[0] green channel |
//          +------------------------+
//          | image[0] blue channel  |
//          +========================+
//          | image[1] red channel   |
//          +------------------------+
//          | image[1] green channel |
//          +------------------------+
//          | image[1] blue channel  |
//          +========================+
//          | image[2] red channel   |
//          +------------------------+
//          |          ...           |
//
// value has to be:
//   0. format: 32 bit floating point
//   1. substract mean? yes
//   2. scale? don't know
//


/* Pair (label, confidence) representing a prediction. */
typedef std::pair<std::string, float> Prediction;
typedef std::vector<Prediction> PredictionList;

//////////////////////////////////////////////////////////////////////////////
// FFmpeg part
//////////////////////////////////////////////////////////////////////////////

class VideoDecoder {
 public:
  VideoDecoder(const std::string& video_file);
  ~VideoDecoder();

  // if ignoreFrame is true, the frame will still be decoded, but will not
  // convert the frame to RGB and not write the result frame out. useful when
  // you want to ignore some frame.
  //
  // return true if decode a frame, false if end-of-video
  bool DecodeOneFrame(bool ignoreFrame);

  AVFrame* GetFrame() { return pFrameRGB; }

 private:
  AVFormatContext*   pFormatCtx;
  AVCodec*           pCodec;
  AVCodecContext*    pCodecCtx;
  AVFrame*           pFrame;
  AVFrame*           pFrameRGB;
  uint8_t*           pBuffer;
  int                firstVideoStreamIndex;
  struct SwsContext* sws_ctx;
} ;

VideoDecoder::VideoDecoder(const std::string& video_file)
  : pFormatCtx(NULL)
  , pCodec(NULL)
  , pCodecCtx(NULL)
  , pFrame(avcodec_alloc_frame())
  , pFrameRGB(avcodec_alloc_frame())
  , pBuffer(NULL)
  , firstVideoStreamIndex(-1)
  , sws_ctx(NULL)
{
  // Open video file
  if (avformat_open_input(&pFormatCtx, video_file.c_str(), NULL, NULL)!=0) {
    std::cerr << "Couldn't open file!" << std::endl;
    assert(0 && "Couldn't open file!");
  }

  // Retrieve stream information
  if(avformat_find_stream_info(pFormatCtx, NULL)<0) {
    std::cerr << "Couldn't find stream information!" << std::endl;
    assert(0 && "Couldn't find stream information!");
  }

  // Dump information about file onto standard error
  av_dump_format(pFormatCtx, 0, video_file.c_str(), 0);

  // Find the first video stream
  for(unsigned int i=0; i<pFormatCtx->nb_streams; i++) {
    if(pFormatCtx->streams[i]->codec->codec_type==AVMEDIA_TYPE_VIDEO) {
      firstVideoStreamIndex = i;
      break;
    }
  }
  if(firstVideoStreamIndex == -1) {
    std::cerr << "Didn't find a video stream!" << std::endl;
    assert(0 && "Didn't find a video stream!");
  }

  // Get a pointer to the codec context for the video stream
  pCodecCtx=pFormatCtx->streams[firstVideoStreamIndex]->codec;

  // Find the decoder for the video stream
  pCodec=avcodec_find_decoder(pCodecCtx->codec_id);
  if(!pCodec) {
    std::cerr << "Unsupported codec!" << std::endl;
    assert(0 && "Unsupported codec!");
  }

  // Open codec
  if(avcodec_open2(pCodecCtx, pCodec, NULL) < 0) {
    std::cerr << "Could not open codec!" << std::endl;
    assert(0 && "Could not open codec!");
  }

  if(pFrame==NULL) {
    std::cerr << "Could not allocate output video frame" << std::endl;
    assert(0 && "Could not allocate output video frame");
  }

  if(pFrameRGB==NULL) {
    std::cerr << "Could not allocate output RGB video frame" << std::endl;
    assert(0 && "Could not allocate output RGB video frame");
  }

  int numBytes = 0;
  // Determine required buffer size and allocate buffer
  numBytes=avpicture_get_size(AV_PIX_FMT_RGB24, pCodecCtx->width, pCodecCtx->height);
  pBuffer=(uint8_t *)av_malloc(numBytes*sizeof(uint8_t));

  // Assign appropriate parts of buffer to image planes in pFrameRGB
  // Note that pFrameRGB is an AVFrame, but AVFrame is a superset of AVPicture
  avpicture_fill((AVPicture *)pFrameRGB, pBuffer, AV_PIX_FMT_RGB24, pCodecCtx->width, pCodecCtx->height);

  // initialize SWS context for software scaling
  sws_ctx = sws_getContext(pCodecCtx->width, pCodecCtx->height, pCodecCtx->pix_fmt, pCodecCtx->width,
    pCodecCtx->height, AV_PIX_FMT_RGB24, SWS_BILINEAR, NULL, NULL, NULL);
}

VideoDecoder::~VideoDecoder() {
  av_free(pBuffer);
  av_free(pFrameRGB);
  av_free(pFrame);

  avcodec_close(pCodecCtx);
  avformat_close_input(&pFormatCtx);
}

bool VideoDecoder::DecodeOneFrame(bool ignoreFrame) {
  AVPacket packet;
  while (av_read_frame(pFormatCtx, &packet) >= 0) {
    // Is this a packet from the video stream?
    if (packet.stream_index != firstVideoStreamIndex)
      continue;

    // Decode video frame
    int gotPicturePtr;
    avcodec_decode_video2(pCodecCtx, pFrame, &gotPicturePtr, &packet);
    // Did we get a video frame?
    if(!gotPicturePtr)
      continue;

    // do we want skip this frame?
    if (ignoreFrame) {
      av_free_packet(&packet);
      return true;
    }

    // Convert the image from its native format to RGB
    sws_scale(sws_ctx, (uint8_t const * const *)pFrame->data, pFrame->linesize, 0, pCodecCtx->height, pFrameRGB->data, pFrameRGB->linesize);
    return true;
  }
  av_free_packet(&packet);
  return false;
}

//////////////////////////////////////////////////////////////////////////////
// Caffe part
//////////////////////////////////////////////////////////////////////////////

class Classifier {
 public:
  Classifier(const std::string& model_file,
             const std::string& trained_file,
             const std::string& mean_file,
             const std::string& label_file);

  std::vector<PredictionList> Classify(const std::vector<cv::Mat>& imgs, int N = 5);
  // return true if this batch is full and ready to fire, false if not
  bool PushImage(AVFrame* f);
  std::vector<std::vector<float> > ForwardBatch();

 private:
  void SetMean(const std::string& mean_file);
  std::vector<std::vector<float> > Predict(const std::vector<cv::Mat>& img);
  void WrapInputLayer(std::vector<cv::Mat>* input_channels, int num_images);
  void Preprocess(const std::vector<cv::Mat>& img,
                  std::vector<cv::Mat>* input_channels);


 private:
  boost::shared_ptr<caffe::Net<float> > net_;
  cv::Size input_geometry_;
  int height_;
  int width_;
  int num_channels_;
  int batch_size_;
  cv::Mat mean_;
  std::vector<std::string> labels_;

  int image_size_;
  int channel_size_;
  float* mutable_cpu_data_;
  int num_pushed_image_;
};

Classifier::Classifier(const std::string& model_file,
                       const std::string& trained_file,
                       const std::string& mean_file,
                       const std::string& label_file)
  : num_pushed_image_(0)
{
#ifdef CPU_ONLY
  caffe::Caffe::set_mode(caffe::Caffe::CPU);
#else
  caffe::Caffe::set_mode(caffe::Caffe::GPU);
#endif

  /* Load the network. */
  net_.reset(new caffe::Net<float>(model_file, caffe::TEST));
  net_->CopyTrainedLayersFrom(trained_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  caffe::Blob<float>* input_layer = net_->input_blobs()[0];
  batch_size_       = input_layer->num();
  num_channels_     = input_layer->channels();
  height_           = input_layer->height();
  width_            = input_layer->width();
  channel_size_     = input_layer->height() * input_layer->width();
  image_size_       = input_layer->channels() * channel_size_;
  mutable_cpu_data_ = input_layer->mutable_cpu_data();

  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

  /* Load the binaryproto mean file. */
  SetMean(mean_file);

  /* Load labels. */
  std::ifstream labels(label_file.c_str());
  CHECK(labels) << "Unable to open labels file " << label_file;
  std::string line;
  while (std::getline(labels, line))
    labels_.push_back(std::string(line));

  caffe::Blob<float>* output_layer = net_->output_blobs()[0];
  CHECK_EQ(labels_.size(), output_layer->channels())
    << "Number of labels is different from the output layer dimension.";
}

static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
  return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<float>& v, int N) {
  std::vector<std::pair<float, int> > pairs;
  for (size_t i = 0; i < v.size(); ++i)
    pairs.push_back(std::make_pair(v[i], i));
  std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

  std::vector<int> result;
  for (int i = 0; i < N; ++i)
    result.push_back(pairs[i].second);
  return result;
}

/* Return the top N predictions. */
std::vector<PredictionList> Classifier::Classify(const std::vector<cv::Mat>& imgs, int N) {
  std::vector<std::vector<float> > output = Predict(imgs);

  N = std::min<int>(labels_.size(), N);

  std::vector<PredictionList> imgs_predictions;
  for (unsigned int idx_image = 0; idx_image < imgs.size(); ++idx_image) {
    std::vector<int> maxN = Argmax(output[idx_image], N);
    std::vector<Prediction> predictions;
    for (int i = 0; i < N; ++i) {
      int idx = maxN[i];
      predictions.push_back(std::make_pair(labels_[idx], output[idx_image][idx]));
    }
    imgs_predictions.push_back(predictions);
  }

  return imgs_predictions;
}

/* Load the mean file in binaryproto format. */
void Classifier::SetMean(const std::string& mean_file) {
  caffe::BlobProto blob_proto;
  ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

  /* Convert from BlobProto to Blob<float> */
  caffe::Blob<float> mean_blob;
  mean_blob.FromProto(blob_proto);
  CHECK_EQ(mean_blob.channels(), num_channels_)
    << "Number of channels of mean file doesn't match input layer.";

  /* The format of the mean file is planar 32-bit float BGR or grayscale. */
  std::vector<cv::Mat> channels;
  float* data = mean_blob.mutable_cpu_data();
  for (int i = 0; i < num_channels_; ++i) {
    /* Extract an individual channel. */
    cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
    channels.push_back(channel);
    data += mean_blob.height() * mean_blob.width();
  }

  /* Merge the separate channels into a single image. */
  cv::Mat mean;
  cv::merge(channels, mean);

  /* Compute the global mean pixel value and create a mean image
   * filled with this value. */
  cv::Scalar channel_mean = cv::mean(mean);
  mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}

std::vector<std::vector<float> > Classifier::ForwardBatch() {

  std::cout << "Classifier::ForwardBatch()..." << std::endl;
  net_->Forward();
  std::cout << "Classifier::ForwardBatch() done" << std::endl;

  // take a long time...

  caffe::Blob<float>* output_layer = net_->output_blobs()[0];
  const float* begin = output_layer->cpu_data();
  std::vector<std::vector<float> > result;
  for (int i = 0; i < batch_size_; ++i) {
    std::vector<float> predict_outputs(
        begin + output_layer->channels() * i,
        begin + output_layer->channels() * (i + 1)
    );
    result.push_back(predict_outputs);
  }
  return result;
}

std::vector<std::vector<float> > Classifier::Predict(const std::vector<cv::Mat>& imgs) {
  caffe::Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(imgs.size(), num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels, imgs.size());

  Preprocess(imgs, &input_channels);

  net_->Forward();

  /* Copy the output layer to a std::vector */
  caffe::Blob<float>* output_layer = net_->output_blobs()[0];
  const float* begin = output_layer->cpu_data();
  //const float* end = begin + imgs.size() * output_layer->channels();

  std::vector<std::vector<float> > result;
  for (unsigned int i = 0; i < imgs.size(); ++i) {
    std::vector<float> predict_outputs(
        begin + output_layer->channels() * i,
        begin + output_layer->channels() * (i + 1)
    );
    result.push_back(predict_outputs);
  }
  return result;
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels, int num_images) {
  caffe::Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels() * num_images; ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void Classifier::Preprocess(const std::vector<cv::Mat>& imgs,
                            std::vector<cv::Mat>* input_channels) {
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (imgs[0].channels() == 3 && num_channels_ == 1)
    cv::cvtColor(imgs[0], sample, cv::COLOR_BGR2GRAY);
  else if (imgs[0].channels() == 4 && num_channels_ == 1)
    cv::cvtColor(imgs[0], sample, cv::COLOR_BGRA2GRAY);
  else if (imgs[0].channels() == 4 && num_channels_ == 3)
    cv::cvtColor(imgs[0], sample, cv::COLOR_BGRA2BGR);
  else if (imgs[0].channels() == 1 && num_channels_ == 3)
    cv::cvtColor(imgs[0], sample, cv::COLOR_GRAY2BGR);
  else
    sample = imgs[0];

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  cv::Mat sample_normalized;
  cv::subtract(sample_float, mean_, sample_normalized);

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  for (unsigned int i = 0; i < imgs.size(); ++i) {
    //cv::split(sample_normalized, *input_channels);
    std::vector<cv::Mat> channels;
    cv::split(sample_normalized, channels);
    for (unsigned int j = 0; j < channels.size(); ++j) {
      channels[j].copyTo((*input_channels)[i * num_channels_ + j]);
    }
  }

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}

void split_to_caffe(uint8_t* source, float* target, int num_img, int image_size, int channel_size, int height, int width) {
  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      int idx_on_frame = w * height + h;
      int idx_on_caffe = num_img * image_size
                       //+ num_channels_ * channel_size_
                       //+ channel * 0 // we dont computer channel offset here, unroll with three line below
                       + h * width
                       + w;

      // on caffe blob, stride of channel is height * width
      // on AVFrame, stride of channel is one pixel (dependends on the format, int or float or ...)
      //
      // on caffe blob, type is 32 bit float
      // on AVFrame, type is ... according to the sws setting, here we use RGB24, each color 8 bit.
      target[idx_on_caffe + channel_size * 0] = source[idx_on_frame + 0];
      target[idx_on_caffe + channel_size * 1] = source[idx_on_frame + 1];
      target[idx_on_caffe + channel_size * 2] = source[idx_on_frame + 2];
    }
  }
}

bool Classifier::PushImage(AVFrame* frame) {
  for (int h = 0; h < height_; ++h) {
    for (int w = 0; w < width_; ++w) {
      int idx_on_frame = (h * frame->width + w) * num_channels_;
      int idx_on_caffe = num_pushed_image_ * image_size_
                       //+ num_channels_ * channel_size_
                       + num_channels_ * 0 // we dont computer channel offset here, unroll with three line below
                       + h * width_
                       + w;

      // on caffe blob, stride of channel is height * width
      // on AVFrame, stride of channel is one pixel (dependends on the format, int or float or ...)
      //
      // on caffe blob, type is 32 bit float
      // on AVFrame, type is ... according to the sws setting, here we use RGB24, each color 8 bit.
      mutable_cpu_data_[idx_on_caffe + channel_size_ * 0] = static_cast<float>(frame->data[0][idx_on_frame + 0]);
      mutable_cpu_data_[idx_on_caffe + channel_size_ * 1] = static_cast<float>(frame->data[0][idx_on_frame + 1]);
      mutable_cpu_data_[idx_on_caffe + channel_size_ * 2] = static_cast<float>(frame->data[0][idx_on_frame + 2]);
    }
  }

  ++num_pushed_image_;
  bool ok_to_fire = batch_size_ == num_pushed_image_;
  if (num_pushed_image_ == batch_size_) {
    num_pushed_image_ = 0;
  }
  return ok_to_fire;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
// main function
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv) {
  // check args
  if (argc == 3 && std::string("benchmark") == argv[1]) {
    //////////////////////////////////////////////////////////////////////////////
    // bench split speed
    //////////////////////////////////////////////////////////////////////////////

    int height = 480;
    int width = 856;
    float cpu[3 * height * width];
    uint8_t img[3 * width * height];

    for (int i = 0; i < 100; ++i) {
      split_to_caffe(img, cpu, 0, 3 * height * width, height * width, height, width);
      std::cout << i << std::endl;
    }

    //////////////////////////////////////////////////////////////////////////////
    // bench decoder speed
    //////////////////////////////////////////////////////////////////////////////

    //std::string video_file   = argv[2];
    //av_register_all();
    //// a video decoder
    //VideoDecoder video_decoder(video_file);
    //int count_frame = 0;
    //while(video_decoder.DecodeOneFrame(count_frame % 30 != 0)) {
    //  //video_decoder.GetFrame();
    //  ++count_frame;
    //  if (count_frame % 1000 == 0) {
    //    std::cout << count_frame << std::endl;
    //  }
    //}

    return 0;
  }
  else if (argc == 6) {
    // init google log
    ::google::InitGoogleLogging(argv[0]);
    // init ffmpeg
    av_register_all();

    // create caffe classifier
    std::string model_file   = argv[1];
    std::string trained_file = argv[2];
    std::string mean_file    = argv[3];
    std::string label_file   = argv[4];
    std::string video_file   = argv[5];

    // a video decoder
    std::cout << "Opening video: " << video_file << std::endl;
    VideoDecoder video_decoder(video_file);
    std::cout << "Open video done" << std::endl;

    // a classifier
    std::cout << "Opening classifier: " << model_file << std::endl;
    Classifier classifier(model_file, trained_file, mean_file, label_file);
    std::cout << "Open classifier done" << std::endl;

    int count_frame = 0;
    while(video_decoder.DecodeOneFrame(count_frame % 30 != 0)) {
      AVFrame* frameRGB = video_decoder.GetFrame();
      if (classifier.PushImage(frameRGB)) {
        classifier.ForwardBatch();
        std::cout << count_frame << std::endl;
      }
      ++count_frame;
      if (count_frame % 1000 == 0) {
        std::cout << "============= " << count_frame << std::endl;
      }
    }
  }
  else {
    std::cerr << "Usage: " << argv[0]
              << " deploy.prototxt network.caffemodel"
              << " mean.binaryproto labels.txt video.avi" << std::endl;
    return 1;
  }
}
