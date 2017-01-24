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

typedef std::pair<std::string, float> Prediction;
typedef std::vector<Prediction> PredictionList;

//////////////////////////////////////////////////////////////////////////////
// FFmpeg part
//////////////////////////////////////////////////////////////////////////////

class VideoDecoder {
 public:
  VideoDecoder(const std::string& video_file);
  ~VideoDecoder();

  /// @param ignoreFrame if true, the frame will still be decoded, but will not
  /// convert the frame to RGB and not write the result frame out. useful when
  /// you want to ignore some frame.
  ///
  /// @return the decoded frame, NULL if end-of-video. Note that, the resourec
  /// of the returned frame is owned by the VideoDecoder, you should NOT free
  /// the AVFrame by yourself.
  ///
  /// @note The return frame format is (width x height x channel), channel order is RGB.
  /// data type of each channle is uint8_t.
  AVFrame* DecodeOneFrame(bool ignoreFrame);

  // get video height
  int Height() { return pCodecCtx->height; }
  // get video width
  int Width() { return pCodecCtx->width; }

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
  numBytes=avpicture_get_size(AV_PIX_FMT_BGR24, pCodecCtx->width, pCodecCtx->height);
  pBuffer=(uint8_t *)av_malloc(numBytes*sizeof(uint8_t));

  // Assign appropriate parts of buffer to image planes in pFrameRGB
  // Note that pFrameRGB is an AVFrame, but AVFrame is a superset of AVPicture
  avpicture_fill((AVPicture *)pFrameRGB, pBuffer, AV_PIX_FMT_BGR24, pCodecCtx->width, pCodecCtx->height);
  pFrameRGB->width  = pCodecCtx->width;
  pFrameRGB->height = pCodecCtx->height;

  // initialize SWS context for software scaling
  sws_ctx = sws_getContext(pCodecCtx->width, pCodecCtx->height, pCodecCtx->pix_fmt, pCodecCtx->width,
    pCodecCtx->height, AV_PIX_FMT_BGR24, SWS_BILINEAR, NULL, NULL, NULL);
}

VideoDecoder::~VideoDecoder() {
  av_free(pBuffer);
  av_free(pFrameRGB);
  av_free(pFrame);

  avcodec_close(pCodecCtx);
  avformat_close_input(&pFormatCtx);
}

AVFrame* VideoDecoder::DecodeOneFrame(bool ignoreFrame) {
  AVPacket packet;
  while (av_read_frame(pFormatCtx, &packet) >= 0) {
    // Is this a packet from the video stream?
    if (packet.stream_index != firstVideoStreamIndex)
      continue;

    // Decode this packet
    int gotPicturePtr;
    avcodec_decode_video2(pCodecCtx, pFrame, &gotPicturePtr, &packet);
    // Check if this packet is a vidoe frame
    if(!gotPicturePtr)
      continue;

    // If we want to skip this frame, free the packet and return
    if (ignoreFrame) {
      av_free_packet(&packet);
      return pFrameRGB;
    }

    // Convert the image from its native format to RGB
    sws_scale(sws_ctx, (uint8_t const * const *)pFrame->data, pFrame->linesize, 0, pCodecCtx->height, pFrameRGB->data, pFrameRGB->linesize);
    return pFrameRGB;
  }
  av_free_packet(&packet);
  return NULL;
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

  // return true if this batch is full and ready to fire, false if not
  bool PushImage(AVFrame* f);
  std::vector<std::vector<float> > ForwardBatch();

 private:
  void SetMean(const std::string& mean_file);

 private:
  boost::shared_ptr<caffe::Net<float> > net_;
  cv::Size input_geometry_;
  int height_;
  int width_;
  int num_channels_;
  int batch_size_;
  caffe::Blob<float> mean_blob_;
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

/* Load the mean file in binaryproto format. */
void Classifier::SetMean(const std::string& mean_file) {
  caffe::BlobProto blob_proto;
  ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

  /* Convert from BlobProto to Blob<float> */
  mean_blob_.FromProto(blob_proto);
  CHECK_EQ(mean_blob_.channels(), num_channels_)
    << "Number of channels of mean file doesn't match input layer.";
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

bool Classifier::PushImage(AVFrame* frame) {
  for (int h = 0; h < height_; ++h) {
    for (int w = 0; w < width_; ++w) {
      int idx_on_frame = (h * frame->width + w) * num_channels_;
      int idx_on_caffe = num_pushed_image_ * image_size_
                       + num_channels_ * 0 // we dont computer channel offset here, unroll with three line below
                       + h * width_
                       + w;

      // on caffe blob, stride of channel is height * width
      // on AVFrame, stride of channel is one pixel (dependends on the format, int or float or ...)
      //
      // on caffe blob, type is 32 bit float
      // on AVFrame, type is ... according to the sws setting, here we use RGB24, each color 8 bit.
      mutable_cpu_data_[idx_on_caffe + channel_size_ * 0] = static_cast<float>(frame->data[0][idx_on_frame + 0]) - mean_blob_.data_at(0, 0, h, w);
      mutable_cpu_data_[idx_on_caffe + channel_size_ * 1] = static_cast<float>(frame->data[0][idx_on_frame + 1]) - mean_blob_.data_at(0, 1, h, w);
      mutable_cpu_data_[idx_on_caffe + channel_size_ * 2] = static_cast<float>(frame->data[0][idx_on_frame + 2]) - mean_blob_.data_at(0, 2, h, w);
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
    // bench decoder speed
    //////////////////////////////////////////////////////////////////////////////

    std::string video_file   = argv[2];
    av_register_all();
    // a video decoder
    VideoDecoder video_decoder(video_file);
    int count_frame = 0;
    while(AVFrame* frame = video_decoder.DecodeOneFrame(count_frame % 30 != 0)) {
      (void)frame;
      ++count_frame;
      if (count_frame % 1000 == 0) {
        std::cout << count_frame << std::endl;
      }
    }

    return 0;
  }
  else if (argc == 6) {
    // init google log
    ::google::InitGoogleLogging(argv[0]);
    // init ffmpeg
    av_register_all();

    // read args
    std::string model_file   = argv[1];
    std::string trained_file = argv[2];
    std::string mean_file    = argv[3];
    std::string label_file   = argv[4];
    std::string video_file   = argv[5];

    // create a video decoder
    std::cout << "Opening video: " << video_file << std::endl;
    VideoDecoder video_decoder(video_file);
    std::cout << "Open video done" << std::endl;

    // create a classifier
    std::cout << "Opening classifier: " << model_file << std::endl;
    Classifier classifier(model_file, trained_file, mean_file, label_file);
    std::cout << "Open classifier done" << std::endl;

    // decode each frame
    int count_frame = 0;
    int count_pushed_frame = 0;
    while(AVFrame* frameRGB = video_decoder.DecodeOneFrame(count_frame % 30 != 0)) {
      bool ok_to_fire = classifier.PushImage(frameRGB);
      ++count_pushed_frame;
      if (ok_to_fire) {
        std::vector<std::vector<float> > predict_results = classifier.ForwardBatch();
      }
      ++count_frame;
      if (count_frame % 1000 == 0) {
        std::cout << "Number of decoded frame: " << count_frame << std::endl;
      }
    }
  }
  else {
    std::cerr << "Usage: " << argv[0]
              << " deploy.prototxt network.caffemodel"
              << " mean.binaryproto labels.txt video.mp4" << std::endl;
    return 1;
  }
}
