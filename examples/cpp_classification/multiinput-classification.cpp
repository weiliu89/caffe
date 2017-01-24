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

/* Pair (label, confidence) representing a prediction. */
typedef std::pair<std::string, float> Prediction;
typedef std::vector<Prediction> PredictionList;

//////////////////////////////////////////////////////////////////////////////
// FFmpeg part
//////////////////////////////////////////////////////////////////////////////

void SaveFrame(AVFrame *pFrame, int width, int height, int iFrame) {
  FILE *pFile;
  char szFilename[32];
  int  y;
  
  // Open file
  sprintf(szFilename, "frame%06d.ppm", iFrame);
  pFile=fopen(szFilename, "wb");
  if(pFile==NULL)
    return;
  
  // Write header
  fprintf(pFile, "P6\n%d %d\n255\n", width, height);
  
  // Write pixel data
  for(y=0; y<height; y++)
    fwrite(pFrame->data[0]+y*pFrame->linesize[0], 1, width*3, pFile);
  
  // Close file
  fclose(pFile);
}

int read_video_and_extract_frames(int argc, char *argv[]) {
  /*int dumpFrames=argv[2];*/

  if (argc<2) {
    fprintf(stderr, "Did not provide input video\n");
    return -1;
  }

  clock_t t;
  t = clock();
  av_register_all();
  AVFormatContext *pFormatCtx = NULL;
  // Open video file
  if(avformat_open_input(&pFormatCtx, argv[1], NULL, NULL)!=0) {
    fprintf(stderr, " Couldn't open file!\n");
    return -1;
  }
  // Retrieve stream information
  if(avformat_find_stream_info(pFormatCtx, NULL)<0) {
    fprintf(stderr, "Couldn't find stream information!\n");
    return -1;
  }
  // Dump information about file onto standard error
  av_dump_format(pFormatCtx, 0, argv[1], 0);

  AVCodec *pCodec = NULL;
  AVCodecContext *pCodecCtx = NULL;

  // Find the first video stream
  int videoStream=-1;
  for(unsigned int i=0; i<pFormatCtx->nb_streams; i++) {
    if(pFormatCtx->streams[i]->codec->codec_type==AVMEDIA_TYPE_VIDEO) {
      videoStream=i;
      break;
    }
  }
  if(videoStream==-1) {
    fprintf(stderr, "Didn't find a video stream!\n");
    return -1;
  }

  // Get a pointer to the codec context for the video stream
  pCodecCtx=pFormatCtx->streams[videoStream]->codec;

  // Find the decoder for the video stream
  pCodec=avcodec_find_decoder(pCodecCtx->codec_id);
  if(!pCodec) {
    fprintf(stderr, "Unsupported codec!\n");
    return -1;
  }

  // Open codec
  if(avcodec_open2(pCodecCtx, pCodec, NULL) < 0) {
    fprintf(stderr, "Could not open codec\n");
    return -1;
  }

  // Allocate video frame
  AVFrame* pFrame=avcodec_alloc_frame();
  if(pFrame==NULL) {
    fprintf(stderr, "Could not allocate video frame\n");
    return -1;
  }

  // Allocate an AVFrame structure
  AVFrame* pFrameRGB=avcodec_alloc_frame();
  if(pFrameRGB==NULL) {
    fprintf(stderr, "Could not allocate output RGB video frame\n");
    return -1;
  }

  uint8_t *buffer = NULL;
  int numBytes;
  // Determine required buffer size and allocate buffer
  numBytes=avpicture_get_size(AV_PIX_FMT_RGB24, pCodecCtx->width, pCodecCtx->height);
  buffer=(uint8_t *)av_malloc(numBytes*sizeof(uint8_t));

  // Assign appropriate parts of buffer to image planes in pFrameRGB
  // Note that pFrameRGB is an AVFrame, but AVFrame is a superset of AVPicture
  avpicture_fill((AVPicture *)pFrameRGB, buffer, AV_PIX_FMT_RGB24, pCodecCtx->width, pCodecCtx->height);

  // initialize SWS context for software scaling
  struct SwsContext *sws_ctx = NULL;
  sws_ctx = sws_getContext(pCodecCtx->width, pCodecCtx->height, pCodecCtx->pix_fmt, pCodecCtx->width,
    pCodecCtx->height, AV_PIX_FMT_RGB24, SWS_BILINEAR, NULL, NULL, NULL);

  unsigned int i=0;

  AVPacket packet;
  while(av_read_frame(pFormatCtx, &packet)>=0) {
    // Is this a packet from the video stream?
    if(packet.stream_index==videoStream) {
      // Decode video frame
      int gotPicturePtr;
      avcodec_decode_video2(pCodecCtx, pFrame, &gotPicturePtr, &packet);
    
      // Did we get a video frame?
      if(gotPicturePtr) {
        // Convert the image from its native format to RGB
        sws_scale(sws_ctx, (uint8_t const * const *)pFrame->data, pFrame->linesize, 0, pCodecCtx->height, pFrameRGB->data, pFrameRGB->linesize);
	
        // Save the frame to disk
        ++i;
        /*if (dumpFrames==1)*/
          //SaveFrame(pFrameRGB, pCodecCtx->width, pCodecCtx->height, i);
      }
    }
    
    // Free the packet that was allocated by av_read_frame
    av_free_packet(&packet);
  }

  // Free the RGB image
  av_free(buffer);
  av_free(pFrameRGB);

  // Free the YUV frame
  av_free(pFrame);

  // Close the codecs
  avcodec_close(pCodecCtx);
  //avcodec_close(pCodecCtxOrig);

  // Close the video file
  avformat_close_input(&pFormatCtx);
  t = clock()-t;
  printf ("Total execution time: %f seconds.\n",((float)t)/CLOCKS_PER_SEC);
  return 0;
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

 private:
  void SetMean(const std::string& mean_file);

  std::vector<std::vector<float> > Predict(const std::vector<cv::Mat>& img);

  void WrapInputLayer(std::vector<cv::Mat>* input_channels, int num_images);

  void Preprocess(const std::vector<cv::Mat>& img,
                  std::vector<cv::Mat>* input_channels);

 private:
  boost::shared_ptr<caffe::Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  cv::Mat mean_;
  std::vector<std::string> labels_;
};

Classifier::Classifier(const std::string& model_file,
                       const std::string& trained_file,
                       const std::string& mean_file,
                       const std::string& label_file) {
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
  num_channels_ = input_layer->channels();
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

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
// main function
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv) {
  // check args
  if (argc < 6) {
    std::cerr << "Usage: " << argv[0]
              << " deploy.prototxt network.caffemodel"
              << " mean.binaryproto labels.txt img.jpg [more img.jpg ...]" << std::endl;
    return 1;
  }

  // init google log
  ::google::InitGoogleLogging(argv[0]);

  // create caffe classifier
  std::string model_file   = argv[1];
  std::string trained_file = argv[2];
  std::string mean_file    = argv[3];
  std::string label_file   = argv[4];
  Classifier classifier(model_file, trained_file, mean_file, label_file);

  ////////////////////////////////////////////////////////////////////////////
  // BEGIN use ffmpeg to open video
  ////////////////////////////////////////////////////////////////////////////

  ////////////////////////////////////////////////////////////////////////////
  // END use ffmpeg to open video
  ////////////////////////////////////////////////////////////////////////////

  // read image file names from argv
  std::vector<std::string> files;
  for (int idx_files = 5; idx_files < argc; ++idx_files) {
    files.push_back(argv[idx_files]);
  }

  // open images & push to vector
  std::vector<cv::Mat> imgs;
  for (unsigned int i = 0; i < files.size(); ++i) {
    cv::Mat img = cv::imread(files[i], -1);
    CHECK(!img.empty()) << "Unable to decode image " << files[i];
    imgs.push_back(img);
  }

  // classify
  std::vector<PredictionList> prediction_lists = classifier.Classify(imgs);

  // print result
  for (unsigned int idx_image = 0; idx_image < imgs.size(); ++idx_image) {
    PredictionList& predictions = prediction_lists[idx_image];
    std::cout << "---------- Prediction for "
              << files[idx_image] << " ----------" << std::endl;
    for (size_t i = 0; i < predictions.size(); ++i) {
      Prediction p = predictions[i];
      std::cout << std::fixed << std::setprecision(4) << p.second << " - \""
                << p.first << "\"" << std::endl;
    }
  }
}
