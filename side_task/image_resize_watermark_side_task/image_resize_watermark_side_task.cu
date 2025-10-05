#include <argparse/argparse.hpp>
#include <iostream>
#include "task.h"
#include <thread>
#include <csignal>
#include <grpcpp/security/server_credentials.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <atomic>

#include <fstream>
#include <cassert>
#include "imageResizeWatermark.h"
#include <filesystem>


//#define OPTIMIZED_HUFFMAN
#define ALPHA_BLEND 50

// *****************************************************************************
// Decode, Resize and Encoder function
// -----------------------------------------------------------------------------


class ImageResizeWatermarkSideTask final : public BubbleBanditTask {
 private:
  const std::string input_dir_;
  const std::string output_dir_;
  const int jpeg_quality_;
  const int resize_width_;
  const int resize_height_;
  std::vector<std::string> input_files_;
  const int image_num_;
  std::atomic<int> image_iter_;
  const int max_iter_;
  unsigned char *pBufferW = nullptr;
  unsigned char *pBuffer = nullptr;
  unsigned char *pResizeBuffer = nullptr;
  unsigned char *pResizeBufferW = nullptr;
// *****************************************************************************
// nvJPEG handles and parameters
// -----------------------------------------------------------------------------
  nvjpegBackend_t impl = NVJPEG_BACKEND_GPU_HYBRID; //NVJPEG_BACKEND_DEFAULT;
  nvjpegHandle_t nvjpeg_handle;
  nvjpegJpegStream_t nvjpeg_jpeg_stream;
  nvjpegDecodeParams_t nvjpeg_decode_params;
  nvjpegJpegState_t nvjpeg_decoder_state;
  nvjpegEncoderParams_t nvjpeg_encode_params;
  nvjpegEncoderState_t nvjpeg_encoder_state;
  nvjpegJpegEncoding_t nvjpeg_encoding;

  int decodeResizeEncodeOneImage(std::string sImagePath,
                                 std::string sOutputPath,
                                 double &time,
                                 int resizeWidth,
                                 int resizeHeight,
                                 int resize_quality) {
    // Decode, Encoder format
    nvjpegOutputFormat_t oformat = NVJPEG_OUTPUT_BGR;
    nvjpegInputFormat_t iformat = NVJPEG_INPUT_BGR;

    // timing for resize
    time = 0.;
    float resize_time = 0.;
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Image reading section
    // Get the file name, without extension.
    // This will be used to rename the output file.
    size_t position = sImagePath.rfind("/");
    std::string
        sFileName = (std::string::npos == position) ? sImagePath : sImagePath.substr(position + 1, sImagePath.size());
    position = sFileName.rfind(".");
    sFileName = (std::string::npos == position) ? sFileName : sFileName.substr(0, position);

#ifndef _WIN64
    position = sFileName.rfind("/");
    sFileName = (std::string::npos == position) ? sFileName : sFileName.substr(position + 1, sFileName.length());
#else
    position = sFileName.rfind("\\");
    sFileName = (std::string::npos == position) ? sFileName : sFileName.substr(position+1, sFileName.length());
#endif

    // Read an watermark image from disk.
    std::ifstream oInputStreamW("NVLogo.jpg", std::ios::in | std::ios::binary | std::ios::ate);
    if (!(oInputStreamW.is_open())) {
      std::cerr << "Cannot open watermark image: " << sImagePath << std::endl;
      return EXIT_FAILURE;
    }

    // Get the size.
    std::streamsize nSizeW = oInputStreamW.tellg();
    oInputStreamW.seekg(0, std::ios::beg);
    // Image buffers.
//    unsigned char *pBufferW = NULL;
    // device image buffers.
    nvjpegImage_t imgDescW;
    size_t pitchDescW;
    NppiSize srcSizeW;

    std::vector<char> vBufferW(nSizeW);
    if (oInputStreamW.read(vBufferW.data(), nSizeW)) {
      unsigned char *dpImageW = (unsigned char *) vBufferW.data();
      // Retrieve the componenet and size info.
      int nComponent = 0;
      nvjpegChromaSubsampling_t subsampling;
      int widths[NVJPEG_MAX_COMPONENT];
      int heights[NVJPEG_MAX_COMPONENT];
      int nReturnCode = 0;
      if (NVJPEG_STATUS_SUCCESS
          != nvjpegGetImageInfo(nvjpeg_handle, dpImageW, nSizeW, &nComponent, &subsampling, widths, heights)) {
        std::cerr << "Error decoding JPEG header: " << sImagePath << std::endl;
        return EXIT_FAILURE;
      }

      srcSizeW = {(int) widths[0], (int) heights[0]};

      if (is_interleaved(oformat)) {
        pitchDescW = NVJPEG_MAX_COMPONENT * widths[0];
      } else {
        pitchDescW = 3 * widths[0];
      }

      cudaError_t eCopy = cudaMalloc(&pBufferW, pitchDescW * heights[0]);
      if (cudaSuccess != eCopy) {
        std::cerr << "cudaMalloc failed : " << cudaGetErrorString(eCopy) << std::endl;
        return EXIT_FAILURE;
      }

      imgDescW.channel[0] = pBufferW;
      imgDescW.channel[1] = pBufferW + widths[0] * heights[0];
      imgDescW.channel[2] = pBufferW + widths[0] * heights[0] * 2;
      imgDescW.pitch[0] = (unsigned int) (is_interleaved(oformat) ? widths[0] * NVJPEG_MAX_COMPONENT : widths[0]);
      imgDescW.pitch[1] = (unsigned int) widths[0];
      imgDescW.pitch[2] = (unsigned int) widths[0];

      if (is_interleaved(oformat)) {
        imgDescW.channel[3] = pBufferW + widths[0] * heights[0] * 3;
        imgDescW.pitch[3] = (unsigned int) widths[0];
      }

      // decode by stages
      nReturnCode = nvjpegDecode(nvjpeg_handle, nvjpeg_decoder_state, dpImageW, nSizeW, oformat, &imgDescW, NULL);
      if (nReturnCode != 0) {
        std::cerr << "Error in nvjpegDecode." << nReturnCode << std::endl;
        return EXIT_FAILURE;
      }
    }




    // Read an image from disk.
    std::ifstream oInputStream(sImagePath.c_str(), std::ios::in | std::ios::binary | std::ios::ate);
    if (!(oInputStream.is_open())) {
      std::cerr << "Cannot open image: " << sImagePath << std::endl;
      return EXIT_FAILURE;
    }

    // Get the size.
    std::streamsize nSize = oInputStream.tellg();
    oInputStream.seekg(0, std::ios::beg);

    // Image buffers.
//    unsigned char *pBuffer = NULL;
//    unsigned char *pResizeBuffer = NULL;
//    unsigned char *pResizeBufferW = NULL;

    std::vector<char> vBuffer(nSize);
    if (oInputStream.read(vBuffer.data(), nSize)) {
      unsigned char *dpImage = (unsigned char *) vBuffer.data();

      // Retrieve the componenet and size info.
      int nComponent = 0;
      nvjpegChromaSubsampling_t subsampling;
      int widths[NVJPEG_MAX_COMPONENT];
      int heights[NVJPEG_MAX_COMPONENT];
      int nReturnCode = 0;
      if (NVJPEG_STATUS_SUCCESS
          != nvjpegGetImageInfo(nvjpeg_handle, dpImage, nSize, &nComponent, &subsampling, widths, heights)) {
        std::cerr << "Error decoding JPEG header: " << sImagePath << std::endl;
        return EXIT_FAILURE;
      }

      if (resizeWidth == 0 || resizeHeight == 0) {
        resizeWidth = widths[0] / 2;
        resizeHeight = heights[0] / 2;
      }

      // image resize
      size_t pitchDesc, pitchResize;
      NppiSize srcSize = {(int) widths[0], (int) heights[0]};
      NppiRect srcRoi = {0, 0, srcSize.width, srcSize.height};
      NppiSize dstSize = {(int) resizeWidth, (int) resizeHeight};
      NppiRect dstRoi = {0, 0, dstSize.width, dstSize.height};
      NppiRect srcRoiW = {0, 0, srcSizeW.width, srcSizeW.height};
      NppStatus st;
      NppStreamContext nppStreamCtx;
      nppStreamCtx.hStream = NULL; // default stream

      // device image buffers.
      nvjpegImage_t imgDesc;
      nvjpegImage_t imgResize;
      nvjpegImage_t imgResizeW;

      if (is_interleaved(oformat)) {
        pitchDesc = NVJPEG_MAX_COMPONENT * widths[0];
        pitchResize = NVJPEG_MAX_COMPONENT * resizeWidth;
      } else {
        pitchDesc = 3 * widths[0];
        pitchResize = 3 * resizeWidth;
      }

      cudaError_t eCopy = cudaMalloc(&pBuffer, pitchDesc * heights[0]);
      if (cudaSuccess != eCopy) {
        std::cerr << "cudaMalloc failed : " << cudaGetErrorString(eCopy) << std::endl;
        return EXIT_FAILURE;
      }
      cudaError_t eCopy1 = cudaMalloc(&pResizeBuffer, pitchResize * resizeHeight);
      if (cudaSuccess != eCopy1) {
        std::cerr << "cudaMalloc failed : " << cudaGetErrorString(eCopy1) << std::endl;
        return EXIT_FAILURE;
      }
      cudaError_t eCopy2 = cudaMalloc(&pResizeBufferW, pitchResize * resizeHeight);
      if (cudaSuccess != eCopy2) {
        std::cerr << "cudaMalloc failed : " << cudaGetErrorString(eCopy2) << std::endl;
        return EXIT_FAILURE;
      }

      imgDesc.channel[0] = pBuffer;
      imgDesc.channel[1] = pBuffer + widths[0] * heights[0];
      imgDesc.channel[2] = pBuffer + widths[0] * heights[0] * 2;
      imgDesc.pitch[0] = (unsigned int) (is_interleaved(oformat) ? widths[0] * NVJPEG_MAX_COMPONENT : widths[0]);
      imgDesc.pitch[1] = (unsigned int) widths[0];
      imgDesc.pitch[2] = (unsigned int) widths[0];

      imgResize.channel[0] = pResizeBuffer;
      imgResize.channel[1] = pResizeBuffer + resizeWidth * resizeHeight;
      imgResize.channel[2] = pResizeBuffer + resizeWidth * resizeHeight * 2;
      imgResize.pitch[0] = (unsigned int) (is_interleaved(oformat) ? resizeWidth * NVJPEG_MAX_COMPONENT : resizeWidth);;
      imgResize.pitch[1] = (unsigned int) resizeWidth;
      imgResize.pitch[2] = (unsigned int) resizeWidth;

      imgResizeW.channel[0] = pResizeBufferW;
      imgResizeW.channel[1] = pResizeBufferW + resizeWidth * resizeHeight;
      imgResizeW.channel[2] = pResizeBufferW + resizeWidth * resizeHeight * 2;
      imgResizeW.pitch[0] =
          (unsigned int) (is_interleaved(oformat) ? resizeWidth * NVJPEG_MAX_COMPONENT : resizeWidth);;
      imgResizeW.pitch[1] = (unsigned int) resizeWidth;
      imgResizeW.pitch[2] = (unsigned int) resizeWidth;

      if (is_interleaved(oformat)) {
        imgDesc.channel[3] = pBuffer + widths[0] * heights[0] * 3;
        imgDesc.pitch[3] = (unsigned int) widths[0];
        imgResize.channel[3] = pResizeBuffer + resizeWidth * resizeHeight * 3;
        imgResize.pitch[3] = (unsigned int) resizeWidth;
        imgResizeW.channel[3] = pResizeBufferW + resizeWidth * resizeHeight * 3;
        imgResizeW.pitch[3] = (unsigned int) resizeWidth;
      }

      // nvJPEG encoder parameter setting
      CHECK_NVJPEG(nvjpegEncoderParamsSetQuality(nvjpeg_encode_params, resize_quality, NULL));

#ifdef OPTIMIZED_HUFFMAN  // Optimized Huffman
      CHECK_NVJPEG(nvjpegEncoderParamsSetOptimizedHuffman(nvjpeg_encode_params, 1, NULL));
#endif
      CHECK_NVJPEG(nvjpegEncoderParamsSetSamplingFactors(nvjpeg_encode_params, subsampling, NULL));


      // Timing start
      CHECK_CUDA(cudaEventRecord(start, 0));

      //parse image save metadata in jpegStream structure
      CHECK_NVJPEG(nvjpegJpegStreamParse(nvjpeg_handle, dpImage, nSize, 1, 0, nvjpeg_jpeg_stream));

      // decode by stages
      nReturnCode = nvjpegDecode(nvjpeg_handle, nvjpeg_decoder_state, dpImage, nSize, oformat, &imgDesc, NULL);
      if (nReturnCode != 0) {
        std::cerr << "Error in nvjpegDecode." << nReturnCode << std::endl;
        return EXIT_FAILURE;
      }

      // image resize
      /* Note: this is the simplest resizing function from NPP. */
      if (is_interleaved(oformat)) {
        st = nppiResize_8u_C3R_Ctx(imgDesc.channel[0],
                                   imgDesc.pitch[0],
                                   srcSize,
                                   srcRoi,
                                   imgResize.channel[0],
                                   imgResize.pitch[0],
                                   dstSize,
                                   dstRoi,
                                   NPPI_INTER_LANCZOS,
                                   nppStreamCtx);

        st = nppiResize_8u_C3R_Ctx(imgDescW.channel[0],
                                   imgDescW.pitch[0],
                                   srcSizeW,
                                   srcRoiW,
                                   imgResizeW.channel[0],
                                   imgResizeW.pitch[0],
                                   dstSize,
                                   dstRoi,
                                   NPPI_INTER_LANCZOS,
                                   nppStreamCtx);

        // Alpha Blending watermarking
        st = nppiAlphaCompC_8u_C3R_Ctx(imgResize.channel[0],
                                       imgResize.pitch[0],
                                       255,
                                       imgResizeW.channel[0],
                                       imgResizeW.pitch[0],
                                       ALPHA_BLEND,
                                       imgResize.channel[0],
                                       imgResize.pitch[0],
                                       dstSize,
                                       NPPI_OP_ALPHA_PLUS,
                                       nppStreamCtx);

      } else {
        st = nppiResize_8u_C1R_Ctx(imgDesc.channel[0],
                                   imgDesc.pitch[0],
                                   srcSize,
                                   srcRoi,
                                   imgResize.channel[0],
                                   imgResize.pitch[0],
                                   dstSize,
                                   dstRoi,
                                   NPPI_INTER_LANCZOS,
                                   nppStreamCtx);
        st = nppiResize_8u_C1R_Ctx(imgDesc.channel[1],
                                   imgDesc.pitch[1],
                                   srcSize,
                                   srcRoi,
                                   imgResize.channel[1],
                                   imgResize.pitch[1],
                                   dstSize,
                                   dstRoi,
                                   NPPI_INTER_LANCZOS,
                                   nppStreamCtx);
        st = nppiResize_8u_C1R_Ctx(imgDesc.channel[2],
                                   imgDesc.pitch[2],
                                   srcSize,
                                   srcRoi,
                                   imgResize.channel[2],
                                   imgResize.pitch[2],
                                   dstSize,
                                   dstRoi,
                                   NPPI_INTER_LANCZOS,
                                   nppStreamCtx);

        st = nppiResize_8u_C1R_Ctx(imgDescW.channel[0],
                                   imgDescW.pitch[0],
                                   srcSizeW,
                                   srcRoiW,
                                   imgResizeW.channel[0],
                                   imgResizeW.pitch[0],
                                   dstSize,
                                   dstRoi,
                                   NPPI_INTER_LANCZOS,
                                   nppStreamCtx);
        st = nppiResize_8u_C1R_Ctx(imgDescW.channel[1],
                                   imgDescW.pitch[1],
                                   srcSizeW,
                                   srcRoiW,
                                   imgResizeW.channel[1],
                                   imgResizeW.pitch[1],
                                   dstSize,
                                   dstRoi,
                                   NPPI_INTER_LANCZOS,
                                   nppStreamCtx);
        st = nppiResize_8u_C1R_Ctx(imgDescW.channel[2],
                                   imgDescW.pitch[2],
                                   srcSizeW,
                                   srcRoiW,
                                   imgResizeW.channel[2],
                                   imgResizeW.pitch[2],
                                   dstSize,
                                   dstRoi,
                                   NPPI_INTER_LANCZOS,
                                   nppStreamCtx);

        // Alpha Blending watermarking
        st = nppiAlphaCompC_8u_C1R_Ctx(imgResize.channel[0],
                                       imgResize.pitch[0],
                                       255,
                                       imgResizeW.channel[0],
                                       imgResizeW.pitch[0],
                                       ALPHA_BLEND,
                                       imgResize.channel[0],
                                       imgResize.pitch[0],
                                       dstSize,
                                       NPPI_OP_ALPHA_PLUS,
                                       nppStreamCtx);

        st = nppiAlphaCompC_8u_C1R_Ctx(imgResize.channel[1],
                                       imgResize.pitch[1],
                                       255,
                                       imgResizeW.channel[1],
                                       imgResizeW.pitch[1],
                                       ALPHA_BLEND,
                                       imgResize.channel[1],
                                       imgResize.pitch[1],
                                       dstSize,
                                       NPPI_OP_ALPHA_PLUS,
                                       nppStreamCtx);

        st = nppiAlphaCompC_8u_C1R_Ctx(imgResize.channel[2],
                                       imgResize.pitch[2],
                                       255,
                                       imgResizeW.channel[2],
                                       imgResizeW.pitch[2],
                                       ALPHA_BLEND,
                                       imgResize.channel[2],
                                       imgResize.pitch[2],
                                       dstSize,
                                       NPPI_OP_ALPHA_PLUS,
                                       nppStreamCtx);

      }

      if (st != NPP_SUCCESS) {
        std::cerr << "NPP resize failed : " << st << std::endl;
        return EXIT_FAILURE;
      }

      // get encoding from the jpeg stream and copy it to the encode parameters
      CHECK_NVJPEG(nvjpegJpegStreamGetJpegEncoding(nvjpeg_jpeg_stream, &nvjpeg_encoding));
      CHECK_NVJPEG(nvjpegEncoderParamsSetEncoding(nvjpeg_encode_params, nvjpeg_encoding, NULL));
      CHECK_NVJPEG(nvjpegEncoderParamsCopyQuantizationTables(nvjpeg_encode_params, nvjpeg_jpeg_stream, NULL));
      CHECK_NVJPEG(nvjpegEncoderParamsCopyHuffmanTables(nvjpeg_encoder_state,
                                                        nvjpeg_encode_params,
                                                        nvjpeg_jpeg_stream,
                                                        NULL));
      CHECK_NVJPEG(nvjpegEncoderParamsCopyMetadata(nvjpeg_encoder_state,
                                                   nvjpeg_encode_params,
                                                   nvjpeg_jpeg_stream,
                                                   NULL));

      // encoding the resize data
      CHECK_NVJPEG(nvjpegEncodeImage(nvjpeg_handle,
                                     nvjpeg_encoder_state,
                                     nvjpeg_encode_params,
                                     &imgResize,
                                     iformat,
                                     dstSize.width,
                                     dstSize.height,
                                     NULL));

      // retrive the encoded bitstream for file writing
      std::vector<unsigned char> obuffer;
      size_t length;
      CHECK_NVJPEG(nvjpegEncodeRetrieveBitstream(
          nvjpeg_handle,
          nvjpeg_encoder_state,
          NULL,
          &length,
          NULL));

      obuffer.resize(length);

      CHECK_NVJPEG(nvjpegEncodeRetrieveBitstream(
          nvjpeg_handle,
          nvjpeg_encoder_state,
          obuffer.data(),
          &length,
          NULL));

      // Timing stop
      CHECK_CUDA(cudaEventRecord(stop, 0));
      CHECK_CUDA(cudaEventSynchronize(stop));

      // file writing
      std::cout << "Resize-width: " << dstSize.width << " Resize-height: " << dstSize.height << std::endl;
      std::string output_filename = sOutputPath + "/" + sFileName + ".jpg";
      char directory[120];
      char mkdir_cmd[256];
      std::string folder = sOutputPath;
      output_filename = folder + "/" + sFileName + ".jpg";
#if !defined(_WIN32)
      sprintf(directory, "%s", folder.c_str());
      sprintf(mkdir_cmd, "mkdir -p %s 2> /dev/null", directory);
#else
      sprintf(directory, "%s", folder.c_str());
        sprintf(mkdir_cmd, "mkdir %s 2> nul", directory);
#endif

      int ret = system(mkdir_cmd);

      std::cout << "Writing JPEG file: " << output_filename << std::endl;
      std::ofstream outputFile(output_filename.c_str(), std::ios::out | std::ios::binary);
      outputFile.write(reinterpret_cast<const char *>(obuffer.data()), static_cast<int>(length));
    }
    // Free memory
    CHECK_CUDA(cudaFree(pBufferW))
    CHECK_CUDA(cudaFree(pBuffer))
    CHECK_CUDA(cudaFree(pResizeBufferW))
    CHECK_CUDA(cudaFree(pResizeBuffer))

    // get timing
    CHECK_CUDA(cudaEventElapsedTime(&resize_time, start, stop));
    time = (double) resize_time;

    return EXIT_SUCCESS;
  }

 public:
  ImageResizeWatermarkSideTask(int task_id,
                               std::string name,
                               std::string device,
                               std::string scheduler_addr,
                               double duration,
                               int profiler_level,
                               std::string input_dir,
                               std::string output_dir,
                               int jpeg_quality,
                               int resize_width,
                               int resize_height,
                               int image_num,
                               int max_iter)
      : BubbleBanditTask(task_id, name, device, scheduler_addr, profiler_level),
        input_dir_(input_dir + "_" + std::to_string(task_id)),
        output_dir_(output_dir + "_" + std::to_string(task_id)),
        jpeg_quality_(jpeg_quality),
        resize_width_(resize_width),
        resize_height_(resize_height),
        image_num_(image_num),
        max_iter_(max_iter) {
    image_iter_ = 0;
    duration_ = duration;
  }

  int processArgs(std::string sInputPath,
                  std::string sOutputPath,
                  int resizeWidth,
                  int resizeHeight,
                  int resize_quality) {

    int error_code = 1;

    double total_time = 0., decode_time = 0.;
    int total_images = 0;

    std::vector<std::string> inputFiles;

    assert(std::filesystem::is_directory(sInputPath));
    try {
      for (const auto &entry : std::filesystem::directory_iterator(sInputPath)) {
        inputFiles.push_back(entry.path().string());
      }
    } catch (const std::filesystem::filesystem_error &err) {
      std::cerr << "Error reading input directory: " << err.what() << std::endl;
      return error_code;
    }

    for (unsigned int i = 0; i < inputFiles.size(); i++) {
      std::string &sFileName = inputFiles[i];
      std::cout << "Processing file: " << sFileName << std::endl;

      int image_error_code =
          decodeResizeEncodeOneImage(sFileName, sOutputPath, decode_time, resizeWidth, resizeHeight, resize_quality);

      if (image_error_code) {
        std::cerr << "Error processing file: " << sFileName << std::endl;
        return image_error_code;
      } else {
        total_images++;
        total_time += decode_time;
      }
    }

    std::cout << "------------------------------------------------------------- " << std::endl;
    std::cout << "Total images resized: " << total_images << std::endl;
    std::cout << "Total time spent on resizing and watermarking: " << total_time << " (ms)" << std::endl;
    std::cout << "Avg time/image: " << total_time / total_images << " (ms)" << std::endl;
    std::cout << "------------------------------------------------------------- " << std::endl;
    return EXIT_SUCCESS;
  }

  auto submitted_to_created() -> void override {
    auto device = std::atoi(&device_.at(5));
    cudaSetDevice(device);
    nvjpegDevAllocator_t dev_allocator = {&dev_malloc, &dev_free};
    CHECK_NVJPEG(nvjpegCreate(impl, &dev_allocator, &nvjpeg_handle));
    CHECK_NVJPEG(nvjpegJpegStateCreate(nvjpeg_handle, &nvjpeg_decoder_state));

    // create bitstream object
    CHECK_NVJPEG(nvjpegJpegStreamCreate(nvjpeg_handle, &nvjpeg_jpeg_stream));
    CHECK_NVJPEG(nvjpegDecodeParamsCreate(nvjpeg_handle, &nvjpeg_decode_params));
    CHECK_NVJPEG(nvjpegEncoderStateCreate(nvjpeg_handle, &nvjpeg_encoder_state, NULL));
    CHECK_NVJPEG(nvjpegEncoderParamsCreate(nvjpeg_handle, &nvjpeg_encode_params, NULL));

    assert(std::filesystem::is_directory(input_dir_));
    assert(std::filesystem::is_directory(output_dir_));

    try {
      for (const auto &entry : std::filesystem::directory_iterator(input_dir_)) {
        input_files_.push_back(entry.path().string());
      }
    } catch (const std::filesystem::filesystem_error &err) {
      std::cerr << "Error reading input directory: " << err.what() << std::endl;
      exit(EXIT_FAILURE);
    }
    assert(input_files_.size() <= image_num_);
  }

  auto created_to_paused() -> void override {
  }

  auto paused_to_running() -> void override {
  }

  auto running_to_paused() -> void override {
  }

  auto running_to_finished() -> void override {
  }

/*

    def to_stopped(self) -> None:
        with open(f"./{self.task_name}_{self.task_id}_side_task.txt", "w") as f:
            f.write(str(self.step_counter * self.batch_size))
            f.flush()
*/

  auto to_stopped() -> void override {
    CHECK_NVJPEG(nvjpegEncoderParamsDestroy(nvjpeg_encode_params))
    CHECK_NVJPEG(nvjpegDecodeParamsDestroy(nvjpeg_decode_params))
    CHECK_NVJPEG(nvjpegEncoderStateDestroy(nvjpeg_encoder_state))
    CHECK_NVJPEG(nvjpegJpegStateDestroy(nvjpeg_decoder_state))
    CHECK_NVJPEG(nvjpegDestroy(nvjpeg_handle))
    std::string out_file_name = name_ + "_" + std::to_string(task_id_) + "_side_task.txt";
    std::ofstream out_file(out_file_name);
    assert(out_file.is_open());
    out_file << image_iter_;
    out_file.flush();
    out_file.close();
  }

  auto is_finished() -> bool override {
    return max_iter_ != 0 and image_iter_ >= max_iter_;
  }

  auto step() -> void override {
    auto input_file = input_files_[image_iter_ % image_num_];
    image_iter_++;
    auto decode_time = double();
    auto image_error_code = decodeResizeEncodeOneImage(input_file,
                                                       output_dir_,
                                                       decode_time,
                                                       resize_width_,
                                                       resize_height_,
                                                       jpeg_quality_);
    assert(image_error_code == 0);
  }
};

grpc::Server *server_ptr;

void signal_handler(int signum) {
  printf("Received signal %d\n", signum);

  // cleanup and close up stuff here
  // terminate program
  sleep(1);
  server_ptr->Shutdown();
  printf("Exit task\n");

  exit(0);
}

int main(int argc, char **argv) {
  argparse::ArgumentParser program("image_resize_watermark_side_task");
  program.add_argument("-n", "--name");
  program.add_argument("-s", "--scheduler_addr");
  program.add_argument("-i", "--task_id");
  program.add_argument("-d", "--device");
  program.add_argument("-a", "--addr");
  program.add_argument("--duration");
  program.add_argument("--profiler_level");
  program.add_argument("--input_dir");
  program.add_argument("--output_dir");
  program.add_argument("--jpeg_quality");
  program.add_argument("--resize_width");
  program.add_argument("--resize_height");
  program.add_argument("--image_num");
  program.add_argument("--max_iter");
  std::cout << __FILE__ << ":" << __LINE__ << std::endl;
  try {
    program.parse_args(argc, argv);
  } catch (const std::runtime_error &err) {
    std::cout << err.what() << std::endl;
    std::cout << program;
    return 1;
  }
  std::cout << __FILE__ << ":" << __LINE__ << std::endl;
  auto name = program.get<std::string>("--name");
  auto scheduler_addr = program.get<std::string>("--scheduler_addr");
  std::cout << __FILE__ << ":" << __LINE__ << std::endl;
  auto task_id = std::stoi(program.get<std::string>("--task_id"));
  std::cout << __FILE__ << ":" << __LINE__ << std::endl;
  auto device = program.get<std::string>("--device");
  std::cout << __FILE__ << ":" << __LINE__ << std::endl;
  auto addr = program.get<std::string>("--addr");
  std::cout << __FILE__ << ":" << __LINE__ << std::endl;
  auto duration = std::stod(program.get<std::string>("--duration"));
  std::cout << __FILE__ << ":" << __LINE__ << std::endl;
  auto profiler_level = std::stoi(program.get<std::string>("--profiler_level"));
  std::cout << __FILE__ << ":" << __LINE__ << std::endl;
  auto input_dir = program.get<std::string>("--input_dir");
  std::cout << __FILE__ << ":" << __LINE__ << std::endl;
  auto output_dir = program.get<std::string>("--output_dir");
  std::cout << __FILE__ << ":" << __LINE__ << std::endl;
  auto jpeg_quality = std::stoi(program.get<std::string>("--jpeg_quality"));
  std::cout << __FILE__ << ":" << __LINE__ << std::endl;
  auto resize_width = std::stoi(program.get<std::string>("--resize_width"));
  std::cout << __FILE__ << ":" << __LINE__ << std::endl;
  auto resize_height = std::stoi(program.get<std::string>("--resize_height"));
  std::cout << __FILE__ << ":" << __LINE__ << std::endl;
  auto image_num = std::stoi(program.get<std::string>("--image_num"));
  std::cout << __FILE__ << ":" << __LINE__ << std::endl;
  auto max_iter = std::stoi(program.get<std::string>("--max_iter"));
  auto task = ImageResizeWatermarkSideTask(task_id,
                                           name,
                                           device,
                                           scheduler_addr,
                                           duration,
                                           profiler_level,
                                           input_dir,
                                           output_dir,
                                           jpeg_quality,
                                           resize_width,
                                           resize_height,
                                           image_num,
                                           max_iter);
  std::cout << __FILE__ << ":" << __LINE__ << std::endl;

  signal(SIGINT, signal_handler);

  auto service = TaskServiceImpl(&task);
  grpc::ServerBuilder builder;
  builder.AddListeningPort(addr, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
  std::cout << "Server listening on " << addr << std::endl;
  task.start_runner();
  server->Wait();

  return 0;
}