#include "yolo.h"
using namespace std;
using namespace cv;
using namespace cv::dnn;

// 容器
std::vector<Output> result;

/******************模型读取************************/
bool Yolo::readModel(Net &net, string &netPath, bool isCuda = true)
{
  try
  {
    net = readNet(netPath);
  }
  catch (const std::exception &)
  {
    return false;
  }
  // cuda
  if (isCuda = 1)
  {
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA); //_FP16
  }
  // cpu
  else
  {
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
  }
  return true;
}

/******************目标检测************************/
bool Yolo::Detect(Mat &SrcImg, Net &net, vector<Output> &output)
{
  Mat blob;
  int col = SrcImg.cols;
  int row = SrcImg.rows;
  int maxLen = MAX(col, row);
  Mat netInputImg = SrcImg.clone();
  if (maxLen > 1.2 * col || maxLen > 1.2 * row)
  {
    Mat resizeImg = Mat::zeros(maxLen, maxLen, CV_8UC3);
    SrcImg.copyTo(resizeImg(Rect(0, 0, col, row)));
    netInputImg = resizeImg;
  }
  blobFromImage(netInputImg, blob, 1 / 255.0, cv::Size(netWidth, netHeight), cv::Scalar(0, 0, 0), true, false);
  // 如果在其他设置没有问题的情况下但是结果偏差很大，可以尝试下用下面两句语句//
  // blobFromImage(netInputImg, blob, 1 / 255.0, cv::Size(netWidth, netHeight), cv::Scalar(104, 117, 123), true, false);
  // blobFromImage(netInputImg, blob, 1 / 255.0, cv::Size(netWidth, netHeight), cv::Scalar(114, 114,114), true, false);
  net.setInput(blob);
  std::vector<cv::Mat> netOutputImg;
  // vector<string> outputLayerName{"345","403", "461","output" };
  // net.forward(netOutputImg, outputLayerName[3]); //获取output的输出//
  try
  {                                                                // release OK
    net.forward(netOutputImg, net.getUnconnectedOutLayersNames()); // debug报错 initCUDABackend CUDA backend will fallback to the CPU implementation for the layer "_input"
  }
  catch (const std::exception &e)
  {
    cout << e.what();
  }

  std::vector<int> classIds;      // 结果id数组//
  std::vector<float> confidences; // 结果每个id对应置信度数组//
  std::vector<cv::Rect> boxes;    // 每个id矩形框//
  float ratio_h = (float)netInputImg.rows / netHeight;
  float ratio_w = (float)netInputImg.cols / netWidth;
  int net_width = className.size() + 5; // 输出的网络宽度是类别数+5//
  float *pdata = (float *)netOutputImg[0].data;
  for (int stride = 0; stride < strideSize; stride++)
  { // stride
    int grid_x = (int)(netWidth / netStride[stride]);
    int grid_y = (int)(netHeight / netStride[stride]);
    for (int anchor = 0; anchor < 3; anchor++)
    { // anchors
      const float anchor_w = netAnchors[stride][anchor * 2];
      const float anchor_h = netAnchors[stride][anchor * 2 + 1];
      for (int i = 0; i < grid_y; i++)
      {
        for (int j = 0; j < grid_x; j++)
        {
          float box_score = pdata[4];
          ; // 获取每一行的box框中含有某个物体的概率//
          if (box_score >= boxThreshold)
          {
            cv::Mat scores(1, className.size(), CV_32FC1, pdata + 5);
            Point classIdPoint;
            double max_class_socre;
            minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
            max_class_socre = (float)max_class_socre;
            if (max_class_socre >= classThreshold)
            {
              // rect [x,y,w,h]
              float x = pdata[0]; // x
              float y = pdata[1]; // y
              float w = pdata[2]; // w
              float h = pdata[3]; // h
              int left = (x - 0.5 * w) * ratio_w;
              int top = (y - 0.5 * h) * ratio_h;
              if (class_select.find(classIdPoint.x) != class_select.end()) // 筛选满足自己的框
              {
                classIds.push_back(classIdPoint.x);
                confidences.push_back(max_class_socre * box_score);
                boxes.push_back(Rect(left, top, int(w * ratio_w), int(h * ratio_h)));
              }
            }
          }
          pdata += net_width; // 下一行//
        }
      }
    }
  }
  // 执行非最大抑制以消除具有较低置信度的冗余重叠框（NMS）//
  vector<int> nms_result;
  NMSBoxes(boxes, confidences, nmsScoreThreshold, nmsThreshold, nms_result);
  for (int i = 0; i < nms_result.size(); i++)
  {
    int idx = nms_result[i];
    Output result;
    result.id = classIds[idx];
    result.confidence = confidences[idx];
    result.box = boxes[idx];
    output.push_back(result);
  }
  if (output.size())
    return true;
  else
    return false;
}

/******************画框************************/
void Yolo::drawPred(Mat &img, vector<Output> result, vector<Scalar> color)
{
  for (int i = 0; i < result.size(); i++)
  {
    int left = result[i].box.x;
    int top = result[i].box.y;
    int color_num = i;
    rectangle(img, result[i].box, color[result[i].id], 2, 8);

    // 为标签加上序号
    string label = to_string(i + 1) + ": " + className[result[i].id] + ":" + to_string(result[i].confidence);

    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = max(top, labelSize.height);
    putText(img, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 1, color[result[i].id], 2);
  }
}

/******************模板匹配************************/
void Yolo::matching(Mat &Lsrcimg, Mat &Rsrcimg, std::vector<Output> Lresultinput, std::vector<Output> Rresultinput, std::vector<std::vector<int>> &OutputInformation)
{

  // cout << "Lresultinput.size=" << Lresultinput.size() << endl;
  // cout << "Rresultinput.size=" << Rresultinput.size() << endl;
  // 遍历左图像中的每个目标，并根据与坐标原点的距离进行排序
  std::sort(Lresultinput.begin(), Lresultinput.end(), [](const Output &a, const Output &b)
            {
        cv::Point aCenter(a.box.x + a.box.width / 2, a.box.y + a.box.height / 2);
        cv::Point bCenter(b.box.x + b.box.width / 2, b.box.y + b.box.height / 2);
        return aCenter.x < bCenter.x; });

  // 遍历右图像中的每个目标，并根据与坐标原点的距离进行排序
  std::sort(Rresultinput.begin(), Rresultinput.end(), [](const Output &a, const Output &b)
            {
        cv::Point aCenter(a.box.x + a.box.width / 2, a.box.y + a.box.height / 2);
        cv::Point bCenter(b.box.x + b.box.width / 2, b.box.y + b.box.height / 2);
        return aCenter.x < bCenter.x; });

  // 确保左右图像的检测数量一致
  if (Lresultinput.size() == Rresultinput.size())
  {
    // 按顺序进行匹配
    cout << "左图数量=" << Lresultinput.size() << "右图数量=" << Rresultinput.size() << endl;
    for (int i = 0; i < Lresultinput.size(); i++)
    {
      // 获取左图中第 i 个检测框的中心坐标
      cv::Point leftCenter(Lresultinput[i].box.x + Lresultinput[i].box.width / 2,
                           Lresultinput[i].box.y + Lresultinput[i].box.height / 2);

      // 获取右图中第 i 个检测框的中心坐标
      cv::Point rightCenter(Rresultinput[i].box.x + Rresultinput[i].box.width / 2,
                            Rresultinput[i].box.y + Rresultinput[i].box.height / 2);

      // 获取左检测框的宽和高
      int leftWidth = Lresultinput[i].box.width;
      int leftHeight = Lresultinput[i].box.height;

      // 输出匹配信息
      cout << "找到匹配项： 左图=" << i << ";" << "右图= " << i << endl;

      // 创建一个临时向量用于存储匹配信息
      std::vector<int> matchInformation;
      // if (Lresultinput.size() > 20)
      // {
      // }
      // 将匹配信息插入 matchInformation 向量中
      matchInformation.push_back(leftCenter.x);
      matchInformation.push_back(leftCenter.y);
      matchInformation.push_back(rightCenter.x);
      matchInformation.push_back(rightCenter.y);
      matchInformation.push_back(leftWidth);
      matchInformation.push_back(leftHeight);
      // cout << "matchInformation=" << matchInformation.size() << endl;
      // 将 matchInformation 插入 OutputInformation 容器
      OutputInformation.push_back(matchInformation);
      matchInformation.clear();
    }
  }
  else
  {
    cout << "!!!!左右检测数不匹配" << endl;
  }
}