#include "sgbm.h"
#include "yolo.h"

int main()
{
  string model_path = "../weights/yolov5s.onnx";

  Yolo YL;
  depth DP;

  // 网络
  cv::dnn::Net net;
  // yolov5容器
  std::vector<Output> leftResult;
  std::vector<Output> rightResult;
  std::vector<std::vector<int>> OutInfoemation;
  // 读取左右图像
  VideoCapture cap(4);
  cap.set(CAP_PROP_FRAME_WIDTH, 2580);
  cap.set(CAP_PROP_FRAME_HEIGHT, 720);
  cap.set(CAP_PROP_FPS, 30);
  cap.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M', 'J', 'P', 'G'));

  // false: CPU  true:GPU
  if (YL.readModel(net, model_path, true))
  {
    cout << "read net ok!" << endl;
  }
  else
  {
    cout << "Fail" << endl;
  }
  pcl::visualization::PCLVisualizer viewer("Point Cloud Viewer");
  viewer.setBackgroundColor(0, 0, 0);
  viewer.addPointCloud<pcl::PointXYZRGB>(cloud_prt, "cloud");
  viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud");

  while (!viewer.wasStopped())
  {
    auto start = getTickCount(); // 帧率计算开始

    cv::cuda::GpuMat gpu;

    /*****读取图像并分割左右图像*********/
    cap >> src;

    Size srize = src.size();
    srcleft = src.colRange(0, srize.width / 2).clone();
    srcright = src.colRange(srize.width / 2, srize.width).clone();

    // ----生成随机颜色yolo部分-----------//
    srand(time(0));
    for (int i = 0; i < 1; i++)
    { // i为你模型class的种类数量
      int b = rand() % 256;
      int g = rand() % 256;
      int r = rand() % 256;
      color.push_back(Scalar(b, g, r));
    }
    color.push_back(Scalar(0, 0, 255));
    //--------------------------------//

    // 对图片进行畸变校正
    DP.calibration(srcleft, srcright);
    // --------------------主程序-------------------------//
    if (YL.Detect(srcleft, net, leftResult) == YL.Detect(srcright, net, rightResult))
    {
      if (leftResult.size() == rightResult.size() && leftResult.size() != 0)
      {
        cout << "L=" << leftResult.size() << endl;
        cout << "R=" << rightResult.size() << endl;
        YL.matching(srcleft, srcright, leftResult, rightResult, OutInfoemation);         // 模板匹配
        YL.drawPred(srcleft, leftResult, color);                                         // 左图画框
        YL.drawPred(srcright, rightResult, color);                                       // 右图画框
        DP.Vec_connector(srcleft, srcright, OutInfoemation, Vec_P, Vec_D, Vec_L, Vec_R); // 信息提取
        DP.get_disp_Img(Vec_L, Vec_R, Vec_disp);                                         // 获取视差
        DP.get_point_cloud(srcleft, Vec_P, Vec_D, Vec_disp, cloud);                      // 获取点云
      }
    }
    //--------------------------------------------------//

    /*******点云可视化***********/
    if (cloud->points.size() != 0)
    {
      cloud->points.resize(cloud->height * cloud->width);
      viewer.updatePointCloud(cloud, "cloud");
      viewer.spinOnce();
    }

    // 创建一个容器来存储每个目标的质心
    vector<pcl::PointXYZRGB> centroids;
    /*******点云处理***********/
    if (cloud->points.size() != 0)
    {
      // DP.filterPointCloud(cloud);
      // DP.get_object_centroids(srcleft, Vec_P, Vec_D, Vec_disp, cloud, centroids);
      // 打印每个目标的质心
      // for (const auto &centroid : centroids)
      // {
      //   cout << "Object centroid: x=" << centroid.x << " y=" << centroid.y << " z=" << centroid.z << endl;
      // }
    }

    // 防止数据残留，在循环的结尾清一次容器
    if (!leftResult.empty())
    {
      // YOLO部分的容器清除
      leftResult.clear();
      rightResult.clear();
      OutInfoemation.clear();
      // PCL部分的容器清除
      Vec_L.clear();
      Vec_R.clear();
      Vec_P.clear();
      Vec_D.clear();
      Vec_disp.clear();
      cloud->clear();
    }

    /******计算帧率结束********************* */
    auto end = getTickCount();
    auto totalTime = (end - start) / getTickFrequency();
    auto fps = 1 / totalTime;
    putText(srcleft, "FPS: " + to_string(int(fps)), Point(50, 50), FONT_HERSHEY_DUPLEX, 1, Scalar(255, 255, 255), 2, false);
    imshow("l", srcleft);
    imshow("r", srcright);
    waitKey(30);
  }
  return 0;
}
