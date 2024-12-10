#pragma once
#include <iostream>
#include <vector>
#include <string>
/********opencv**************/
#include <opencv2/opencv.hpp>
#include <opencv2/cudastereo.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <pcl/point_cloud.h>
/********pcl*****************/
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/point_types.h>
/********点云处理**************/
#include <pcl/filters/passthrough.h> //体素滤波
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h> //统计滤波
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>

using namespace std;
using namespace cv;
using namespace pcl;

// point cloud
// 图像
Mat src, srcleft, srcright, disp, R_L, R_R;
// 容器
vector<int> OneImgInformation, Vec_D;
vector<Point2i> Vec_P;
vector<Mat> Vec_L, Vec_R, Vec_disp;
pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_prt(new pcl::PointCloud<pcl::PointXYZRGB>);
pcl::PointCloud<pcl::PointXYZRGB>::Ptr filter_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);  // 创建体素滤波器
pcl::PointCloud<pcl::PointXYZRGB>::Ptr cluster_cloud(new pcl::PointCloud<pcl::PointXYZRGB>); // 创建统计滤波器
pcl::PointCloud<pcl::PointXYZ>::Ptr cldPtr(new pcl::PointCloud<pcl::PointXYZ>);              // 聚类
pcl::PointCloud<pcl::PointXYZRGB>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZRGB>);

class depth
{
private:
public:
  /***********************相机标定参数********************************************************/
  double K_left[3][3] = {1042.1805768790434, 0.0, 645.2332888285383, 0.0, 1042.073985059462, 524.3825390763863, 0.0, 0.0, 1.0};
  Mat K1 = cv::Mat(3, 3, cv::DataType<double>::type, K_left);
  double d_left[1][5] = {-0.00023898719095752754, 0.04806184914023602, 0.0028936757606397834, -0.005719968415176577, 0.0};
  Mat D1 = cv::Mat(1, 5, cv::DataType<double>::type, d_left);
  double K_right[3][3] = {1039.7046813454006, 0.0, 673.2128355167965, 0.0, 1039.9100011107857, 507.81532063549355, 0.0, 0.0, 1.0};
  Mat K2 = cv::Mat(3, 3, cv::DataType<double>::type, K_right);
  double d_right[1][5] = {-0.03559694316544863, 0.08106676541971364, -0.0021041788649294713, -0.006250474895216592, 0.0};
  Mat D2 = cv::Mat(1, 5, cv::DataType<double>::type, d_right);
  double R_stereo[3][3] = {0.9999826031345356, 0.001573180502263105, -0.0056849389957245805, -0.001539759627957232, 0.9999815363280321, 0.005878447355957454, 0.0056940818896397865, -0.005869591649846638, 0.9999665621036026};
  Mat R = cv::Mat(3, 3, cv::DataType<double>::type, R_stereo);
  Vec3d T = {-0.12001459985777725, 0.00039837659436836885, -0.0002745980651770644};
  /*******************************************************************************************/

  /****************视差获取函数**********************************************************/
  void calibration(Mat &left, Mat &right);
  void SGBM(Mat srcleft, Mat srcright, Mat &disp);
  void SGBM_CUDA(Mat SL, Mat SR, Mat &DP);
  void get_disp_Img(vector<Mat> input_L, vector<Mat> input_R, vector<Mat> &disp);
  void ImgProcessing(Mat &left, Mat &right);
  void get_point_cloud(Mat L, vector<Point2i> input_P, vector<int> input_D, vector<Mat> disp, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &output_cloud);
  void Vec_connector(Mat In_L, Mat In_R, vector<vector<int>> Input, vector<Point2i> &output_P, vector<int> &output_D, vector<Mat> &output_L, vector<Mat> &output_R);

  /******************点云处理函数**********************************/
  void filterPointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &filter_cloud);                            // 滤波
  bool PointCenter(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud, pcl::PointXYZ &center, double &radius); // 使用RANSAC算法拟合球体并获取中心
  void calculateCenter(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, pcl::PointXYZRGB &centroid);         // 计算点云的质心（中心点）
  void processPointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud, pcl::PointXYZRGB &centroid);      // 对每个目标点云进行处理，并计算质心
  void get_object_centroids(Mat LIamge, vector<Point2i> input_P, vector<int> input_D, vector<Mat> disp, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &output_cloud, vector<pcl::PointXYZRGB> &centroids);

  Mat P1, R1, P2, R2, Q, Lmapx, Lmapy, Rmapx, Rmapy;
  enum mode_view
  {
    LEFT,
    RIGHT,
    WLS
  }; // 定义了一个枚举类  mode_view
  mode_view view; // 输出左视差图or右视差
};