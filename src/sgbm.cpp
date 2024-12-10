#include "sgbm.h"
#include "yolo.h"

/******************畸变矫正************************/
void depth::calibration(Mat &left, Mat &right)
{
    stereoRectify(K1, D1, K2, D2, left.size(), R, T, R1, R2, P1, P2, Q);
    // 进行矫正畸变与基线对齐
    cv::initUndistortRectifyMap(K1, D1, R1, P1, left.size(), CV_32F, Lmapx, Lmapy);
    cv::initUndistortRectifyMap(K2, D2, R2, P2, right.size(), CV_32F, Lmapx, Rmapy);
    cv::remap(left, left, Lmapx, Lmapy, cv::INTER_LINEAR);
    cv::remap(right, right, Lmapx, Rmapy, cv::INTER_LINEAR);
    Rect roil(40, 10, 1200, 680);
    Rect roir(40, 10, 1200, 680);
    left = left(roil);
    right = right(roir);
}

/******************模板匹配************************/
void depth::Vec_connector(Mat In_L, Mat In_R, vector<vector<int>> Input, vector<Point2i> &output_P, vector<int> &output_D, vector<Mat> &output_L, vector<Mat> &output_R)
{
    if (Input.size() > 0)
    {
        for (int i = 0; i < Input.size(); i++)
        {
            Rect L, R;

            // if (Input[i][6] == 0)
            // {
            L = Rect(Input[i][0] - (Input[i][4] / 2), Input[i][1] - (Input[i][5] / 2), Input[i][4], Input[i][5]);
            R = Rect(Input[i][2] - (Input[i][4] / 2), Input[i][1] - (Input[i][5] / 2), Input[i][4], Input[i][5]);
            // }
            // else
            // {
            //     L = Rect(Input[i][0] - (Input[i][4] / 2), Input[i][3] - (Input[i][5] / 2), Input[i][4], Input[i][5]);
            //     R = Rect(Input[i][2] - (Input[i][4] / 2), Input[i][3] - (Input[i][5] / 2), Input[i][4], Input[i][5]);
            // }

            // 检查ROI是否在图像边界内
            if (L.x >= 0 && L.y >= 0 && (L.x + L.width) <= In_L.cols && (L.y + L.height) <= In_L.rows &&
                R.x >= 0 && R.y >= 0 && (R.x + R.width) <= In_R.cols && (R.y + R.height) <= In_R.rows)
            {
                // 使用临时变量保存ROI
                Mat leftROI = In_L(L).clone();
                Mat rightROI = In_R(R).clone();

                // 输出
                Point P(L.x, L.y);
                output_L.push_back(leftROI);
                output_R.push_back(rightROI);
                output_P.push_back(P);
                output_D.push_back(Input[i][0] - Input[i][2]);
            }
        }
    }
}

/******************转灰度图************************/
void depth::ImgProcessing(Mat &left, Mat &right)
{
    cvtColor(left, left, COLOR_BGR2GRAY);
    cvtColor(right, right, COLOR_BGR2GRAY);
}

/******************SGBM算法************************/
void depth::SGBM(Mat srcleft, Mat srcright, Mat &disp)
{
    depth d;
    d.ImgProcessing(srcleft, srcright);
    // 定义SGBM算法参数
    Size leftsize = srcleft.size();
    int min_disp = 1;                                  // 表示最小视差，即两幅图像中对应点的水平位置差的最小值。默认值为0。
    int num_disp = ((leftsize.width / 16) + 15) & -16; // 表示搜索的视差范围，值越大，计算量越大，计算时间越长。需要根据摄像头的分辨率和物体距离等因素来进行设置。默认值为64。一般是16的倍数
                                                       //    cout<<"num_disp:  "<<num_disp<<endl;
    int block_size = ((num_disp / 16) + 3) & -3;       // 表示视差计算时使用的块大小，通常设置为奇数，常用的值为3、5、7。值越大，计算量越大，计算时间越长，但可以获得更加精确的视差图。
    // cout<<"block_size:  "<<block_size<<endl;
    int p1 = 8 * srcleft.channels() * block_size * block_size;  // 表示SGBM算法中惩罚参数1，用于控制视差变化时的代价。默认值为0。
    int p2 = 32 * srcleft.channels() * block_size * block_size; // 表示SGBM算法中惩罚参数2，用于控制相邻像素之间视差变化时的代价。默认值为0。
    // P1、P2是惩罚参数，用于平滑视差图。需要根据具体的场景进行调整。通常情况下，P1和P2的比值在5~10之间。
    int disp12_max_diff = 1;       // 表示左右视差图之间的最大差异，通常设置为0或1即可。
    int pre_filter_cap = 63;       // 表示预处理过程中像素值的最大容许范围，用于平滑图像，通常取值在10~63之间。
    int uniqueness_ratio = 10;     // 表示视差值一致性检查的代价函数。默认值为0。
    int speckle_window_size = 100; // 表示检查视差图上的相同区域时使用的窗口大小。默认值为0
    int speckle_range = 32;        // 表示在检查视差图上的相同区域时，两个视差值之间的最大容许差异。默认值为0。
    int full_dp = 2;               // 模式
    // 创建SGBM算法对象并计算视差图
    Ptr<StereoSGBM> sgbm = StereoSGBM::create(min_disp, num_disp, block_size, p1, p2, disp12_max_diff, pre_filter_cap, uniqueness_ratio, speckle_window_size, speckle_range, full_dp);
    // Ptr<StereoSGBM> sgbm = StereoSGBM::create(1, 96, 9, 8 * 9 * 9, 32 * 9 * 9, 1, 63, 10, 100, 32,1);
    sgbm->compute(srcleft, srcright, disp);
    disp.convertTo(disp, CV_32F, 1 / 16.0f);
    // 指定中值滤波的窗口大小（奇数值）
    int kernel_size = 3;
    // bilateralFilter( disp, disp8,5, 2.5, 10,cv::BORDER_DEFAULT);
    // 应用中值滤波
    medianBlur(disp, disp, kernel_size);
    // imshow("disp", disp / num_disp);
}

/******************SGBM算法CUDA运算************************/
void depth::SGBM_CUDA(Mat SL, Mat SR, Mat &DP)
{
    depth d;
    d.ImgProcessing(SL, SR);

    cv::cuda::GpuMat d_srcleft, d_srcright, d_disp;

    // 将图像上传到 GPU
    d_srcleft.upload(SL);
    d_srcright.upload(SR);

    // 创建 CUDA 版的 SGBM 对象
    // Size leftsize = SL.size();
    // int num_disp = ((leftsize.width / 16) + 15) & -16; // 表示搜索的视差范围，值越大，计算量越大，计算时间越长。需要根据摄像头的分辨率和物体距离等因素来进行设置。默认值为64。一般是16的倍数
    //                                                    //    cout<<"num_disp:  "<<num_disp<<endl;
    // int block_size = ((num_disp / 16) + 3) & -3;       // 表示视差计算时使用的块大小，通常设置为奇数，常用的值为3、5、7。值越大，计算量越大，计算时间越长，但可以获得更加精确的视差图。
    // int num_disp = ((SL.size().width / 16) + 15) & -16;
    // int block_size = 3;
    int num_disp = 64; // 设置为固定的 16 的倍数，且在合理范围内
    int block_size = 3;

    cv::Ptr<cv::cuda::StereoSGM> sgbm = cv::cuda::createStereoSGM(0, num_disp, block_size);

    // 计算视差图
    sgbm->compute(d_srcleft, d_srcright, d_disp);

    // 从 GPU 下载视差图到 CPU
    d_disp.download(DP);

    // 将视差图转换为 CV_32F 格式
    DP.convertTo(DP, CV_32F, 1.0 / 16.0f);

    // 对视差图进行中值滤波
    int kernel_size = 3;
    cv::medianBlur(DP, DP, kernel_size);
}

/******************视差获取************************/
void depth::get_disp_Img(vector<Mat> input_L, vector<Mat> input_R, vector<Mat> &disp)
{
    depth de;
    Mat dis;
    for (int i = 0; i < input_R.size(); i++)
    {
        // de.SGBM(input_L[i], input_R[i], dis);
        de.SGBM_CUDA(input_L[i], input_R[i], dis);
        disp.push_back(dis);
    }
}

/******************点云计算************************/
void depth::get_point_cloud(Mat L, vector<Point2i> input_P, vector<int> input_D, vector<Mat> disp, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &output_cloud)
{
    double fx = K_left[0][0], fy = K_left[1][1], cx = K_left[0][2], cy = K_left[1][2];
    double b = 0.11984263187384732; // 基线距离

    for (int i = 0; i < disp.size(); i++)
    {
        for (int v = 0; v < disp[i].cols; ++v)
        {
            for (int u = 0; u < disp[i].rows; ++u)
            {
                // 获取当前像素处的深度值
                Vec3d point(0, 0, 0); // 前三维为xyz
                // ushort d = disp[i].ptr<ushort>(u)[v];
                // 将深度值转换为世界坐标
                double x = ((input_P[i].y + u) - cy) / fy;
                double y = ((input_P[i].x + v) - cx) / fx;
                double depth = fx * b / (input_D[i] + (disp[i].at<float>(u, v)));
                double depth1 = fx * b / (input_D[i]);
                // 判断深度值是否为inf
                if (std::isinf(depth) || depth < 0 || depth == depth1) //|| depth > 10)
                    continue;                                          // 跳过深度值为inf的点
                // 将点云数据添加到点云对象中
                pcl::PointXYZRGB pclPoint;
                pclPoint.x = -x * depth;
                pclPoint.y = y * depth;
                pclPoint.z = -depth;
                // 获取彩色图像中对应像素处的颜色值
                pclPoint.b = L.ptr<uchar>(input_P[i].y + u)[(input_P[i].x + v) * 3];
                pclPoint.g = L.ptr<uchar>(input_P[i].y + u)[(input_P[i].x + v) * 3 + 1];
                pclPoint.r = L.ptr<uchar>(input_P[i].y + u)[(input_P[i].x + v) * 3 + 2];
                output_cloud->push_back(pclPoint);
            }
        }
    }
}

/******************点云滤波************************/
void depth::filterPointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &filter_cloud)
{
    // VoxelGrid滤波器
    pcl::VoxelGrid<pcl::PointXYZRGB> sor;
    sor.setInputCloud(cloud);
    sor.setLeafSize(0.01f, 0.01f, 0.01f); // 设置叶大小
    sor.filter(*cloud);

    // 统计滤波器，移除离群点
    pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor_rem;
    sor_rem.setInputCloud(cloud);
    sor_rem.setMeanK(50);
    sor_rem.setStddevMulThresh(1.0);
    sor_rem.filter(*cloud);
}

/******************RANSAC拟合+获取中心坐标************************/
bool depth::PointCenter(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud, pcl::PointXYZ &center, double &radius)
{
    pcl::SACSegmentation<pcl::PointXYZRGB> seg;
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_sphere(new pcl::PointCloud<pcl::PointXYZRGB>);

    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_SPHERE); // 选择球体模型
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.01); // 距离阈值
    seg.setInputCloud(cloud);
    seg.segment(*inliers, *coefficients);

    if (inliers->indices.size() == 0)
    {
        std::cerr << "Could not estimate a spherical model for the given dataset." << std::endl;
    }

    // 提取圆球的点云
    pcl::ExtractIndices<pcl::PointXYZRGB> extract;
    extract.setInputCloud(cloud);
    extract.setIndices(inliers);
    extract.setNegative(false);
    extract.filter(*cloud_sphere);

    // 获取圆心和半径
    center.x = coefficients->values[0];
    center.y = coefficients->values[1];
    center.z = coefficients->values[2];
    radius = coefficients->values[3];
}

// 点云质心坐标计算
void depth::calculateCenter(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, pcl::PointXYZRGB &centroid)
{
    double sum_x = 0.0, sum_y = 0.0, sum_z = 0.0;
    int point_count = cloud->points.size();

    for (const auto &point : cloud->points)
    {
        sum_x += point.x;
        sum_y += point.y;
        sum_z += point.z;
    }

    if (point_count > 0)
    {
        centroid.x = sum_x / point_count;
        centroid.y = sum_y / point_count;
        centroid.z = sum_z / point_count;
    }
}

// 对每个目标点云进行处理，并计算质心
void depth::processPointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud, pcl::PointXYZRGB &centroid)
{
    // 先进行滤波处理
    filterPointCloud(cloud);

    pcl::PointXYZ center;
    double radius;
    if (depth::PointCenter(cloud, center, radius))
    {
        std::cout << "Sphere center: (" << center.x << ", " << center.y << ", " << center.z << ")" << std::endl;
        std::cout << "Sphere radius: " << radius << std::endl;
    }

    // 计算处理后的点云质心
    depth::calculateCenter(cloud, centroid);
}

// 获取每个目标的点云并计算其质心
void depth::get_object_centroids(Mat LImage, vector<Point2i> input_P, vector<int> input_D, vector<Mat> disp, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &output_cloud, vector<pcl::PointXYZRGB> &centroids)
{
    // 对每个目标提取点云并计算质心
    for (size_t i = 0; i < input_P.size(); ++i)
    {
        // 创建点云对象
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

        // 确保传递给 get_point_cloud 的参数类型与定义匹配
        // 注意: input_P 和 input_D 是 vectors，这里传递的是 single elements
        depth::get_point_cloud(LImage, vector<Point2i>{input_P[i]}, vector<int>{input_D[i]}, vector<Mat>{disp[i]}, cloud);

        // 计算点云的质心
        pcl::PointXYZRGB centroid;
        depth::calculateCenter(cloud, centroid);

        // 将计算出的质心添加到 centroids 列表中
        centroids.push_back(centroid);

        // 将处理过的点云添加到 output_cloud 中
        *output_cloud += *cloud;
    }
}
