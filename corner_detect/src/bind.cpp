#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>
#include "CornerDetAC.h"
#include "ChessboradStruct.h"

namespace py = pybind11;

py::array_t<float> get_corners(py::array_t<uint8_t> img, int cols, int rows) {
    // transform the numpy → cv::Mat
    py::buffer_info buf = img.request();
    int height = buf.shape[0];
    int width = buf.shape[1];
    uint8_t* ptr = static_cast<uint8_t*>(buf.ptr);

    cv::Mat src1(height, width, CV_8UC1, ptr);
    cv::Mat src = src1.clone();

    if (src.empty()) {
        throw std::runtime_error("Input image is empty!");
    }

    // detect the corner
    std::vector<cv::Point> corners_p;
    std::vector<cv::Mat> chessboards;
    CornerDetAC corner_detector(src);
    ChessboradStruct chessboardstruct;
    Corners corners_s;
    corner_detector.detectCorners(src, corners_p, corners_s, 0.01);
    chessboardstruct.chessboardsFromCorners(corners_s, chessboards, 0.6);

    // transform the result
    if (chessboards.empty()) {
        return py::array_t<float>(std::vector<ssize_t>{0, 2});
    }

    const cv::Mat& first_board = chessboards[0];
    size_t num_rows = first_board.rows;
    size_t num_cols = first_board.cols;
    size_t total_points = num_rows * num_cols;

    py::array_t<float> output(std::vector<ssize_t>{static_cast<ssize_t>(total_points), 2});
    auto r = output.mutable_unchecked<2>();

    for (size_t i = 0; i < num_rows; i++) {
        for (size_t j = 0; j < num_cols; j++) {
            int idx = first_board.at<int>(i, j);
            cv::Point2f pt = corners_s.p[idx];
            size_t flat_idx = i * num_cols + j;

            r(flat_idx, 0) = pt.x;
            r(flat_idx, 1) = pt.y;
        }
    }

    return output;
}

PYBIND11_MODULE(tcar, m) {
    m.def("get_corners", &get_corners, "Detect chessboard corners",
          py::arg("img"), py::arg("cols"), py::arg("rows"));
}