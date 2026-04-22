#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

Mat loadHomography(string filename) {
    Mat H(3, 3, CV_64F);
    FILE* file = fopen(filename.c_str(), "r");

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            fscanf(file, "%lf", &H.at<double>(i, j));
        }
    }

    fclose(file);
    return H;
}

void featherBlend(Mat& base, Mat& overlay) {
    for (int y = 0; y < base.rows; y++) {
        for (int x = 0; x < base.cols; x++) {

            Vec3b pix1 = base.at<Vec3b>(y, x);
            Vec3b pix2 = overlay.at<Vec3b>(y, x);

            if (pix2 != Vec3b(0,0,0)) {
                if (pix1 == Vec3b(0,0,0)) {
                    base.at<Vec3b>(y, x) = pix2;
                } else {
                    base.at<Vec3b>(y, x) = (pix1 / 2 + pix2 / 2);
                }
            }
        }
    }
}

int main() {

    Mat img1 = imread(./images/left.jpg);
    Mat img2 = imread(./images/right.jpg);

    if (img1.empty() || img2.empty()) {
        cout << "Error loading images!" << endl;
        return -1;
    }

    Mat H = loadHomography("H.txt");

    int width = img1.cols + img2.cols;
    int height = max(img1.rows, img2.rows);

    Mat result(height, width, CV_8UC3, Scalar(0,0,0));

    warpPerspective(img1, result, H, result.size());

    Mat half(result, Rect(0, 0, img2.cols, img2.rows));
    img2.copyTo(half);

    featherBlend(result, result);

    imwrite("panorama.jpg", result);

    cout << "Panorama saved as panorama.jpg" << endl;

    return 0;
}
