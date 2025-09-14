#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>
#include <fstream>

struct Detection {
    cv::Rect bbox;
    float confidence;
    std::string class_name;
    cv::Point3f position_3d;
};

class DeepVisualSLAM {
private:
    cv::Ptr<cv::ORB> orb;
    cv::BFMatcher bf;
    
    cv::Mat prev_frame;
    std::vector<cv::KeyPoint> prev_keypoints;
    cv::Mat prev_descriptors;
    
    // 카메라 궤적
    std::vector<cv::Point2f> trajectory;
    cv::Mat camera_position;
    
    // 카메라 내부 파라미터
    cv::Mat camera_matrix;
    cv::Mat dist_coeffs;
    
    // 딥러닝 모델들
    cv::dnn::Net yolo_net;
    cv::dnn::Net depth_net;
    bool use_yolo;
    bool use_depth;
    
    // YOLO 설정
    std::vector<std::string> class_names;
    std::vector<Detection> current_detections;

public:
    DeepVisualSLAM() {
        // ORB 특징점 검출기 초기화
        orb = cv::ORB::create(500);
        bf = cv::BFMatcher(cv::NORM_HAMMING, true);
        
        // 카메라 초기 위치
        camera_position = cv::Mat::zeros(3, 1, CV_64F);
        trajectory.push_back(cv::Point2f(0, 0));
        
        // 기본 카메라 파라미터 (웹캠용 대략값)
        camera_matrix = (cv::Mat_<double>(3, 3) << 
            640, 0, 320,
            0, 640, 240,
            0, 0, 1);
        dist_coeffs = cv::Mat::zeros(4, 1, CV_64F);
        
        // 딥러닝 모델 초기화
        use_yolo = false;
        use_depth = false;
        
        loadDeepLearningModels();
    }
    
    void loadDeepLearningModels() {
        std::cout << "🤖 Loading Deep Learning Models..." << std::endl;
        
        // YOLO 모델 로드 시도
        try {
            if (std::ifstream("models/yolov5s.onnx").good()) {
                yolo_net = cv::dnn::readNetFromONNX("models/yolov5s.onnx");
                loadClassNames("models/coco.names");
                use_yolo = true;
                std::cout << "✅ YOLO model loaded successfully!" << std::endl;
            }
            else if (std::ifstream("models/yolov4.weights").good() && 
                     std::ifstream("models/yolov4.cfg").good()) {
                yolo_net = cv::dnn::readNetFromDarknet("models/yolov4.cfg", "models/yolov4.weights");
                loadClassNames("models/coco.names");
                use_yolo = true;
                std::cout << "✅ YOLOv4 model loaded successfully!" << std::endl;
            }
            else {
                std::cout << "⚠️  YOLO models not found in models/ directory" << std::endl;
            }
        }
        catch (const cv::Exception& e) {
            std::cout << "❌ Failed to load YOLO: " << e.what() << std::endl;
        }
        
        // Depth estimation 모델 로드 시도
        try {
            if (std::ifstream("models/monodepth2.onnx").good()) {
                depth_net = cv::dnn::readNetFromONNX("models/monodepth2.onnx");
                use_depth = true;
                std::cout << "✅ Depth estimation model loaded!" << std::endl;
            }
            else {
                std::cout << "⚠️  Depth model not found (monodepth2.onnx)" << std::endl;
            }
        }
        catch (const cv::Exception& e) {
            std::cout << "❌ Failed to load depth model: " << e.what() << std::endl;
        }
        
        if (!use_yolo && !use_depth) {
            std::cout << "📝 Running in traditional SLAM mode" << std::endl;
            std::cout << "   To enable deep learning features:" << std::endl;
            std::cout << "   1. Run python model_downloader.py" << std::endl;
            std::cout << "   2. Convert models to ONNX format" << std::endl;
        }
    }
    
    void loadClassNames(const std::string& filename) {
        std::ifstream file(filename);
        std::string line;
        class_names.clear();
        
        while (std::getline(file, line)) {
            class_names.push_back(line);
        }
        
        if (class_names.empty()) {
            // COCO 클래스 기본값
            class_names = {"person", "bicycle", "car", "motorcycle", "airplane", "bus", 
                          "train", "truck", "boat", "traffic light", "fire hydrant", 
                          "stop sign", "parking meter", "bench", "bird", "cat", "dog"};
        }
    }
    
    std::vector<Detection> runYOLO(const cv::Mat& frame) {
        std::vector<Detection> detections;
        
        if (!use_yolo) return detections;
        
        try {
            cv::Mat blob;
            cv::dnn::blobFromImage(frame, blob, 1.0/255.0, cv::Size(640, 640), 
                                 cv::Scalar(), true, false, CV_32F);
            
            yolo_net.setInput(blob);
            std::vector<cv::Mat> outputs;
            yolo_net.forward(outputs, yolo_net.getUnconnectedOutLayersNames());
            
            // YOLO 출력 후처리
            float conf_threshold = 0.5;
            float nms_threshold = 0.4;
            
            std::vector<cv::Rect> boxes;
            std::vector<float> confidences;
            std::vector<int> class_ids;
            
            for (const auto& output : outputs) {
                for (int i = 0; i < output.rows; i++) {
                    const float* data = output.ptr<float>(i);
                    float confidence = data[4];
                    
                    if (confidence > conf_threshold) {
                        cv::Point class_id_point;
                        double max_class_score;
                        cv::minMaxLoc(output.row(i).colRange(5, output.cols), 
                                    0, &max_class_score, 0, &class_id_point);
                        
                        if (max_class_score > conf_threshold) {
                            int center_x = (int)(data[0] * frame.cols);
                            int center_y = (int)(data[1] * frame.rows);
                            int width = (int)(data[2] * frame.cols);
                            int height = (int)(data[3] * frame.rows);
                            
                            cv::Rect box(center_x - width/2, center_y - height/2, width, height);
                            
                            boxes.push_back(box);
                            confidences.push_back(confidence);
                            class_ids.push_back(class_id_point.x);
                        }
                    }
                }
            }
            
            // Non-Maximum Suppression
            std::vector<int> indices;
            cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, indices);
            
            for (int idx : indices) {
                Detection det;
                det.bbox = boxes[idx];
                det.confidence = confidences[idx];
                det.class_name = (class_ids[idx] < class_names.size()) ? 
                                class_names[class_ids[idx]] : "unknown";
                detections.push_back(det);
            }
        }
        catch (const cv::Exception& e) {
            std::cout << "YOLO inference error: " << e.what() << std::endl;
        }
        
        return detections;
    }
    
    cv::Mat processFrame(const cv::Mat& frame) {
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        
        // 🤖 딥러닝: YOLO 객체 검출
        current_detections = runYOLO(frame);
        
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        
        // 특징점 검출 및 기술자 계산
        orb->detectAndCompute(gray, cv::Mat(), keypoints, descriptors);
        
        cv::Mat result = frame.clone();
        
        if (!prev_frame.empty() && !descriptors.empty() && !prev_descriptors.empty()) {
            // 특징점 매칭
            std::vector<cv::DMatch> matches;
            bf.match(prev_descriptors, descriptors, matches);
            
            // 매칭 결과를 거리순으로 정렬
            std::sort(matches.begin(), matches.end());
            
            if (matches.size() > 20) {
                // 좋은 매칭점들만 선택 (상위 50개 또는 전체의 30%)
                int num_good_matches = std::min(50, (int)(matches.size() * 0.3));
                std::vector<cv::DMatch> good_matches(matches.begin(), 
                                                   matches.begin() + num_good_matches);
                
                // 매칭된 점들 추출
                std::vector<cv::Point2f> prev_pts, curr_pts;
                for (const auto& match : good_matches) {
                    prev_pts.push_back(prev_keypoints[match.queryIdx].pt);
                    curr_pts.push_back(keypoints[match.trainIdx].pt);
                }
                
                // Essential Matrix를 이용한 카메라 모션 추정
                if (prev_pts.size() >= 8) {
                    try {
                        cv::Mat mask;
                        cv::Mat E = cv::findEssentialMat(prev_pts, curr_pts, camera_matrix, 
                                                       cv::RANSAC, 0.999, 1.0, mask);
                        
                        if (!E.empty()) {
                            cv::Mat R, t;
                            int inliers = cv::recoverPose(E, prev_pts, curr_pts, 
                                                        camera_matrix, R, t, mask);
                            
                            if (inliers > 10) {
                                // 위치 업데이트
                                camera_position += t * 0.1;
                                
                                float x = camera_position.at<double>(0);
                                float z = camera_position.at<double>(2);
                                trajectory.push_back(cv::Point2f(x, z));
                                
                                // 상태 정보 출력
                                std::cout << "Inliers: " << inliers 
                                         << ", Position: (" << x << ", " << z << ")";
                                if (use_yolo) {
                                    std::cout << ", Objects: " << current_detections.size();
                                }
                                std::cout << std::endl;
                            }
                        }
                    }
                    catch (const cv::Exception& e) {
                        std::cout << "Motion estimation failed: " << e.what() << std::endl;
                    }
                }
                
                // 매칭 결과 그리기
                cv::drawMatches(prev_frame, prev_keypoints, gray, keypoints, 
                              good_matches, result, cv::Scalar::all(-1), 
                              cv::Scalar::all(-1), std::vector<char>(), 
                              cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
            }
        }
        else {
            // 첫 프레임이거나 매칭 실패시 특징점만 표시
            cv::drawKeypoints(frame, keypoints, result, cv::Scalar(0, 255, 0));
        }
        
        // 🤖 딥러닝: YOLO 검출 결과 오버레이
        drawDetections(result);
        
        // 현재 프레임 정보 저장
        prev_frame = gray.clone();
        prev_keypoints = keypoints;
        prev_descriptors = descriptors.clone();
        
        // 궤적 정보 표시
        drawTrajectoryInfo(result);
        
        return result;
    }
    
    void drawDetections(cv::Mat& frame) {
        if (!use_yolo) return;
        
        for (const auto& det : current_detections) {
            // 바운딩 박스 그리기
            cv::rectangle(frame, det.bbox, cv::Scalar(0, 255, 255), 2);
            
            // 클래스 이름과 신뢰도 표시
            std::string label = det.class_name + " " + 
                              std::to_string((int)(det.confidence * 100)) + "%";
            
            int baseline;
            cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 
                                               0.5, 1, &baseline);
            
            cv::Point text_origin(det.bbox.x, det.bbox.y - 5);
            cv::rectangle(frame, text_origin + cv::Point(0, baseline),
                         text_origin + cv::Point(text_size.width, -text_size.height),
                         cv::Scalar(0, 255, 255), cv::FILLED);
            
            cv::putText(frame, label, text_origin, cv::FONT_HERSHEY_SIMPLEX, 
                       0.5, cv::Scalar(0, 0, 0), 1);
        }
    }
    
    void drawTrajectoryInfo(cv::Mat& frame) {
        // 궤적 정보를 프레임에 표시
        std::string info = "Points tracked: " + std::to_string(trajectory.size());
        cv::putText(frame, info, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 
                   0.7, cv::Scalar(0, 255, 0), 2);
        
        if (trajectory.size() > 1) {
            cv::Point2f current_pos = trajectory.back();
            std::string pos_info = "Pos: (" + std::to_string(current_pos.x).substr(0, 5) + 
                                  ", " + std::to_string(current_pos.y).substr(0, 5) + ")";
            cv::putText(frame, pos_info, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 
                       0.7, cv::Scalar(0, 255, 0), 2);
        }
        
        // 딥러닝 모델 상태 표시
        std::string dl_status = "DL: ";
        if (use_yolo) dl_status += "YOLO ";
        if (use_depth) dl_status += "Depth ";
        if (!use_yolo && !use_depth) dl_status += "None";
        
        cv::putText(frame, dl_status, cv::Point(10, 90), cv::FONT_HERSHEY_SIMPLEX, 
                   0.7, cv::Scalar(255, 255, 0), 2);
    }
    
    void showTrajectory() {
        if (trajectory.size() < 2) return;
        
        // 궤적을 그릴 이미지 생성
        int img_size = 600;
        cv::Mat traj_img = cv::Mat::zeros(img_size, img_size, CV_8UC3);
        
        // 궤적 스케일 조정
        float scale = 50.0f;
        cv::Point2f center(img_size/2, img_size/2);
        
        // 궤적 그리기
        for (size_t i = 1; i < trajectory.size(); i++) {
            cv::Point2f p1 = trajectory[i-1] * scale + center;
            cv::Point2f p2 = trajectory[i] * scale + center;
            
            // 범위 체크
            if (p1.x >= 0 && p1.x < img_size && p1.y >= 0 && p1.y < img_size &&
                p2.x >= 0 && p2.x < img_size && p2.y >= 0 && p2.y < img_size) {
                cv::line(traj_img, p1, p2, cv::Scalar(0, 255, 0), 2);
                cv::circle(traj_img, p2, 3, cv::Scalar(0, 0, 255), -1);
            }
        }
        
        // 원점 표시
        cv::circle(traj_img, center, 5, cv::Scalar(255, 0, 0), -1);
        cv::putText(traj_img, "Start", center + cv::Point2f(10, -10), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        
        cv::imshow("Camera Trajectory", traj_img);
        cv::waitKey(0);
        cv::destroyWindow("Camera Trajectory");
    }
    
    void printStats() {
        std::cout << "\n=== SLAM Statistics ===" << std::endl;
        std::cout << "Total trajectory points: " << trajectory.size() << std::endl;
        if (trajectory.size() > 1) {
            cv::Point2f start = trajectory.front();
            cv::Point2f end = trajectory.back();
            float distance = cv::norm(end - start);
            std::cout << "Total distance moved: " << distance << std::endl;
        }
    }
};

int main() {
    std::cout << "=== Deep Visual SLAM Demo (C++) ===" << std::endl;
    std::cout << "Controls:" << std::endl;
    std::cout << "  'q' - Quit" << std::endl;
    std::cout << "  't' - Show trajectory" << std::endl;
    std::cout << "  's' - Show statistics" << std::endl;
    std::cout << "  'd' - Toggle detection display" << std::endl;
    std::cout << "\nMove camera slowly for better tracking!" << std::endl;
    
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open camera!" << std::endl;
        return -1;
    }
    
    // 카메라 해상도 설정
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    
    DeepVisualSLAM slam;
    cv::Mat frame;
    bool show_detections = true;
    
    while (true) {
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Error: Empty frame!" << std::endl;
            break;
        }
        
        // SLAM 처리
        cv::Mat result = slam.processFrame(frame);
        
        // 결과 표시
        cv::imshow("Deep Visual SLAM Demo", result);
        
        // 키 입력 처리
        int key = cv::waitKey(1) & 0xFF;
        if (key == 'q' || key == 27) {  // 'q' 또는 ESC
            break;
        }
        else if (key == 't') {
            slam.showTrajectory();
        }
        else if (key == 's') {
            slam.printStats();
        }
        else if (key == 'd') {
            show_detections = !show_detections;
            std::cout << "Detection display: " << (show_detections ? "ON" : "OFF") << std::endl;
        }
    }
    
    cap.release();
    cv::destroyAllWindows();
    
    // 최종 결과 표시
    slam.printStats();
    slam.showTrajectory();
    
    return 0;
}