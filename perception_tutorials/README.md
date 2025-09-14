# Sensing and Perception Tutorials with TurtleBot3

이 튜토리얼들은 대학원 **Sensing and Perception** 수업을 위한 ROS2 기반 실습 예제들입니다.

## 📚 학습 목표

### 1단계: 센서 기초 이해
- LIDAR 센서의 동작 원리와 데이터 구조
- 카메라 센서의 이미지 처리 기초
- 센서 데이터의 좌표계 변환

### 2단계: 신호 처리 및 전처리  
- 노이즈 제거 및 필터링
- 데이터 정규화 및 보간
- 센서 캘리브레이션

### 3단계: 센서 융합
- 다중 센서 데이터 결합
- Cross-modal 검증
- 로버스트 인식 시스템 구축

## 🚀 실행 방법

### 사전 준비
```bash
# TurtleBot3 시뮬레이션 실행
export TURTLEBOT3_MODEL=waffle_pi
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py

# 별도 터미널에서 RViz 실행
ros2 launch turtlebot3_bringup rviz2.launch.py
```

### Tutorial 1: LIDAR 기초
```bash
cd ~/ros2_ws
python3 perception_tutorials/01_lidar_basics.py
```
**학습 내용:**
- LIDAR 스캔 데이터 구조 분석
- 극좌표 → 직교좌표 변환  
- 기본 장애물 감지 알고리즘
- RViz 시각화

**예상 출력:**
```
📡 LIDAR Basics Tutorial Started!
📊 SCAN ANALYSIS:
   📐 Readings: 360
   📏 Angle range: 0.0° to 360.0°
   🎯 Resolution: 1.00° per reading
   📡 Range: 0.12m to 3.50m
📈 DATA QUALITY:
   ✅ Valid readings: 347/360 (96.4%)
   📊 Distance stats: min=0.50m, max=3.50m, avg=2.1m
   🎯 Closest obstacle: 0.50m
⚠️ Front obstacle at 0.75m!
```

### Tutorial 2: 카메라 기초
```bash
python3 perception_tutorials/02_camera_basics.py
```
**학습 내용:**
- 이미지 속성 분석 (해상도, 색상 공간)
- BGR, RGB, HSV 색상 공간 변환
- 기본 이미지 처리 (그레이스케일, 엣지 검출)
- 이미지 통계 분석

**RViz에서 확인:**
- Topic: `/camera/processed` - 처리된 4분할 이미지

### Tutorial 3: 센서 융합
```bash  
python3 perception_tutorials/03_sensor_fusion.py
```
**학습 내용:**
- LIDAR 포인트를 카메라 이미지에 투영
- 다중 센서를 이용한 장애물 감지
- 센서 간 데이터 연관성 분석
- 융합된 결과 시각화

**RViz에서 확인:**
- Topic: `/fusion_results` - 융합된 장애물 마커
- Topic: `/camera/overlay` - LIDAR 오버레이된 카메라 영상

## 🔬 실험 과제

### 과제 1: 센서 성능 비교
1. 다양한 거리에서 LIDAR 정확도 측정
2. 조명 조건에 따른 카메라 성능 분석  
3. 센서별 장단점 정리

### 과제 2: 알고리즘 개선
1. 노이즈 필터링 알고리즘 구현
2. 더 정확한 장애물 클러스터링
3. 동적 객체 추적

### 과제 3: 응용 시스템
1. 자율 주행을 위한 센서 융합
2. 실시간 SLAM 시스템 구축
3. 강화학습과 센서 데이터 통합

## 📊 데이터 수집 및 분석

### 로그 데이터 저장
```bash
# ROS2 bag 기록
ros2 bag record /scan /camera/image_raw /odom

# 데이터 분석
ros2 bag info <bag_file>
ros2 bag play <bag_file>
```

### 성능 메트릭
- **정확도**: 실제 거리 vs 센서 측정값
- **정밀도**: 반복 측정의 일관성  
- **지연시간**: 센서 데이터 처리 시간
- **강건성**: 노이즈 환경에서의 성능

## 🎯 고급 주제

### 1. 딥러닝 기반 인식
- YOLO를 이용한 객체 검출
- PointNet을 이용한 LIDAR 처리
- Sensor fusion with neural networks

### 2. SLAM 알고리즘
- FastSLAM 구현
- Graph-based SLAM
- Visual-Inertial SLAM

### 3. 실시간 시스템
- ROS2 실시간 처리
- 센서 동기화
- 병렬 처리 최적화

## 📚 참고 자료

- **논문**: "Probabilistic Robotics" by Thrun, Burgard, Fox
- **ROS2 공식 문서**: https://docs.ros.org/en/humble/
- **OpenCV 튜토리얼**: https://opencv.org/
- **PCL (Point Cloud Library)**: https://pointclouds.org/

## 🤝 기여 방법

1. Fork this repository
2. 새로운 튜토리얼 추가 또는 기존 개선
3. Pull Request 제출

## 📞 문의사항

- 수업 관련: 담당 교수님 또는 TA
- 기술적 문제: GitHub Issues
- 일반 문의: 이메일

---
**Happy Learning! 🎓🤖**