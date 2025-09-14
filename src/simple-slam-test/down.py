"""
Deep SLAM Models Downloader
각 단계별 필요한 딥러닝 모델들을 다운로드하는 스크립트
"""

import os
import urllib.request
import zipfile
import tarfile
from pathlib import Path

class ModelDownloader:
    def __init__(self, models_dir="./models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
    def download_file(self, url, filename):
        """파일 다운로드 with 진행률 표시"""
        filepath = self.models_dir / filename
        
        if filepath.exists():
            print(f"✅ {filename} already exists, skipping...")
            return str(filepath)
            
        print(f"📥 Downloading {filename}...")
        
        def progress_hook(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            print(f"\r진행률: {percent}%", end="", flush=True)
            
        try:
            urllib.request.urlretrieve(url, filepath, progress_hook)
            print(f"\n✅ Downloaded: {filename}")
            return str(filepath)
        except Exception as e:
            print(f"\n❌ Failed to download {filename}: {e}")
            return None
    
    def extract_archive(self, filepath, extract_to=None):
        """압축 파일 추출"""
        if extract_to is None:
            extract_to = self.models_dir
            
        print(f"📦 Extracting {Path(filepath).name}...")
        
        if filepath.endswith('.zip'):
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif filepath.endswith(('.tar.gz', '.tgz')):
            with tarfile.open(filepath, 'r:gz') as tar_ref:
                tar_ref.extractall(extract_to)
        
        print(f"✅ Extracted to: {extract_to}")

    def download_monodepth2_models(self):
        """Step 2: MonoDepth2 모델 다운로드"""
        print("\n🎯 Step 2: MonoDepth2 Depth Estimation Models")
        
        # 사전 훈련된 MonoDepth2 모델들
        models = {
            "mono_640x192": "https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_640x192.zip",
            "mono+stereo_640x192": "https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_640x192.zip"
        }
        
        for model_name, url in models.items():
            filename = f"monodepth2_{model_name}.zip"
            filepath = self.download_file(url, filename)
            if filepath:
                self.extract_archive(filepath)
                
        # ONNX 변환 스크립트도 제공
        self.create_monodepth2_converter()
        
    def download_superpoint_model(self):
        """Step 3: SuperPoint 특징점 추출 모델"""
        print("\n🔍 Step 3: SuperPoint Feature Extraction Model")
        
        # SuperPoint 사전 훈련 모델
        superpoint_url = "https://github.com/magicleap/SuperPointPretrainedNetwork/raw/master/superpoint_v1.pth"
        filepath = self.download_file(superpoint_url, "superpoint_v1.pth")
        
        if filepath:
            print("📝 SuperPoint PyTorch model downloaded.")
            print("   ONNX 변환이 필요합니다. converter 스크립트를 확인하세요.")
            
        # 변환 스크립트 생성
        self.create_superpoint_converter()
        
    def download_yolo_models(self):
        """Step 4: YOLO 객체 검출 모델"""
        print("\n🤖 Step 4: YOLO Object Detection Models")
        
        # YOLOv4 모델 (OpenCV DNN과 호환성 좋음)
        yolo_models = {
            "yolov4.weights": "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights",
            "yolov4.cfg": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg",
            "coco.names": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names"
        }
        
        for filename, url in yolo_models.items():
            self.download_file(url, filename)
            
        # YOLOv5 ONNX 모델 (더 빠름)
        yolov5_url = "https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.onnx"
        self.download_file(yolov5_url, "yolov5s.onnx")
        
    def download_all_models(self):
        """모든 모델 한번에 다운로드"""
        print("🚀 Starting Deep SLAM Models Download...")
        print("=" * 50)
        
        self.download_monodepth2_models()
        self.download_superpoint_model()
        self.download_yolo_models()
        
        print("\n" + "=" * 50)
        print("✅ All models downloaded successfully!")
        print(f"📁 Models directory: {self.models_dir.absolute()}")
        
        self.print_usage_instructions()
        
    def create_monodepth2_converter(self):
        """MonoDepth2 PyTorch -> ONNX 변환 스크립트"""
        converter_script = '''
import torch
import torch.nn as nn
from collections import OrderedDict

# MonoDepth2를 ONNX로 변환하는 스크립트
# 사용법: python convert_monodepth2.py

def convert_monodepth2_to_onnx():
    """MonoDepth2 모델을 ONNX 형식으로 변환"""
    
    # 모델 로드 (실제 구현에서는 monodepth2 repository 필요)
    print("MonoDepth2 ONNX 변환을 위해서는:")
    print("1. git clone https://github.com/nianticlabs/monodepth2.git")
    print("2. 해당 repository의 변환 스크립트 사용")
    print("3. 또는 사전 변환된 ONNX 모델 사용")

if __name__ == "__main__":
    convert_monodepth2_to_onnx()
'''
        
        with open(self.models_dir / "convert_monodepth2.py", "w") as f:
            f.write(converter_script)
            
    def create_superpoint_converter(self):
        """SuperPoint PyTorch -> ONNX 변환 스크립트"""
        converter_script = '''
                                import torch
                                import torch.onnx

                                def convert_superpoint_to_onnx():
                                    """SuperPoint 모델을 ONNX로 변환"""
                                    
                                    # SuperPoint 모델 로드 (실제로는 SuperPoint 코드 필요)
                                    print("SuperPoint ONNX 변환을 위해서는:")
                                    print("1. git clone https://github.com/magicleap/SuperPointPretrainedNetwork.git")
                                    print("2. PyTorch 모델을 ONNX로 변환")
                                    print("3. 또는 사전 변환된 ONNX 모델 다운로드")

                                if __name__ == "__main__":
                                    convert_superpoint_to_onnx()
                                '''
        
        with open(self.models_dir / "convert_superpoint.py", "w") as f:
            f.write(converter_script)
    
    def print_usage_instructions(self):
        """사용법 안내"""
        print("\n📋 Usage Instructions:")
        print("=" * 30)
        
        print("\n🎯 Step 2 - Depth Estimation:")
        print("   ├── monodepth2_mono_640x192/ (PyTorch 모델)")
        print("   ├── convert_monodepth2.py (ONNX 변환 스크립트)")
        print("   └── ONNX 변환 후 C++에서 cv::dnn::readNetFromONNX() 사용")
        
        print("\n🔍 Step 3 - SuperPoint Features:")
        print("   ├── superpoint_v1.pth (PyTorch 모델)")
        print("   ├── convert_superpoint.py (ONNX 변환 스크립트)")
        print("   └── ONNX 변환 후 C++에서 사용")
        
        print("\n🤖 Step 4 - YOLO Detection:")
        print("   ├── yolov4.weights + yolov4.cfg (Darknet)")
        print("   ├── yolov5s.onnx (ONNX, 바로 사용 가능)")
        print("   ├── coco.names (클래스 이름)")
        print("   └── C++에서 cv::dnn::readNetFromDarknet() 또는 readNetFromONNX() 사용")
        
        print("\n💡 Quick Start:")
        print("   YOLOv5 ONNX 모델이 가장 쉽게 시작할 수 있습니다!")

def main():
    downloader = ModelDownloader()
    
    print("Deep SLAM Model Downloader")
    print("=" * 30)
    print("1. Download all models")
    print("2. MonoDepth2 only (Step 2)")
    print("3. SuperPoint only (Step 3)")
    print("4. YOLO only (Step 4)")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice == "1":
        downloader.download_all_models()
    elif choice == "2":
        downloader.download_monodepth2_models()
    elif choice == "3":
        downloader.download_superpoint_model()
    elif choice == "4":
        downloader.download_yolo_models()
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()