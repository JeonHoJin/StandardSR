
## Introduction
광운대학교 졸업작품

저해상도 이미지 고해상도 이미지 변환(Super-Resolution)

작품 설명, 시연은 [여기]()를 눌러주세요

## Development Enviroment
- Windows 7 64 bit,(and Windows 10)
- Python 3.5, tensorflow-gpu 1.2.1
- GTX 1080

## STD-model
![standard-model](/assets/standard-model.jpg)
#### summary
- `residual learning` - 저해상도로 변하며 사라진 고해상도 정보를 선별해 학습하는 효과
- `L2 -> L1 loss` - loss 값 계산시 분산값에 강인한 효과를 소폭 얻음
- `dropout` - 학습 정체기에 변화를 주어 모델 정확도 향상

## Training Flow chart
![training-flow](/assets/training-flow_ao0mco5br.jpg)

## Result Image
![resultX2](/assets/resultX2.JPG)

## Benchmark results
![model-benchmark](/assets/model-benchmark.jpg)
## How To Use
### Training
```py
# if start from training
python StandardSR.py
# if start from with a checkpoint
python StandardSR.py --model_path ./checkpoints/your_model.ckpt
```

### Testing
```py
# this will test a specific model through checkpoint which is set in TEST.py
python TEST.py
```

## Caution
  data파일 용량 문제로 저장소에는 <kbd>ignore로 제외</kbd>

 사용에 TEST.py에서 다음과 같은 경로로 사용중이니 주의 바랍니다

- Test.py
```py
DATA_PATH = "./data/test/"
path_dir = "./data/test/Set14/"	# Set5, Set14, B100, Urban100
result_path_dir = "./Result/"
```

## Reference
["Accurate Image Super-Resolution Using Very Deep Convolutional Networks", CVPR 16'.](http://cv.snu.ac.kr/research/VDSR/VDSR_CVPR2016.pdf) , **Author's paper**

["Jongchan/tensorflow-vdsr"](https://github.com/Jongchan/tensorflow-vdsr) , **Reference model**

["Image Super-Resolution Using Deep Convolutional Networks"](http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html) , **SRCNN paper**

["Train Data"](http://cv.snu.ac.kr/research/VDSR/train_data.zip) , **291 data set**

["Test Data"](http://cv.snu.ac.kr/research/VDSR/test_data.zip) , **Set14, Set5, Urban100, B100**
