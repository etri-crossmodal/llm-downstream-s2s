# Downstream task trainer/tester for huggingface pretrained s2s Language model

label을 auto-regressive 하게 뽑아내는 downstream task 학습기.

huggingface transformers 모델 중 s2s 사전학습 언어모델인 BART, T5, mT5, ByT5와 호환 됨

Copyright (C) 2023, Jong-hun Shin. ETRI LIRS.

## 실행/개발 환경 구성 방법

### (optional, pip로 대체 가능) conda 실행환경 구성 및 pytorch 설치
(1) conda 실행이 불가능할 경우, https://github.com/conda-forge/miniforge 에서 각 OS/플랫폼에 맞는 Miniforge3을 설치 후 터미널/셸을 재시작하여 활성화하고, 아래 명령어를 사용해 자신에게 맞는 python 실행 환경을 생성합니다:

```
$ conda create -n kebyt5-dev python=3.8
$ conda activate kebyt5-dev 
```
-n 옵션의 kebyt5-dev 라는 이름은 환경 명으로, 원하는대로 변경할 수 있습니다. activate 명령어를 통해서 해당 환경을 활성화할 수 있습니다.

(2) llm-downstream-s2s 디렉터리로 이동, 다음 명령어를 수행하여 환경에 사용될 패키지를 설치합니다. 다음의 명령어를 사용해 패키지를 설치합니다:
```
$ conda env update -n kebyt5-dev --file conda-environment.yaml
```

conda를 사용하지 않는 경우, pip를 사용하여 pytorch 1.12 버전 이상을 설치합니다. pytorch.org의 설치 방법을 참고하십시오.

### pip로 추가 의존성 해소

다음 명령어를 사용하여 requirements.txt에 기재된 의존성을 해소합니다.

```
$ pip install -r requirements.txt
```

### 실행 방법
Tab으로 구분된 학습데이터 (입력<탭>정답<엔터>) 구성을 갖는 텍스트 파일을 통한 학습기 실행 방법은 다음과 같습니다:

run_script/train_s2s_kebyt5-small.sh 파일을 참조해도 좋습니다.

```bash
python train.py -task seq2seq \
  -train_data [학습 데이터 텍스트 파일 경로] \
  { -valid_data [검증 데이터 텍스트 파일 경로, 없으면 생략 가능] } \
  -valid_data_proportions [숫자, 예: 0.05. -valid_data 옵션으로 검증 데이터를 따로 입력시 생략 가능.] \
  -save_path [학습 모델 및 로그 저장 위치, 없으면 자동으로 디렉터리가 생성됨] \
  -init_model [ 사전학습 모델 파일 경로. pytorch_model.bin이 들어있는 디렉터리를 지정하면 됩니다. ] \
  -max_epoch [최대 학습 epoch 수, 기본값 4] \
  -optimizer [adam, adafactor 둘 중 하나. 예: adafactor] \
  -learning_rate [adam 옵티마이저의 학습률. 기본 최적화 알고리즘인 adafactor를 사용하는 경우 자동으로 지정되며, 입력한 값은 무시됩니다.] \
  -gpus [사용될 GPU 수] -strategy ddp \
  -float_precision [연산 정밀도, 기본값 32. NVIDIA Ampere 급 이상을 사용하는 경우 16 지정 가능. 그 미만 카드에서는 16 지정시 NaN loss가 나올 수 있음] \
  -grad_acc [gradient accumulation, 기본값 1] -batch_size [배치크기.]

```
#### 이어서 학습하는 방법
중단한 checkpoint로 부터 학습을 재개하려면, ``-resume_checkpoint [다시 시작하고 싶은 checkpoint 파일]`` 옵션을 지정하여 학습을 재개할 수 있습니다.

#### 추론 테스트 방법
```bash
python inference.py \
  -model [학습된 모델 checkpoint] \
  -float_precision [연산 정밀도, 기본값 32.] \
  -gpus 1 -batch_size [배치 크기, VRAM 크기에 맞게 조정] -task [태스크 명, task_utils.py 참조.]
```

탭으로 구분된 평가 데이터를 통한 평가 방법은 다음과 같습니다:
```bash
python inference.py -task seq2seq \
  -test_data [테스트 데이터 파일 명] \
  -model [.ckpt로 끝나는 학습 체크포인트 파일] \
  -tokenizer [학습 당시의 init_model 옵션으로 넣은 모델 위치 또는 HF 모델 명. e.g. google/byt5-small] \
  -gpus 1 -float_precision 32 \
  -save_output [출력결과를 저장할 파일 이름] -batch_size 128 \
  -beam_size [beam 크기. 기본값 1., 번역 등은 2~5 사이의 값을 사용 OK.] \
  -max_predict_length [최대 추론 길이. 기본값 512. 레이블 추정의 경우 64~128로 설정하면 됨.]
```

#### 실행 예시
실행 예시로, nsmc 분류 테스트 샘플을 run_scripts 안에 포함하였습니다.
  * SKT kobart-v2를 사용한 방법은 train_nsmc_skt-kobart-v2.sh 파일을,
  * kebyt5-* 모델이 있으면 그에 맞게 train_nsmc_kebyt5-small.sh 파일 등을 살펴보시면 됩니다.
  * 추론 예시는 nsmc_test.sh 파일을 참조하십시오.

tab으로 구분된 데이터의 학습, 평가는 다음의 스크립트를 참조하십시오:
  * 학습 - run_script/train_s2s_kebyt5-small.sh
  * 평가(추론) - run_script/inference_s2s_kebyt5-small.sh


## 더 해야 할 일
  * 공개 라이선스 결정
  * data collator를 datamodule에 통합 (PTLM 학습과 달리 stochastic한 조작이 필요없음) --> (23/01/30) 하지 않는 것으로. 불필요함.
  * edit-distance based predict-label corrector를 datamodule에 통합, 새 메서드를 제공 --> (23/01/30) 추가 작업 필요. 일단은 datamodule에 label map을 binding 하도록 구현함
  * 일부 텍스트 샘플로 huggingface datasets를 바로 생성하는 wrapper 모듈 정비 --> (23/02/21) 완료
  * 학습/테스트/추론을 위한 데이터 연결을 프로그램 실행 인자로 할당하게 바꿔야 함
  * 번역 태스크 구현, 다국어, zero-shot 테스트 등 (e.g. en-jp, trained on ko/en, ko/jp, ko/zh) --> tab으로 구분된 데이터를 처리하는 -task seq2seq 구현으로 대체
  * adafactor 적용, scheduler의 경우 linear decaying 대신 cyclic lr이나 다른걸 사용해야 함 --> (23/01/30) 완료
  * truncate/discard를 max_seq_length와 결합. max_seq_length가 의도대로 작동하게 수정 필요. --> 현재 discard만 작동.
  * multi-gpu를 사용한 추론 루틴 구현, 테스트, uneven dataset의 distributed sampler 호환 등
