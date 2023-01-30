# Downstream task trainer/tester for huggingface pretrained s2s Language model

label을 auto-regressive 하게 뽑아내는 downstream task 학습기.

huggingface transformers 모델 중 s2s 사전학습 언어모델인 BART, T5, mT5, ByT5와 호환 됨

## 더 해야 할 일

  * data collator를 datamodule에 통합 (PTLM 학습과 달리 stochastic한 조작이 필요없음)
			--> (23/01/30) 하지 않는 것으로. 불필요함.
	* edit-distance based predict-label corrector를 datamodule에 통합, 새 메서드를 제공
			--> (23/01/30) 추가 작업 필요. 일단은 datamodule에 label map을 binding 하도록 구현함
	* 일부 텍스트 샘플로 huggingface datasets를 바로 생성하는 wrapper 모듈 정비
	* 학습/테스트/추론을 위한 데이터 연결을 프로그램 실행 인자로 할당하게 바꿔야 함
	* 번역 태스크 구현, 다국어, zero-shot 테스트 등 (e.g. en-jp, trained on ko/en, ko/jp, ko/zh)
	* adafactor 적용, scheduler의 경우 linear decaying 대신 cyclic lr이나 다른걸 사용해야 함 
	    --> (23/01/30) 완료
