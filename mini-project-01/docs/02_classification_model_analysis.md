# 004. 모델 실험 및 오류 분석

**프로젝트:** 온라인 쇼핑 구매 여부 예측 (PyTorch 신경망 분류)  
**작성 범위:** CP-04~CP-08

> Notebook에서 확인한 실제 수치와 관찰을 기록합니다. `___`는 실제 값 또는 1~2문장으로 바꾸세요. Precision·Recall·F1은 구매 클래스(`Revenue=1`) 기준입니다.

## 1. 기준 모델 및 평가 지표 (CP-04)

- **양성 클래스:** 구매 (`Revenue=1`)
- **주 평가 지표:** F1 (구매 클래스)
- **보조 지표:** Precision, Recall, Accuracy
- **모델 비교에 사용한 데이터:** 훈련데이터
- **DummyClassifier — Validation Accuracy / F1:** 0.845 / 0.00
- **LogisticRegression — Train F1 / Validation F1:** 0.505 / 0.49
- **LogisticRegression — Validation Precision / Recall:** 0.734 / 0.367
- **LogisticRegression의 Validation 혼동행렬:** TN 1526 / FP 38 / FN 181 / TP 105

**분석:** 불균형 데이터에서 기준 모델의 Accuracy와 F1이 다르게 나타나는 이유를 작성합니다.  
DummyClassifier의 높은 Accuracy는 다수 클래스인 미구매 데이터에 편중된 결과이며, 구매 예측 성능을 평가하기 위해서는 구매 클래스의 F1-score를 주 평가 지표로 사용하는 것이 적절하다.

## 2. PyTorch MLP-A 구현 및 학습 (CP-05)

- **신경망 구조:** 입력 → 64 → 32 → 1 (ReLU, Dropout 0.1)
- **손실 함수 / 옵티마이저:** `BCEWithLogitsLoss` / Adam
- **학습 조건:** 최대 25 epoch, batch size 256, learning rate 0.001
- **Validation F1이 가장 높았던 에포크:** 25
- **선택된 MLP-A의 Train F1 / Validation F1:** 0.709 / 0.633
- **MLP-A Validation Precision / Recall:** 0.717 / 0.566

**분석:** Train loss와 Train·Validation F1 그래프에서 확인한 학습 경향을 작성합니다.  
Train loss는 epoch 4이전은 급격히 하락했고, 이후부터 경사가 완만해지기 시작했으며, F1 그래프에서는 epoch 4 이전은 급격시 상승했으나, 이후부터는 완만함

## 3. 클래스 가중 손실과 MLP 비교 (CP-06)

- **MLP-B 변경 사항:** 양성 클래스 가중치(`pos_weight`)를 적용한 `BCEWithLogitsLoss`
- **Train 데이터의 `pos_weight`(미구매 수 ÷ 구매 수):** Train 불균형 비율(neg/pos): 5.46
- **MLP-B의 최적 에포크:** 17
- **MLP-B Train F1 / Validation F1:** 0.692 / 0.645
- **MLP-B Validation Precision / Recall:** 0.575 / 0.734
- **MLP-A와 MLP-B의 Validation F1 차이:** A - B = 0.633 - 0.645 = - 0.0121
- **LogisticRegression과 두 MLP의 성능 비교에서 발견한 점:** LogisticRegression의 Validation F1은 0.490으로 MLP-A(0.633)와 MLP-B(0.645)보다 낮았다. MLP 모델이 구매 여부를 예측하는 데 더 높은 성능을 보였으며, 특히 클래스 가중치를 적용한 MLP-B가 가장 높은 F1-score를 기록하였다.
- **Train·Validation F1 차이에서 관찰한 점:** MLP-B의 Train F1은 0.692, Validation F1은 0.645로 차이는 0.047이다. 두 데이터의 성능 차이가 비교적 크지 않아 심각한 과적합 징후는 두드러지지 않는 것으로 판단된다.

**분석:** 양성 클래스 가중치를 적용한 뒤 Precision·Recall·F1이 어떻게 달라졌는지 작성합니다.  
양성 클래스 가중치를 적용한 결과 구매 클래스의 Recall이 높아져 실제 구매 고객을 더 많이 찾아낼 수 있게 되었다. 반면 Precision은 낮아져 실제로 구매하지 않을 고객을 구매한다고 잘못 예측하는 경우가 증가하였다. 그러나 Precision과 Recall의 균형을 나타내는 F1-score는 0.633에서 0.645로 상승하였다. 따라서 클래스 가중치 적용이 구매 고객 탐지 성능 개선에 긍정적인 효과를 보였다고 판단할 수 있다.

> 전체 실험 결과는 Notebook의 `results_df`와 `results/02_model_comparison.csv`를 참고합니다.

## 4. 예측 오류 및 임계값 분석 (CP-07)

- **비교한 두 신경망:** MLP-A / MLP-B
- **기본 임계값 0.5에서 MLP-A의 FP / FN:** 64 / 124
- **기본 임계값 0.5에서 MLP-B의 FP / FN:** 155 / 76
- **임계값 0.3에서 Precision·Recall 및 FP·FN 변화:** Precision 0.109 감소, Recall 0.112 증가 및 FP 61증가, FN 32 감소
- **임계값 0.7에서 Precision·Recall 및 FP·FN 변화:** Precision 0.118 증가, Recall 0.113 감소 및 FP 44 감소, FN 61 증가
- **Validation F1이 가장 높은 MLP·임계값 조합:** MLP-B (weighted BCE), 임계값 0.5
- **선택한 MLP의 최적 에포크 수:** 17
- **(선택) PR 곡선 또는 추가 임계값 실험에서 확인한 점:** ___

**분석:** 모델과 임계값의 선택에서 Precision·Recall 및 FP·FN을 어떻게 고려했는지 작성합니다.  
구매 고객 탐지 능력과 Precision·Recall의 균형을 고려하여 Validation F1이 가장 높은 MLP-B와 임계값 0.5를 선택하였다. 최적 에포크는 17로 설정하여 최종 모델 재학습에 활용한다.
임계값을 0.3으로 낮추면 Recall이 증가하고 FN이 감소하여 구매 고객 탐지에 유리하지만, FP 증가와 Precision 감소라는 단점이 발생한다. 반대로 임계값을 0.7로 높이면 Precision이 증가하고 FP가 감소하지만, FN이 증가하여 실제 구매 고객을 놓치는 경우가 많아진다.

> 임계값별 상세 결과는 `threshold_df`와 `results/02_threshold_comparison.csv`를 참고합니다.

## 5. 최종 신경망과 Test 평가 (CP-08)

- **Test 확인 전에 확정한 MLP 설정 / 임계값:** MLP-B (weighted BCE) / 임계값: 0.5
- **최종 재학습 데이터(Train + Validation):** 10480 행 / 최종 epoch: 17
- **최종 재학습 에포크 수 / 가중 손실 적용 여부:** 17 / 적용
- **Test Accuracy / Precision / Recall / F1:** 0.859 / 0.527 / 0.843 / 0.843
- **Test 혼동행렬:** TN 1348 / FP 216 / FN 45 / TP 241
- **모델 번들(`02_final_classifier.joblib`)·설정 JSON 저장 및 재로딩 확인:** 저장한 결과 파일:
- 02_final_classifier.joblib
- 02_model_comparison.csv
- 02_model_config.json
- 02_threshold_comparison.csv
- **`predict_purchase()` 실행 결과:** 0 ({'purchase_probability': 0.0058, 'prediction': 0, 'threshold': 0.5})

**해석:** 최종 Test 결과와 동일한 모델·임계값의 Validation 결과를 비교해 알게 된 점을 작성합니다.  
최종 Test 결과 Accuracy는 0.859, Precision은 0.527, Recall과 F1은 0.843으로 나타났다. Test F1이 Validation F1보다 높게 나타났지만, 이는 데이터 분할에 따른 차이일 수 있으므로 Test 결과만으로 모델의 일반화 성능이 항상 더 높다고 판단할 수는 없다.

> Test 결과는 모델 선택에 사용하지 않습니다. 최종 Test 수치는 Notebook과 이 문서에만 기록합니다.

## 6. 실험 요약

- **가장 중요한 실험 결과:**  Test F1이 Validation F1보다 높게 나타났지만, 이는 데이터 분할에 따른 차이일 수 있으므로 Test 결과만으로 모델의 일반화 성능이 항상 더 높다고 판단할 수는 없다.
- **남아 있는 모델의 한계:** Precision이 0.527로 상대적으로 낮아 구매로 예측한 세션 중 실제 미구매 세션도 상당수 존재한다. 또한 특정 데이터 분할에서 얻은 성능이므로 다른 기간이나 새로운 고객 데이터에서도 동일한 성능이 유지되는지는 추가 검증이 필요하다.

> **최종 MLP 설정·임계값의 선택과 근거**는 `adr.md`의 의사결정 ④에 작성합니다.
