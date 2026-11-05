# 002. 모델 실험 및 오류 분석

**프로젝트:** 공유자전거 대여 수요 예측 (회귀)  
**작성 범위:** CP-04~CP-08

> Notebook에서 확인한 실제 성능과 관찰을 작성합니다. 각 항목의 `___`를 실제 값이나 1~2문장으로 바꾸세요.

## 1. 모델 평가 기준 (CP-04)

- **주 평가 지표:** MAE
- **보조 지표:** RMSE, R²
- **모델 비교에 사용한 데이터:** 공유자전거 시간대별 대여 수량, 훈련/검증 데이터
- **MAE를 주요 지표로 선택한 이유:** 자전거 대여량 예측에서는 예측값이 실제 대여량과 평균적으로 몇 대 정도 차이 나는지 직관적으로 이해할 수 있기 때문에 MAE를 주요 평가 지표로 선택하였다.

## 2. 모델별 실험 결과 (CP-04~CP-06)

대표 설정별 Train MAE / Validation MAE를 기록합니다. 전체 실험 결과는 Notebook의 `results_df`에서 확인합니다.

- **DummyRegressor (mean):** Train 119.572 / Validation 194.518
- **LinearRegression:** Train 68.195 / Validation 114.140
- **DecisionTreeRegressor (`max_depth=12`):** Train 17.4799 / Validation 55.748
- **RandomForestRegressor (설정 "n_estimators": 50, "max_depth": 10):** Train 22.840 / Validation 46.708

**실험에서 발견한 점**

- **Decision Tree 깊이 변화에 따른 성능:** 트리의 깊이가 길어질수록 MAE가 감소하는 경향
- **Random Forest 두 설정의 차이:** RandomForest_B는 학습 데이터에 대한 적합도가 높지만, 실제 예측 성능에서는 RandomForest_A가 조금 더 우수했다.
- **Train·Validation 성능 차이에서 관찰한 현상:** 대부분의 모델에서 Train MAE보다 Validation MAE가 높게 나타났다. 과적합가능성이 있음
 (`001_data_analysis`의 2절 `cnt` 평균 차이와 연결해 작성)

## 3. 예측 오류 분석 (CP-07)

- **비교한 후보 모델 2개(Validation MAE 상위 2개):** RandomForest_A / RandomForest_B
- **실제값·예측값 그래프에서 관찰한 점:** RandomForest_A가 잔차가 작게 나타남
- **시간대별 MAE가 큰 구간:** 17 ~ 19시 구간
- **모델이 상대적으로 잘 예측하지 못하는 상황:** RandomForest_B가 다수의 시간대에서 퇴근시간 대량의 대여가 발생하는 경우 오히려 상대적으로 예측을 많이 벗어남 
- **(선택) 특성 중요도에서 확인한 점:** 모델 특이구간이 상대적으로 작은 RandomForest_A

**분석:** 이 오류가 최종 모델 선택에 어떤 영향을 주었는지 작성합니다.  
예측이 특이구역에서 많이 벗어나는 것은 예측성능을 저하로 나타나 전반적으로 차이가 작은 모델을 선택함

## 4. 최종 모델과 Test 평가 (CP-08)

- **Test 확인 전에 선택한 최종 모델·설정:** RandomForest_A
- **재학습에 사용한 데이터(Train + Validation) 크기:** 14777행
- **Test MAE:** 56.24
- **Test RMSE:** 84.46
- **Test R²:** 0.8446
- **모델 저장·재로딩 및 예측 결과:** 저장전후 예측일자 일치

**해석:** 최종 Test 결과로 알게 된 점을 1~2문장으로 작성합니다.  
___

> Test는 최종 평가용입니다. 서로 다른 데이터 집합에서 나온 점수를 직접 개선율로 비교하지 않습니다.

## 5. 실험 요약

- **가장 중요한 실험 결과:** ___
- **남아 있는 모델의 한계:** ___

> 최종 모델을 **왜 선택했는지**는 `adr.md`의 의사결정 ②에 작성합니다.
