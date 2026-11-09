# 003. 데이터 분석 및 전처리

**프로젝트:** 온라인 쇼핑 구매 여부 예측 (PyTorch 신경망 분류)  
**작성 범위:** CP-01~CP-03

> Notebook에서 확인한 결과를 기록합니다. 각 `___`를 실제 값 또는 1~2문장으로 바꾸세요.

## 1. 데이터 탐색 및 클래스 분포 (CP-01)

- **데이터 크기:** 12330 행, 18 열
- **결측값 및 중복 행:** 결측값 없음, 중복행 125 (중복 행을 삭제하지 않은 이유: 일시와 사용자정보가 없는 건으로 중복이 가능한 log함)
- **수치형 특성 분포의 특징(`ProductRelated`, `PageValues` 등):** 대부분의 값이 낮은 구간에 집중되어 있고, 일부 큰 값이 존재하는 오른쪽 꼬리(right-skewed) 분포를 보인다. 특히 PageValues는 0인 값이 많아 분포가 매우 불균형한 특성을 나타낸다.
- **구매(`Revenue=1`) 건수와 비율:** 1908 건 / 15.47 %
- **미구매(`Revenue=0`) 건수와 비율:** 10422 건 / 84.53 %

**분석:** 모두 미구매로 예측했을 때 Accuracy가 약 84.53 %인 이유와 이 수치만으로 평가하기 어려운 점을 작성합니다.
정확도가 높은 구간으로 보이나, 구매건이 1908 건으로 존재하나, 구매건을 전혀 구분하지 못하기 때문에 정확도만으로 평가하기 어려움.

## 2. 계층화 데이터 분할 (CP-02)

- **Train:** 8630 건 / 구매 비율 15.48 %
- **Validation:** 1850 건 / 구매 비율 15.46 %
- **Test:** 1850 건 / 구매 비율 15.46 %
- **`stratify`를 적용한 이유:** 구매여부의 비율을 유지하면서 계층화분할하기 위함
- **1일차 회귀의 날짜 순서 분할과 다른 이유:** 1일차 공유자전거 대여여부는 미래 데이터가 학습에 미리 포함되는 데이터 누수(Data Leakage)를 방지가 필요하고, 금번 무작위 분할을 하면 각 데이터셋의 구매 비율이 달라질 수 있습니다.

**분석:** Train·Validation·Test의 구매 비율이 유사한지 확인하고 의미를 1~2문장으로 작성합니다.  
Train(15.48%), Validation(15.46%), Test(15.46%)의 구매 비율이 거의 동일하여 데이터가 균형 있게 분할되었음을 확인할 수 있다. 이는 stratify를 적용한 효과로, 각 데이터셋이 전체 데이터의 구매/비구매 분포를 잘 대표하므로 모델 학습 및 성능 평가의 신뢰성을 높여준다.

## 3. 구매 여부와 관련된 특성 분석 (CP-02)

- **방문자 유형(`VisitorType`)별 구매율:** Returning_Visitor 13.95 % / New_Visitor 25.2 % / Other 15.9 %
- **구매율이 가장 높은 방문자 유형:** New_Visitor
- **구매·미구매 세션의 `ProductRelated` 중앙값:** 29 / 16
- **구매·미구매 세션의 `ProductRelated_Duration` 중앙값:** 1069.658 / 517.580
- **구매·미구매 세션의 `PageValues` 중앙값:** 17.543 / 0.000
- **구매·미구매 세션의 `ExitRates` 중앙값:** 0.015 / 0.029
- **특성 분석을 Train 데이터에서 수행한 이유:** 모델이 아직 접하지 않은 Validation과 Test 데이터의 정보를 학습 및 분석 과정에서 미리 사용하지 않기 위해서입니다.

**분석:** 구매 여부와 관련된 특성에서 관찰한 점과 `PageValues`의 예측 시점상 한계를 작성합니다.  
구매 세션은 미구매 세션보다 ProductRelated와 ProductRelated_Duration의 중앙값이 높아 상품 관련 페이지를 더 많이, 더 오랫동안 탐색하는 경향을 보였다. 또한 PageValues 중앙값은 구매 세션에서 크게 높고 ExitRates는 더 낮아, 구매 고객일수록 구매 의도가 높고 사이트를 이탈할 가능성이 낮음을 시사한다.

다만 PageValues는 사용자의 행동 결과를 바탕으로 계산되는 지표이므로 예측 시점에 따라 실제 서비스 운영 환경에서는 사용할 수 없거나 정보 누수(Data Leakage)를 유발할 수 있다. 따라서 모델 구축 시에는 PageValues의 생성 시점과 활용 가능 여부를 반드시 검토해야 한다.

## 4. 데이터 전처리 및 PyTorch 입력 준비 (CP-03)

- **범주형 특성(One-Hot 인코딩):** "Month", "OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType"
- **수치형 특성(표준화):** "Administrative", "Administrative_Duration", "Informational", "Informational_Duration", "ProductRelated",
"ProductRelated_Duration", "BounceRates", "ExitRates", "PageValues", "SpecialDay"
- **이진 특성(변환 없이 사용):** "Weekend"
- **숫자로 표현된 코드값(`OperatingSystems`, `Browser` 등)을 범주형으로 처리한 이유:** 숫자 자체의 크기나 순서가 실제 의미를 나타내지 않고, 분류를 구분하기 때문입니다.
- **`handle_unknown="ignore"`를 설정한 이유:** Train 데이터에서 학습하지 못한 새로운 범주가 Validation 또는 Test 데이터에 등장하더라도 오류 없이 처리하기 위해서입니다.
- **Train 전처리 전후 Shape:** (8630, 17) → (8630, 73)
- **Validation에 적용한 변환 기준:** Train 전처리기준
- **PyTorch 입력·타깃 Tensor의 자료형 및 Shape:** 
  자료형: torch.float32 torch.float32
  Shape: Tensor: torch.Size([8630, 73]) torch.Size([8630, 1])
dtype: torch.float32 torch.float32
Mini-batches: 34
- **`DataLoader` 배치 크기와 배치 수:** 256 / 34

**분석:** 신경망 입력을 `float32` Tensor로 준비하고 미니배치를 만드는 이유를 1~2문장으로 작성합니다.  
신경망 모델은 행렬 연산을 수행하므로 입력 데이터를 PyTorch에서 효율적으로 처리할 수 있는 float32 Tensor로 준비하고,
또한 미니배치(Mini-batch) 학습을 통해 데이터를 일정 크기로 나누어 학습함으로써  학습 효율성을 높이기 위함.

## 5. 데이터 분석 요약

**모델 학습 전에 확인한 중요한 점 1~2가지:**  
구매여부는 class 불균형데이터로 계층화 분할하였으며, 신경망모델학습을 위해 Tensor 데이터유형으로 준비함.

> 불균형 데이터에서 **어떤 지표를 우선할지**는 `adr.md`의 의사결정 ③에 작성합니다.
