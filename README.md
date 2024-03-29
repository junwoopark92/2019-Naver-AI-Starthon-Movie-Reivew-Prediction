# 영화 리뷰를 이용한 평점 예측
## 리더보드
![leaderborad](https://user-images.githubusercontent.com/26558158/62705874-4d035400-ba29-11e9-89f0-bb56003fab9e.PNG)
## 문제설명
네이버 영화 리뷰와 평점 기반으로 학습하여 새로운 리뷰의 평점을 분류(예측)하는 문제입니다.

## dataset
Dataset: `16_tcls_movie`
* 데이터 소스 : 네이버 영화 사용자 리뷰 데이터
* 데이터 특성 : 영화 54,183편에 대한 사용자 리뷰 및 평점 데이터
* 영화 평점(label) : 0 ~ 10점
* \# of data
  * train : 약 10.5M
  * test : 약 1.3M

### data example
| Text | Label |
|:---:|:---:|
| 사람들 평점이 좋아 기대됨 | 7 |
| 이런 대작을 결코 놓쳐서는 안된다. 음악도 너무너무 좋다~ | 10 |
| 이런 배우들로 이런 퀄리티를 내놓다니. 이건 죄악아닌가? | 1 |
| 어느 누가 과연 마지막 장면에 미소 짓지 않을 수 있으리오...... | 9 |


## how to run
베이스라인 모델 학습 시작
```bash
nsml run -d 16_tcls_movie -e main.py
```

## ref
https://github.com/lime-robot/product-categories-classification
