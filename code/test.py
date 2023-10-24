import pandas as pd
import sklearn as skl
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

apply = pd.read_csv("dataset/apply_train.csv")
comp = pd.read_csv("dataset/company.csv")
rec = pd.read_csv("dataset/recruitment.csv")
re_cer = pd.read_csv("dataset/resume_certificate.csv")
re_edu = pd.read_csv("dataset/resume_education.csv")
re_lan = pd.read_csv("dataset/resume_language.csv")
resume = pd.read_csv("dataset/resume.csv")
submission = pd.read_csv("dataset/sample_submission.csv")


res_all = pd.merge(resume,re_cer,how='outer', on='resume_seq')
res_all = pd.merge(res_all,re_edu,how='outer', on='resume_seq')
res_all = pd.merge(res_all,re_lan,how='outer', on='resume_seq')
# for a in res_all.keys():
#     print(res_all[a].value_counts())

comp_all = pd.merge(comp,rec,how='outer',on='recruitment_seq')
# print(comp)
comp_all.info()
# for a in comp.keys():
#     print(comp[a].value_counts())
comp = comp_all.iloc[:,[1,2,3,4,5,6,7,8,10,11,12]]
comp = comp.fillna(0)
print(comp)
from sklearn.cluster import KMeans # model
import matplotlib.pyplot as plt # 시각화

scaler = StandardScaler()
# 데이터 학습
scaler.fit(comp)
# 변환
scaler_data = scaler.transform(comp)

pca = PCA(n_components = 2)

pca.fit(scaler_data)

data2 = pd.DataFrame(data = pca.transform(scaler_data), columns=['pc1', 'pc2'])

print(data2.head())
x = []   # k 가 몇개인지 
y = []   # 응집도가 몇인지 

for k in range(1, 30):
    kmeans = KMeans(n_clusters = k)
    kmeans.fit(data2)
    
    x.append(k)
    y.append(kmeans.inertia_)
plt.plot(x, y)
plt.show()
model = KMeans(n_clusters=7, random_state=0, algorithm='auto')
# n_clusters=3 : 군집의 개수 (k) (이미 알고 있음)
# random_state=0 : seed 역할 (모델을 일정하게 생성 = 랜덤X)
model.fit(data2)
# # 3. 클러스터링(군집) 결과
pred = model.predict(data2)
pred
len(pred) # 150 (관측치 개수만큼 예측치 생성됨)
# 4. 군집결과 시각화
plt.scatter(x=data2['pc1'], y=data2['pc2'], c=pred)
plt.show()

