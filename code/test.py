import pandas as pd
import sklearn as skl
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

apply = pd.read_csv("dataset/apply_train.csv")

comp = pd.read_csv("dataset/company.csv")
rec = pd.read_csv("dataset/recruitment.csv")

re_edu = pd.read_csv("dataset/resume_education.csv")
re_lan = pd.read_csv("dataset/resume_language.csv")
resume = pd.read_csv("dataset/resume.csv")

re_cer = pd.read_csv("dataset/resume_certificate.csv")

submission = pd.read_csv("dataset/sample_submission.csv")

res_all = pd.merge(resume,re_edu,how='outer', on='resume_seq')
res_all = pd.merge(res_all,re_lan,how='outer', on='resume_seq')
res_all = res_all.drop_duplicates(subset=None, keep='first', inplace=False, ignore_index=True)
comp_all = pd.merge(comp,rec,how='outer',on='recruitment_seq')

new_df = pd.DataFrame(columns= res_all.iloc[:,0], index=comp_all.iloc[:,0])

for al in range(len(apply)):
    resumes = apply.iloc[al]['resume_seq']
    recruitments = apply.iloc[al]['recruitment_seq']
    new_df.at[recruitments,resumes] = 1
new_df = new_df.fillna(0)
print(new_df.shape) #(6695, 8530)

res_all = res_all.iloc[:,[0,3,4,5,6,11,13,17,18,19,20,21,24,25,26,27,28]]
res_all = res_all.fillna(0)
comp_all = comp_all.iloc[:,[0,1,2,3,4,5,6,7,8,10,11,12]]
comp_all = comp_all.fillna(0)

# columns=list(res_all.columns)+list(comp_all.columns)
print(res_all.shape,comp_all.shape) #(8530, 17) (6695, 12)
list_all = pd.DataFrame()
# print(list_all)
n=0
for r in res_all.index[:]:
    for c in comp_all.index[:]:
        ri = res_all.iloc[r]
        ci = comp_all.iloc[c]
        result = new_df.loc[ci['recruitment_seq'],ri['resume_seq']]
        rc = pd.concat([ri,ci],axis=0)
        rc['result'] = result
        list_all[n]=rc
        n += 1
        if n % 1000 == 0:
            print(n)
list_all = list_all.transpose()
rs_all = list_all['recruitment_seq']
list_all.drop(['recruitment_seq'],axis=1,inplace=True)
list_all.insert(loc=1, column='recruitment_seq', value=rs_all)
print(list_all)
list_all.to_excel("list_all.xlsx")

#         r= pd.Series(r)
#         # print(r)

#         # rc = pd.concat([pd.DataFrame(r),pd.DataFrame(c)],axis=1)
#         rc = pd.DataFrame(r)
    
#     # result.append(value)
#     print(rc)

# list_all = pd.DataFrame()
# for r in res_all:
#     for c in comp_all:
#         print(r,c)
#         rc = pd.concat([r,c])
#         print(rc)
#         list_all = list_all.append(r+c)

# col_all = list(res_all.columns)+list(comp_all.columns)
# col_df = pd.DataFrame(columns=col_all)
# print(col_df)
        
# apply_list = apply.copy()

# apply_list = pd.merge(apply_list,res_all,how='outer', on='resume_seq')
# apply_list = pd.merge(apply_list,comp_all,how='outer', on='recruitment_seq')

# apply_list.info()
# print(apply_list.shape)
