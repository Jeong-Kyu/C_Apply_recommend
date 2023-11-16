from numpy import dot
from numpy.linalg import norm
import pandas as pd
import sklearn as skl
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def cos_sim(a, b):
    return dot(a,b)/(norm(a)*norm(b))

from scipy.stats import pearsonr
import numpy as np
def pearson_similarity(a, b):
    return np.dot((a - np.mean(a)), (b - np.mean(b))) / ((np.linalg.norm(a - np.mean(a))) * (np.linalg.norm(b - np.mean(b))))


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

new_df = pd.DataFrame(columns= comp_all.iloc[:,0], index=res_all.iloc[:,0])

for al in range(len(apply)):
    resumes = apply.iloc[al]['resume_seq']
    recruitments = apply.iloc[al]['recruitment_seq']
    new_df.at[resumes,recruitments] = 1
new_df = new_df.fillna(0)
print(new_df.shape) #(6695, 8530)
dict_df = {}
n=0
for k in range(len(new_df)):
    nk = new_df.iloc[k]
    name = nk.name
    value = nk[nk==1].index.values
    dict_df[f"{name}"] = value
    n+=1
    if n%10 == 0:
        print(n)

res_all = res_all.iloc[:,[0,3,4,5,6,11,13,17,18,19,20,21,24,25,26,27,28]]
res_all = res_all.fillna(0)
comp_all = comp_all.iloc[:,[0,1,2,3,4,5,6,7,8,10,11,12]]
comp_all = comp_all.fillna(0)
m=0
for k in range(len(res_all)): #len(res_all)
    cos_rank = pd.DataFrame(columns=["resume_seq", "cos_sim"])
    rk1 = res_all.iloc[k]
    for ks in range(len(res_all)):
        rk2 = res_all.iloc[ks]
        cos_num = pearson_similarity(rk1[1:].values,rk2[1:].values)
        dfn = rk2[0]
        dfn_df = pd.DataFrame({"resume_seq":[dfn],"cos_sim":[cos_num]})

        if len(cos_rank) < 6:
            cos_rank = cos_rank.append(dfn_df,ignore_index = True)
            cos_rank = cos_rank.sort_values(by='cos_sim' ,ascending=False)
        if cos_rank.iloc[-1]["cos_sim"] < cos_num:
            cos_rank = cos_rank[:10]
            cos_rank = cos_rank.append(dfn_df,ignore_index = True)
            cos_rank = cos_rank.sort_values(by='cos_sim' ,ascending=False)
            # print(cos_rank)
    final_list = []
    for resume_s in cos_rank['resume_seq'][1:]:
        final_list.extend(list(dict_df[f"{resume_s}"]))
    final_list = list(set(final_list))
    for f_s in dict_df[f"{cos_rank['resume_seq'][0]}"]:
        if f_s in final_list:
            final_list.remove(f_s)
    final_lists=final_list
    for fls in final_list:
        # print(rk1['degree'])
        # print(comp_all[comp_all['recruitment_seq']==fls])
        # print(comp_all[comp_all['recruitment_seq']==fls]['education'].values)
        if rk1['degree'] < comp_all[comp_all['recruitment_seq']==fls]['education'].values:
            final_lists.remove(fls)
    if len(final_lists) < 5:
        while len(final_lists) > 5:
            final_lists.append('R06059')
    s_index = submission[submission['resume_seq']==rk1[0]].index
    submission[s_index[0]:s_index[-1]+1]['recruitment_seq'] = final_lists[:5]
    m+=1
    print(f'{len(res_all)} // {m}')
submission.to_csv('final_submission1.csv',index=False)

# # columns=list(res_all.columns)+list(comp_all.columns)
# print(res_all.shape,comp_all.shape) #(8530, 17) (6695, 12)
# list_all = pd.DataFrame()
# # print(list_all)
# n=0
# for r in res_all.index[:]:
#     for c in comp_all.index[:]:"p  ㅡ                     "
#         ri = res_all.iloc[r]
#         ci = comp_all.iloc[c]
#         result = new_df.loc[ci['recruitment_seq'],ri['resume_seq']]
#         rc = pd.concat([ri,ci],axis=0)
#         rc['result'] = result
#         list_all[n]=rc
#         n += 1
#         if n % 1000 == 0:
#             print(n)
# list_all = list_all.transpose()
# rs_all = list_all['recruitment_seq']
# list_all.drop(['recruitment_seq'],axis=1,inplace=True)
# list_all.insert(loc=1, column='recruitment_seq', value=rs_all)
# print(list_all)
# list_all.to_excel("list_all.xlsx")
