import pandas as pd

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
print(res_all)

comp_all = pd.merge(comp,rec,how='outer',on='recruitment_seq')
print(comp_all)