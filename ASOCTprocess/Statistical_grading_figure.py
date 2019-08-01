"""
This file is aims to plot figure for zhongshan
"""
import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.optimize as opt
import seaborn as sns
import pickle as pkl

import matplotlib.pyplot as plt
plt.switch_backend('agg')

df = pd.read_csv('data.csv')
# ----------------------------------------
# Two-side test for null hypothesis that 2 independent samples
# have identical average
# ----------------------------------------

# stat_val, p_val = stats.ttest_ind(nucleus_density_sick,nucleus_density_health, equal_var=False)
# print ('Two-sample t-statistic for nucleus: t-statistic = %f, p-value = %f' % (stat_val, p_val))
#
# stat_val, p_val = stats.ttest_ind(cortex_density_sick,cortex_density_health, equal_var=False)
# print ('Two-sample t-statistic for cortex: t-statistic = %f, p-value = %f' % (stat_val, p_val))
#
# stat_val, p_val = stats.ttest_ind(lens_density_sick,lens_density_health, equal_var=False)
# print ('Two-sample t-statistic for lens: t-statistic = %f, p-value = %f' % (stat_val, p_val))

# set the default setting of figure
df.rename(columns={ df.columns[0]:'tmp' }, inplace=True)
df.drop(columns=['tmp'],inplace=True)

df_sick = df.iloc[:,:][df['h_type']=='sick']
df_health = df.iloc[:,:][df['h_type']=='health']

df.to_csv('/home/intern1/qiuzhen/Works/test/lensclassify/figure/Total_data.csv')
df.describe().to_csv('/home/intern1/qiuzhen/Works/test/lensclassify/figure/Total_data_describe.csv')
df_sick.to_csv('/home/intern1/qiuzhen/Works/test/lensclassify/figure/Sick_data.csv')
df_sick.describe().to_csv('/home/intern1/qiuzhen/Works/test/lensclassify/figure/Sick_data_describe.csv')
df_health.to_csv('/home/intern1/qiuzhen/Works/test/lensclassify/figure/Health_data.csv')
df_health.describe().to_csv('/home/intern1/qiuzhen/Works/test/lensclassify/figure/Health_data_describe.csv')



sns.set(style="whitegrid")
plt.figure(figsize=(30,10))  # fix and change default image size
plt.xlabel('Tpye')
plt.ylabel('Pixel loss')

f, ax= plt.subplots(figsize = (30, 15))
ax.set_title('Pixel loss of all data in different Parts')
sns.violinplot(data=df)
plt.savefig('/home/intern1/qiuzhen/Works/test/lensclassify/figure/Total.png')
plt.clf()

f, ax= plt.subplots(figsize = (30, 15))
ax.set_title('Pixel loss of health data in different Parts')
sns.violinplot(data=df_health)
plt.savefig('/home/intern1/qiuzhen/Works/test/lensclassify/figure/Health.png')
plt.clf()

f, ax= plt.subplots(figsize = (30, 15))
ax.set_title('Pixel loss of sick data in different Parts')
sns.violinplot(data=df_sick)
plt.savefig('/home/intern1/qiuzhen/Works/test/lensclassify/figure/Sick.png')
plt.clf()

statical_name = ['lens', 'cortex',  'nucleus',
                 'lens_up', 'lens_down','cortex_up', 'cortex_down',
                 'nucleus_up', 'nucleus_down']

# print(df)
# df = df[df['h_type']=='sick'] & df[df['h_type']=='health']
# df = df.drop(df['h_type']=='uncertain')

for tmp_name in statical_name:
    stat_val, p_val = stats.ttest_ind(df_sick[tmp_name], df_health[tmp_name], equal_var=False)
    print ('Two-side t-statistic for %s: t-statistic = %f, p-value = %f' % (tmp_name,stat_val, p_val))
    f, ax = plt.subplots(figsize=(15, 15))
    ax.set_title('Two-sample t-statistic for %s: t-statistic = %f, p-value = %f' % (tmp_name,stat_val, p_val))
    sns.violinplot(data=df, x='h_type', y=tmp_name, scale="count", scale_hue=True)
    plt.savefig('/home/intern1/qiuzhen/Works/test/lensclassify/figure/%s.png'%tmp_name)
    plt.clf()


# f, ax= plt.subplots(figsize = (14, 10))
# ax.set_title('Correlation between features')
# sns.violinplot(data=df, x='h_type', y='lens_down', scale="count",scale_hue=True)
# plt.savefig('../figure/lens.png')
# plt.clf()
#
# plt.text(-3, 40, "function: y = x * x", size = 15,\
#          family = "fantasy", color = "r", style = "italic", weight = "light",\
#          bbox = dict(facecolor = "r", alpha = 0.2))
#
# plt.show(sns.violinplot(data=df, x='h_type', y='nucleus'))
#
#
# plt.savefig('../figure/nucleus.png')
