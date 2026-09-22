#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!pip install lightgbm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
from scipy.stats import gaussian_kde
#from scipy.cluster.hierarchy import leaves_list, linkage
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import lightgbm as lgb
from lightgbm import plot_importance
import warnings
warnings.filterwarnings('ignore')

# ==================== 1. 全局配置（字体 + 美观） ====================
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['lines.linewidth'] = 2.5
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 13

sns.set_theme(style='whitegrid', context='talk', palette='viridis')
sns.set(font='Microsoft YaHei')
sns.set_style("whitegrid", {'font.sans-serif': ['Microsoft YaHei', 'SimHei']})

color_brand = '#2E86AB'
color_accent = '#F18F01'
color_success = '#2C9F6E'
color_diverging = sns.color_palette("coolwarm", 8)

# 可选：手动添加字体
#from matplotlib import font_manager
#import os
#font_path = r"C:\Windows\Fonts\msyh.ttc"
#if os.path.exists(font_path):
#    if 'Microsoft YaHei' not in [f.name for f in font_manager.fontManager.ttflist]:
#        font_manager.fontManager.addfont(font_path)
 #       print("已手动添加微软雅黑字体")
  #  else:
   #     print("微软雅黑字体已加载")
#else:
 #   print("警告：未找到微软雅黑字体文件，将使用系统默认中文字体")

# ==================== 2. ODS层：导入原始数据 ====================
file_path = r"C:\Users\Administrator\Desktop\天猫数据_1月到11月.xlsx"
ods = pd.read_excel(file_path)
print("=" * 60)
print(f"原始数据导入完成，数据行数：{ods.shape[0]}，列数：{ods.shape[1]}")
print("=" * 60)
#ods.to_excel(r"C:\Users\Administrator\Desktop\1_ODS原始数据.xlsx", index=False)

# 缺失值探查
print("\n【缺失值分析】")
missing_rate = ods.isnull().sum() / len(ods) * 100
missing_rate = missing_rate[missing_rate > 0].sort_values(ascending=False)

if len(missing_rate) > 0:
    fig_miss, axes_miss = plt.subplots(1, 2, figsize=(14, 5))
    axes_miss[0].barh(missing_rate.index, missing_rate.values, color='#ff6b6b')
    axes_miss[0].set_xlabel('缺失率 (%)')
    axes_miss[0].set_title('各字段缺失率分布', fontsize=12)
    axes_miss[0].axvline(x=5, color='blue', linestyle='--', label='5%阈值')
    axes_miss[0].legend()
    sns.heatmap(ods[missing_rate.index].isnull(), yticklabels=False, cbar=True, ax=axes_miss[1], cmap='viridis')
    axes_miss[1].set_title('缺失值分布热力图', fontsize=12)
    plt.tight_layout()
    plt.show()
    plt.close(fig_miss)
    print(f"共有{len(missing_rate)}个字段存在缺失值，最高缺失率：{missing_rate.iloc[0]:.2f}%")
else:
    print("无缺失值")

# ==================== 3. DWD层：数据清洗 ====================
dwd = ods.drop_duplicates()
num_cols = ['邮费', '最终促销价', '预估凑单价', '原价', '折后价', 
            '当日促销销量', '近2小时促销销量', '补贴金额', '优质品牌标识']
for col in num_cols:
    if col in dwd.columns:
        dwd[col] = dwd[col].fillna(0)
text_cols = ['品牌名称', '发货地', '店铺名称', '商品标题', '预估凑单说明', 
             '核心优惠规则', '更多优惠规则', '优惠标签', '所属品类']
for col in text_cols:
    if col in dwd.columns:
        dwd[col] = dwd[col].fillna("未知")
if '最终促销价' in dwd.columns and '邮费' in dwd.columns:
    dwd = dwd[(dwd['最终促销价'] >= 0) & (dwd['邮费'] >= 0)]
    dwd['邮费'] = dwd['邮费'].astype(int)
    dwd['最终促销价'] = dwd['最终促销价'].astype(float)
if '品牌名称' in dwd.columns:
    dwd = dwd[~dwd['品牌名称'].isin(['无品牌', '未知品牌'])]
    def is_dirty_brand(name):
        return bool(re.fullmatch(r'^\d+$', str(name))) or bool(re.match(r'^\d+.*', str(name)))
    dwd = dwd[~dwd['品牌名称'].apply(is_dirty_brand)]
print(f"\n【数据清洗完成】清洗后有效行数：{dwd.shape[0]}")
#dwd.to_excel(r"C:\Users\Administrator\Desktop\2_DWD清洗数据.xlsx", index=False)

# ==================== 3-1. 特征相关性热力图 ====================
print("\n【特征相关性分析】")
available_num_cols = [col for col in num_cols if col in dwd.columns and dwd[col].dtype in ['int64', 'float64']]
numeric_dwd = dwd[available_num_cols].replace([np.inf, -np.inf], np.nan).dropna()
if numeric_dwd.shape[1] >= 2:
    plt.figure(figsize=(12, 8))
    corr_matrix = numeric_dwd.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=0.5, cbar_kws={'shrink': 0.8})
    plt.title('特征相关性热力图', fontsize=14)
    plt.tight_layout()
    plt.show()
    plt.close()
    if '当日促销销量' in corr_matrix.columns:
        sales_corr = corr_matrix['当日促销销量'].sort_values(ascending=False)
        print("与『当日促销销量』相关性最高的特征：")
        print(sales_corr)

# ==================== 4. DWS层：数据汇总 ====================
if '品牌名称' in dwd.columns and '商品标题' in dwd.columns:
    dws_brand = dwd.groupby('品牌名称').agg(
        商品数量=('商品标题', 'count'),
        平均售价=('最终促销价', 'mean'),
        最高售价=('最终促销价', 'max'),
        最低售价=('最终促销价', 'min'),
        总销量=('当日促销销量', 'sum')
    ).reset_index()
if '所属品类' in dwd.columns:
    dws_category = dwd.groupby('所属品类').agg(
        商品数量=('商品标题', 'count'),
        平均售价=('最终促销价', 'mean'),
        总销量=('当日促销销量', 'sum')
    ).reset_index()
if '发货地' in dwd.columns:
    dwd['省份'] = dwd['发货地'].str.split(' ', expand=True)[0]
print("\n【DWS层汇总完成】")

#dws_brand.to_excel(r"C:\Users\Administrator\Desktop\3_DWS品牌汇总.xlsx", index=False)
#dws_category.to_excel(r"C:\Users\Administrator\Desktop\4_DWS品类汇总.xlsx", index=False)

# ==================== 7. ADS层：业务指标 ====================
ads_top10_brand = dws_brand.sort_values('商品数量', ascending=False).head(10)
ads_category_ratio = dws_category.copy()
ads_category_ratio['占比'] = ads_category_ratio['商品数量'] / ads_category_ratio['商品数量'].sum()
ads_overall = pd.DataFrame({
    '总商品数': [dwd.shape[0]],
    '总销量': [dwd['当日促销销量'].sum()],
    '整体均价': [dwd['最终促销价'].mean()],
    '包邮商品占比': [len(dwd[dwd['邮费'] == 0]) / len(dwd) * 100]
})
print("\n【ADS层业务指标】")
print(ads_overall)
#ads_overall.to_excel(r"C:\Users\Administrator\Desktop\5_ADS业务指标.xlsx", index=False)
#ads_top10_brand.to_excel(r"C:\Users\Administrator\Desktop\6_ADS_TOP10品牌.xlsx", index=False)
print("\n✅ 数仓各层已导出到桌面！")

# ==================== 9. 高级数据可视化 ====================
# 预处理：创建价格区间列
dwd['价格区间'] = pd.cut(
    dwd['最终促销价'], 
    bins=[0,20,50,100,200,500,1000, 5000],
    labels=['0-20元','20-50元','50-100元','100-200元','200-500元','500-1000元','1000元以上']
)

# 9.1 综合面板
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('电商商品数据分析面板', fontsize=16, fontweight='bold', y=0.98)

ax1 = axes[0, 0]
bars = sns.barplot(data=ads_top10_brand, y='品牌名称', x='商品数量', palette='viridis', ax=ax1)
ax1.set_title('TOP10品牌商品数量分布', fontsize=12)
ax1.set_xlabel('商品数量', fontsize=10)
ax1.set_ylabel('品牌名称', fontsize=10)
for bar in bars.patches:
    width = bar.get_width()
    ax1.text(width + 0.2, bar.get_y() + bar.get_height()/2,
             f'{int(width)}', ha='left', va='center', fontsize=9)

ax2 = axes[0, 1]
top_cat = ads_category_ratio.sort_values('商品数量', ascending=False).head(8)
other_sum = ads_category_ratio.sort_values('商品数量', ascending=False)[8:]['商品数量'].sum()
if other_sum > 0:
    other_df = pd.DataFrame({'所属品类': ['其他'], '商品数量': [other_sum]})
    plot_cat = pd.concat([top_cat, other_df], ignore_index=True)
else:
    plot_cat = top_cat
wedges, texts, autotexts = ax2.pie(
    plot_cat['商品数量'], labels=plot_cat['所属品类'], autopct='%1.1f%%',
    colors=sns.color_palette('pastel'), wedgeprops={'edgecolor': 'white', 'linewidth': 1}
)
ax2.set_title('商品品类占比分布', fontsize=12)
for autotext in autotexts:
    autotext.set_fontsize(9)

ax3 = axes[1, 0]
dwd_filtered = dwd[dwd['最终促销价'] <= 500]
sns.histplot(data=dwd_filtered, x='最终促销价', kde=True, color='#ff922b', ax=ax3)
ax3.set_title('商品价格分布直方图', fontsize=12)
ax3.set_xlabel('最终促销价', fontsize=10)
ax3.set_ylabel('商品数量', fontsize=10)

ax4 = axes[1, 1]
sns.scatterplot(data=dwd_filtered, x='最终促销价', y='当日促销销量', color='#51cf66', alpha=0.6, s=30, ax=ax4)
ax4.set_title('商品价格与销量关系散点图', fontsize=12)
ax4.set_xlabel('最终促销价', fontsize=10)
ax4.set_ylabel('当日促销销量', fontsize=10)

plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.show()
plt.close(fig)

# 9.2 价格区间销量面积图
price_sales = dwd.groupby('价格区间', observed=False)['当日促销销量'].sum().reset_index()
price_order = ['0-20元','20-50元','50-100元','100-200元','200-500元','500-1000元','1000元以上']
price_sales['价格区间'] = pd.Categorical(price_sales['价格区间'], categories=price_order, ordered=True)
price_sales = price_sales.sort_values('价格区间')
fig3, ax3 = plt.subplots(figsize=(12, 5))
ax3.fill_between(price_sales['价格区间'], price_sales['当日促销销量'], alpha=0.4, color=color_brand)
ax3.plot(price_sales['价格区间'], price_sales['当日促销销量'], marker='o', markersize=8,
         color=color_brand, linewidth=3, label='总销量')
max_idx = price_sales['当日促销销量'].idxmax()
ax3.plot(price_sales['价格区间'].iloc[max_idx], price_sales['当日促销销量'].iloc[max_idx],
         'o', markersize=12, color=color_accent, markeredgecolor='white', markeredgewidth=2)
ax3.annotate(f'峰值: {price_sales["当日促销销量"].iloc[max_idx]:,.0f}',
             xy=(price_sales['价格区间'].iloc[max_idx], price_sales['当日促销销量'].iloc[max_idx]),
             xytext=(10, 15), textcoords='offset points', fontsize=11,
             arrowprops=dict(arrowstyle='->', color=color_accent))
ax3.set_title('不同价格区间的总销量趋势', fontsize=18, pad=20)
ax3.set_xlabel('价格区间', fontsize=13)
ax3.set_ylabel('总销量', fontsize=13)
plt.xticks(rotation=30)
ax3.grid(True, alpha=0.3, linestyle='--')
sns.despine()
plt.tight_layout()
plt.show()
plt.close(fig3)

# 9.3 六边形分箱图 + 边际直方图
dwd_filtered = dwd[(dwd['最终促销价'] <= 500) & (dwd['当日促销销量'] <= 20000)]
plt.close('all')
fig4 = plt.figure(figsize=(12, 8))
gs = fig4.add_gridspec(4, 4, hspace=0.05, wspace=0.05)
ax_main = fig4.add_subplot(gs[1:, :-1])
ax_top = fig4.add_subplot(gs[0, :-1], sharex=ax_main)
ax_right = fig4.add_subplot(gs[1:, -1], sharey=ax_main)
hb = ax_main.hexbin(dwd_filtered['最终促销价'], dwd_filtered['当日促销销量'],
                    gridsize=40, cmap='YlOrRd', mincnt=1, alpha=0.8)
ax_main.set_xlabel('最终促销价 (元)', fontsize=12)
ax_main.set_ylabel('当日促销销量', fontsize=12)
ax_main.set_title('')
fig4.colorbar(hb, ax=ax_main, label='商品数量')
sns.histplot(dwd_filtered['最终促销价'], bins=30, color=color_brand, ax=ax_top, alpha=0.7)
ax_top.set_xlabel('')
ax_top.set_ylabel('频次')
ax_top.tick_params(axis='x', labelbottom=False)
ax_top.grid(axis='y', alpha=0.3)
ax_right.hist(dwd_filtered['当日促销销量'], bins=30, color=color_brand, alpha=0.7, orientation='horizontal')
ax_right.set_xlabel('频次')
ax_right.set_ylabel('')
ax_right.tick_params(axis='y', labelleft=False)
ax_right.grid(axis='x', alpha=0.3)
fig4.suptitle('价格-销量密度分布 (六边形分箱 + 边际分布)', fontsize=16, y=0.98)
plt.tight_layout()
plt.show()
plt.close(fig4)

# 9.4 双轴图
top10 = dws_brand.sort_values('商品数量', ascending=False).head(10)
fig5, ax5 = plt.subplots(figsize=(14, 7))
bars5 = ax5.bar(top10['品牌名称'], top10['商品数量'], color=color_brand, alpha=0.8, label='商品数量')
ax5.set_ylabel('商品数量 (件)', fontsize=13, color=color_brand)
ax5.tick_params(axis='y', labelcolor=color_brand)
ax5.set_xlabel('品牌名称', fontsize=13)
ax5_2 = ax5.twinx()
ax5_2.plot(top10['品牌名称'], top10['平均售价'], marker='D', markersize=8,
           color=color_accent, linewidth=2.5, label='平均售价')
ax5_2.set_ylabel('平均售价 (元)', fontsize=13, color=color_accent)
ax5_2.tick_params(axis='y', labelcolor=color_accent)
for bar in bars5:
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2, height + 3, f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
for i, price in enumerate(top10['平均售价']):
    ax5_2.text(i, price + 2, f'{price:.1f}', ha='center', va='bottom', fontsize=10, color=color_accent)
ax5.set_title('TOP10品牌: 商品数量 vs 平均售价', fontsize=18, pad=20)
ax5.set_xticklabels(top10['品牌名称'], rotation=45, ha='right')
fig5.tight_layout()
plt.show()
plt.close(fig5)


# 9.7 二维密度图 + 等高线
fig8, ax8 = plt.subplots(figsize=(10, 8))
x = dwd_filtered['最终促销价']
y = dwd_filtered['当日促销销量']
xy = np.vstack([x, y])
z = gaussian_kde(xy)(xy)
idx = z.argsort()
x_sorted, y_sorted, z_sorted = x.iloc[idx], y.iloc[idx], z[idx]
sc = ax8.scatter(x_sorted, y_sorted, c=z_sorted, cmap='plasma', s=10, alpha=0.6, edgecolors='none')
ax8.tricontour(x, y, z, levels=5, colors='white', linewidths=0.5)
ax8.set_xlabel('最终促销价 (元)', fontsize=13)
ax8.set_ylabel('当日促销销量', fontsize=13)
ax8.set_title('价格-销量联合分布密度图（含等高线）', fontsize=16)
cbar = plt.colorbar(sc, ax=ax8, label='密度')
cbar.ax.tick_params(labelsize=10)
sns.despine()
plt.tight_layout()
plt.show()
plt.close(fig8)

# 9.8 雷达图
radar_brands = ads_top10_brand.head(5)['品牌名称'].tolist()
brand_metrics = dwd[dwd['品牌名称'].isin(radar_brands)].groupby('品牌名称').agg({
    '当日促销销量': 'sum',
    '最终促销价': 'mean',
    '邮费': 'mean'
}).reset_index()
scaler_radar = MinMaxScaler()
scaled_vals = scaler_radar.fit_transform(brand_metrics[['当日促销销量', '最终促销价', '邮费']])
scaled_df = pd.DataFrame(scaled_vals, columns=['总销量', '平均售价', '平均邮费'])
scaled_df['品牌名称'] = brand_metrics['品牌名称']
angles = np.linspace(0, 2*np.pi, 3, endpoint=False).tolist()
angles += angles[:1]
fig9, ax9 = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
for i, row in scaled_df.iterrows():
    values = row[['总销量', '平均售价', '平均邮费']].tolist()
    values += values[:1]
    ax9.plot(angles, values, 'o-', linewidth=2, label=row['品牌名称'])
    ax9.fill(angles, values, alpha=0.1)
ax9.set_xticks(angles[:-1])
tick_labels = ['总销量', '平均售价', '平均邮费']
ax9.set_xticklabels(tick_labels, fontsize=12)
ax9.set_yticklabels([])
ax9.set_title('TOP5品牌 指标对比雷达图', fontsize=16, pad=20)
ax9.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
ax9.grid(True)
plt.tight_layout()
plt.show()
plt.close(fig9)

# ==================== 10. K-Means 聚类分析（改进版：为所有商品生成标签） ====================
print("\n" + "=" * 60)
print("【K-Means 聚类分析】")
print("=" * 60)

features = ['最终促销价', '当日促销销量', '邮费']
available_features = [f for f in features if f in dwd.columns]

# 1. 训练聚类模型（用无缺失数据的子集）
cluster_train_data = dwd[available_features].dropna().copy()
scaler = StandardScaler()
cluster_scaled_train = scaler.fit_transform(cluster_train_data)

# 肘部法
wcss = []
k_range = range(2, 9)
for k in k_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_temp.fit(cluster_scaled_train)
    wcss.append(kmeans_temp.inertia_)

fig_elbow, ax_elbow = plt.subplots(figsize=(10,4))
ax_elbow.plot(k_range, wcss, marker='o', color='#42a5f5', linewidth=2)
ax_elbow.set_title('肘部法确定最佳聚类数K', fontsize=12)
ax_elbow.set_xlabel('聚类数 K')
ax_elbow.set_ylabel('WCSS (簇内平方和)')
ax_elbow.grid(alpha=0.3)
ax_elbow.axvline(x=4, color='red', linestyle='--', label='选择 K=4')
ax_elbow.legend()
plt.tight_layout()
plt.show()
plt.close(fig_elbow)

best_k = 4
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
kmeans.fit(cluster_scaled_train)
kmeans_labels_train = kmeans.labels_
sil = silhouette_score(cluster_scaled_train, kmeans_labels_train)
print(f"K-Means (K={best_k}) 轮廓系数: {sil:.4f}")

# 2. 为所有商品生成聚类标签（填充缺失值后预测）
cluster_all = dwd[available_features].fillna(0)   # 缺失值极少，填0不影响
cluster_scaled_all = scaler.transform(cluster_all)
dwd['cluster_label'] = kmeans.predict(cluster_scaled_all)
print("\n聚类标签分布：")
print(dwd['cluster_label'].value_counts().sort_index())

# 3. 可视化聚类效果（基于训练数据）
cluster_data_vis = cluster_train_data.copy()
cluster_data_vis['cluster_label'] = kmeans_labels_train
fig_cluster, (ax_cl1, ax_cl2) = plt.subplots(1, 2, figsize=(14, 6))
sns.scatterplot(data=cluster_data_vis, x='最终促销价', y='当日促销销量', 
                hue='cluster_label', palette='Set1', alpha=0.6, ax=ax_cl1)
ax_cl1.set_title('K-Means聚类结果 (价格-销量)', fontsize=12)
ax_cl1.set_xlabel('最终促销价')
ax_cl1.set_ylabel('当日促销销量')
ax_cl1.legend(title='聚类')

pca = PCA(n_components=2)
pca_result = pca.fit_transform(cluster_scaled_train)
cluster_data_vis['pca1'] = pca_result[:, 0]
cluster_data_vis['pca2'] = pca_result[:, 1]
sns.scatterplot(data=cluster_data_vis, x='pca1', y='pca2', 
                hue='cluster_label', palette='Set1', alpha=0.6, ax=ax_cl2)
ax_cl2.set_title('K-Means聚类结果 (PCA降维)', fontsize=12)
ax_cl2.set_xlabel(f'PCA1 ({pca.explained_variance_ratio_[0]:.1%})')
ax_cl2.set_ylabel(f'PCA2 ({pca.explained_variance_ratio_[1]:.1%})')
ax_cl2.legend(title='聚类')
plt.tight_layout()
plt.show()
plt.close(fig_cluster)

# 4. 聚类特征画像（基于全量数据）
cluster_summary = dwd.groupby('cluster_label').agg(
    商品数量=('最终促销价', 'count'),
    平均价格=('最终促销价', 'mean'),
    平均销量=('当日促销销量', 'mean'),
    平均邮费=('邮费', 'mean')
).reset_index()
print("\n【聚类特征画像（全量商品）】")
print(cluster_summary.to_string(index=False))

# 箱线图
fig_box1, ax_box1 = plt.subplots(figsize=(10,6))
sns.boxplot(data=dwd, x='cluster_label', y='最终促销价', palette='coolwarm', ax=ax_box1)
ax_box1.set_title('不同聚类商品价格分布箱线图', fontsize=12)
ax_box1.set_xlabel('聚类类别')
ax_box1.set_ylabel('最终促销价')
ax_box1.grid(alpha=0.3)
plt.tight_layout()
plt.show()
plt.close(fig_box1)

fig_box2, ax_box2 = plt.subplots(figsize=(10,6))
sns.boxplot(data=dwd, x='cluster_label', y='当日促销销量', palette='coolwarm', ax=ax_box2)
ax_box2.set_title('不同聚类商品销量分布箱线图', fontsize=12)
ax_box2.set_xlabel('聚类类别')
ax_box2.set_ylabel('当日促销销量')
ax_box2.grid(alpha=0.3)
plt.tight_layout()
plt.show()
plt.close(fig_box2)

# ==================== 机器学习训练与评估（加入聚类标签） ====================
# 构造标签：是否高销量商品
median_sale = dwd['当日促销销量'].median()
dwd['is_hot'] = (dwd['当日促销销量'] > median_sale).astype(int)

# 原始数值特征
feature_cols = ['最终促销价', '原价', '折后价', '邮费', '补贴金额', '优质品牌标识']
X_base = dwd[feature_cols].fillna(0)

# 对聚类标签进行 one-hot 编码
cluster_dummies = pd.get_dummies(dwd['cluster_label'], prefix='cluster', dtype=int)
X = pd.concat([X_base, cluster_dummies], axis=1)

y = dwd['is_hot']

# 划分训练集/测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 定义模型
models = {
    "逻辑回归": LogisticRegression(max_iter=1000),
    "决策树": DecisionTreeClassifier(max_depth=5, random_state=42),
    "LightGBM": lgb.LGBMClassifier(random_state=42, verbose=-1)
}

print("\n" + "=" * 60)
print("【模型训练与评估】天猫商品高销量预测（加入聚类特征）")
print("=" * 60)

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)

    results[name] = {
        "准确率": round(acc, 3),
        "精确率": round(pre, 3),
        "F1": round(f1, 3),
        "AUC": round(auc, 3)
    }

res_df = pd.DataFrame(results).T
print(res_df)

print("\n【最优模型：LightGBM 详细结果】")
best_model = models["LightGBM"]
y_pred_best = best_model.predict(X_test)
print(classification_report(y_test, y_pred_best))

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['普通商品', '高销量商品'],
            yticklabels=['普通商品', '高销量商品'])
plt.title('LightGBM 混淆矩阵（含聚类特征）')
plt.xlabel('预测')
plt.ylabel('真实')
plt.show()

# 特征重要性
plt.figure(figsize=(10, 6), dpi=100)
ax = plot_importance(
    best_model,
    importance_type='gain',
    max_num_features=10,
    grid=False,
    color='#2E86AB'
)
plt.title('LightGBM 特征重要性（Top10，包含聚类特征）', fontsize=14)
plt.xlabel('重要性（信息增益）', fontsize=12)
plt.ylabel('特征', fontsize=12)
plt.tight_layout()
plt.savefig('lgb_feature_importance.png', bbox_inches='tight')
plt.show()

print("\n✅ 机器学习训练评估完成！")
print("\n" + "=" * 60)
print("全部分析完成！")
print("=" * 60)


# In[ ]:




