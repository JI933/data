import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import matplotlib.font_manager as fm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# ==================== 字体设置 ====================
font_names = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS', 'PingFang SC']
available_fonts = [f.name for f in fm.fontManager.ttflist]
chinese_font = next((f for f in font_names if f in available_fonts), 'DejaVu Sans')
print(f"使用字体: {chinese_font}")

sns.set(font=chinese_font, style="white", font_scale=1.0)
rcParams['font.sans-serif'] = [chinese_font]
rcParams['axes.unicode_minus'] = False
rcParams['figure.dpi'] = 150
rcParams['savefig.dpi'] = 300

# ==================== 配色 ====================
C_BLUE = '#2563EB'
C_ORANGE = '#EA580C'
C_GREEN = '#059669'
C_RED = '#DC2626'
C_PURPLE = '#7C3AED'
C_GRAY = '#6B7280'

# ==================== 读取数据 ====================
df = pd.read_excel("数据分析师招聘信息_清洗后.xlsx", sheet_name="清洗后数据")
total = len(df)
print(f"总样本量: {total} 条")



# ---------- 特征选择与标准化 ----------
print("\n--- 特征选择与标准化 ---")

features = df[['平均薪资', '学历数值', '经验数值']].copy()
print(f"聚类特征: 平均薪资、学历数值、经验数值")
print(f"特征维度: {features.shape}")

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
print("特征标准化完成")

# ---------- 确定最佳聚类数 ----------
print("\n--- 确定最佳聚类数 ---")

inertias = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(features_scaled, labels))

best_k = k_range[np.argmax(silhouette_scores)]
print(f"轮廓系数最优: k={best_k}, 轮廓系数={max(silhouette_scores):.4f}")
print(f"k=3时轮廓系数: {silhouette_scores[1]:.4f}")

# 画聚类效果评价图
fig, ax1 = plt.subplots(figsize=(10, 6))

color1 = C_BLUE
ax1.set_xlabel('聚类数量 k', fontsize=11)
ax1.set_ylabel('SSE（误差平方和）', fontsize=11, color=color1)
ax1.plot(k_range, inertias, 'o-', color=color1, linewidth=2, markersize=8, label='SSE')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(True, alpha=0.2, linestyle=':')

ax2 = ax1.twinx()
color2 = C_RED
ax2.set_ylabel('轮廓系数', fontsize=11, color=color2)
ax2.plot(k_range, silhouette_scores, 's-', color=color2, linewidth=2, markersize=8, label='轮廓系数')
ax2.tick_params(axis='y', labelcolor=color2)

plt.title('聚类效果评价：肘部法则与轮廓系数', fontsize=14, pad=15)
sns.despine()
plt.tight_layout()
plt.savefig('图4-1_聚类效果评价.png', bbox_inches='tight')
plt.close()
print("✓ 图4-1 聚类效果评价图")

# ---------- K-Means聚类 ----------
print("\n--- K-Means聚类（k=3） ---")

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['聚类标签'] = kmeans.fit_predict(features_scaled)
print("聚类完成")

final_silhouette = silhouette_score(features_scaled, df['聚类标签'])
print(f"k=3时轮廓系数: {final_silhouette:.4f}")

# ---------- PCA降维与可视化 ----------
print("\n--- PCA降维可视化 ---")

pca = PCA(n_components=2, random_state=42)
features_pca = pca.fit_transform(features_scaled)
df['PCA1'] = features_pca[:, 0]
df['PCA2'] = features_pca[:, 1]

print(f"累计解释方差: {pca.explained_variance_ratio_.sum():.2%}")

centers_pca = pca.transform(kmeans.cluster_centers_)

plt.figure(figsize=(10, 7))

cluster_colors = [C_GREEN, C_BLUE, C_RED]
cluster_names = ['入门岗', '进阶岗', '专家岗']

for i in range(3):
    cluster_data = df[df['聚类标签'] == i]
    plt.scatter(cluster_data['PCA1'], cluster_data['PCA2'],
                c=cluster_colors[i], alpha=0.6, s=40, edgecolors='white', linewidth=0.5,
                label=f'{cluster_names[i]}（{len(cluster_data)}个）')

plt.scatter(centers_pca[:, 0], centers_pca[:, 1],
            c='black', s=200, marker='*', edgecolors='white', linewidth=2,
            label='聚类中心', zorder=5)

plt.title(f'K-Means聚类结果可视化（PCA降维）\n轮廓系数: {final_silhouette:.3f}', fontsize=14, pad=15)
plt.xlabel(f'第一主成分（解释方差{pca.explained_variance_ratio_[0]:.1%}）', fontsize=11)
plt.ylabel(f'第二主成分（解释方差{pca.explained_variance_ratio_[1]:.1%}）', fontsize=11)
plt.legend(fontsize=10, frameon=True, fancybox=True)
plt.grid(True, alpha=0.2, linestyle=':')
sns.despine()

plt.tight_layout()
plt.savefig('图4-2_聚类结果PCA可视化.png', bbox_inches='tight')
plt.close()
print("✓ 图4-2 聚类结果PCA可视化")

# ---------- 聚类特征分析 ----------
print("\n--- 聚类特征分析 ---")

cluster_stats = df.groupby('聚类标签').agg(
    岗位数量=('职位名称', 'count'),
    平均薪资=('平均薪资', 'mean'),
    平均学历数值=('学历数值', 'mean'),
    平均经验数值=('经验数值', 'mean'),
).sort_values('平均薪资')

cluster_mapping = {cluster_stats.index[0]: 0, cluster_stats.index[1]: 1, cluster_stats.index[2]: 2}
df['聚类类别'] = df['聚类标签'].map(cluster_mapping)
df['聚类名称'] = df['聚类类别'].map({0: '入门岗', 1: '进阶岗', 2: '专家岗'})

cluster_stats_final = df.groupby('聚类名称').agg(
    岗位数量=('职位名称', 'count'),
    占比=('职位名称', lambda x: len(x)/total*100),
    平均薪资=('平均薪资', 'mean'),
    薪资中位数=('平均薪资', 'median'),
    最低薪资=('平均薪资', 'min'),
    最高薪资=('平均薪资', 'max'),
    平均学历数值=('学历数值', 'mean'),
    平均经验数值=('经验数值', 'mean'),
).reindex(['入门岗', '进阶岗', '专家岗'])

print("\n【聚类结果统计】")
print(cluster_stats_final.round(2).to_string())

# ---------- 聚类特征对比图 ----------
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

cluster_order = ['入门岗', '进阶岗', '专家岗']
colors_cluster = [C_GREEN, C_BLUE, C_RED]

ax = axes[0]
salary_means = [cluster_stats_final.loc[c, '平均薪资'] for c in cluster_order]
bars = ax.bar(cluster_order, salary_means, color=colors_cluster, alpha=0.75, width=0.6)
ax.set_title('平均薪资对比', fontsize=12, pad=10)
ax.set_ylabel('平均薪资（元/月）', fontsize=10)
for bar, val in zip(bars, salary_means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
            f'{int(val)}', ha='center', va='bottom', fontsize=10)
ax.grid(axis='y', alpha=0.2, linestyle=':')

ax = axes[1]
edu_labels = ['不限', '大专', '本科', '硕士', '博士']
edu_means = [cluster_stats_final.loc[c, '平均学历数值'] for c in cluster_order]
bars = ax.bar(cluster_order, edu_means, color=colors_cluster, alpha=0.75, width=0.6)
ax.set_title('平均学历水平对比', fontsize=12, pad=10)
ax.set_ylabel('学历等级', fontsize=10)
ax.set_ylim(0, 4)
ax.set_yticks([0, 1, 2, 3, 4])
ax.set_yticklabels(edu_labels)
for bar, val in zip(bars, edu_means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f'{val:.2f}', ha='center', va='bottom', fontsize=10)
ax.grid(axis='y', alpha=0.2, linestyle=':')

ax = axes[2]
exp_means = [cluster_stats_final.loc[c, '平均经验数值'] for c in cluster_order]
bars = ax.bar(cluster_order, exp_means, color=colors_cluster, alpha=0.75, width=0.6)
ax.set_title('平均工作经验对比', fontsize=12, pad=10)
ax.set_ylabel('工作经验（年）', fontsize=10)
for bar, val in zip(bars, exp_means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
            f'{val:.1f}年', ha='center', va='bottom', fontsize=10)
ax.grid(axis='y', alpha=0.2, linestyle=':')

plt.suptitle('三类岗位的核心特征对比', fontsize=14, y=1.02)
sns.despine()
plt.tight_layout()
plt.savefig('图4-3_聚类特征对比.png', bbox_inches='tight')
plt.close()
print("✓ 图4-3 聚类特征对比")

# ---------- 区域分布图 ----------

region_dist = pd.crosstab(df['聚类名称'], df['区域'], normalize='index') * 100
region_dist = region_dist.reindex(['入门岗', '进阶岗', '专家岗'])

print(f"区域数量: {len(region_dist.columns)}")
print(f"区域列表: {region_dist.columns.tolist()}")

plt.figure(figsize=(10, 6))

regions = region_dist.columns.tolist()
bottom = np.zeros(3)

# 用seaborn自动生成颜色，就不会不够用了
colors_region = sns.color_palette("Set2", n_colors=len(regions))

for i, region in enumerate(regions):
    values = region_dist[region].values
    bars = plt.bar(['入门岗', '进阶岗', '专家岗'], values, bottom=bottom,
                   color=colors_region[i], alpha=0.85, label=region, width=0.55, edgecolor='white')

    for j, bar in enumerate(bars):
        height = bar.get_height()
        if height > 5:
            plt.text(bar.get_x() + bar.get_width() / 2, bottom[j] + height / 2,
                     f'{height:.0f}%', ha='center', va='center', fontsize=9, color='white', fontweight='bold')

    bottom += values

plt.title('三类岗位的区域分布对比', fontsize=14, pad=15)
plt.ylabel('占比（%）', fontsize=11)
plt.legend(title='区域', fontsize=10, bbox_to_anchor=(1.02, 1), loc='upper left')
plt.ylim(0, 105)
plt.grid(axis='y', alpha=0.2, linestyle=':')

sns.despine()
plt.tight_layout()
plt.savefig('图4-4_聚类区域分布.png', bbox_inches='tight')
plt.close()
print("✓ 图4-4 聚类区域分布")

# ---------- 城市等级分布图 ----------

city_tier_dist = pd.crosstab(df['聚类名称'], df['城市等级'], normalize='index') * 100
city_tier_dist = city_tier_dist.reindex(index=['入门岗', '进阶岗', '专家岗'],
                                        columns=['一线城市', '新一线城市', '二线及以下'])

print(f"城市等级数量: {len(city_tier_dist.columns)}")
print(f"城市等级列表: {city_tier_dist.columns.tolist()}")

plt.figure(figsize=(10, 6))

city_tiers = city_tier_dist.columns.tolist()
bottom = np.zeros(3)

colors_tier = sns.color_palette("RdYlGn_r", n_colors=len(city_tiers))

for i, tier in enumerate(city_tiers):
    values = city_tier_dist[tier].values
    bars = plt.bar(['入门岗', '进阶岗', '专家岗'], values, bottom=bottom,
                   color=colors_tier[i], alpha=0.85, label=tier, width=0.55, edgecolor='white')

    for j, bar in enumerate(bars):
        height = bar.get_height()
        if height > 5:
            plt.text(bar.get_x() + bar.get_width() / 2, bottom[j] + height / 2,
                     f'{height:.0f}%', ha='center', va='center', fontsize=9, color='white', fontweight='bold')

    bottom += values

plt.title('三类岗位的城市等级分布对比', fontsize=14, pad=15)
plt.ylabel('占比（%）', fontsize=11)
plt.legend(title='城市等级', fontsize=10, bbox_to_anchor=(1.02, 1), loc='upper left')
plt.ylim(0, 105)
plt.grid(axis='y', alpha=0.2, linestyle=':')

sns.despine()
plt.tight_layout()
plt.savefig('图4-5_聚类城市等级分布.png', bbox_inches='tight')
plt.close()
print("✓ 图4-5 聚类城市等级分布")