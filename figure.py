import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import matplotlib.font_manager as fm

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

# ==================== 图1：城市四象限图 ====================
print("\n✓ 图1：城市四象限图")
plt.figure(figsize=(11, 8))

city_stats = df.groupby('城市').agg(
    岗位数量=('职位名称', 'count'),
    平均薪资=('平均薪资', 'mean')
).query('岗位数量 >= 10').reset_index()

median_jobs = city_stats['岗位数量'].median()
median_salary = city_stats['平均薪资'].median()

x_min, x_max = city_stats['岗位数量'].min() * 0.5, city_stats['岗位数量'].max() * 1.1
y_min, y_max = city_stats['平均薪资'].min() * 0.7, city_stats['平均薪资'].max() * 1.1

plt.fill_between([median_jobs, x_max], median_salary, y_max,
                 alpha=0.08, color=C_GREEN, zorder=0)
plt.fill_between([x_min, median_jobs], median_salary, y_max,
                 alpha=0.08, color=C_ORANGE, zorder=0)
plt.fill_between([median_jobs, x_max], y_min, median_salary,
                 alpha=0.08, color=C_PURPLE, zorder=0)
plt.fill_between([x_min, median_jobs], y_min, median_salary,
                 alpha=0.04, color=C_GRAY, zorder=0)

sizes = city_stats['岗位数量'] * 1.5
plt.scatter(city_stats['岗位数量'], city_stats['平均薪资'],
            s=sizes, c=C_BLUE, alpha=0.7, edgecolors='white', linewidth=1.5, zorder=2)

plt.axvline(median_jobs, color='#9CA3AF', linestyle='--', linewidth=1.2, alpha=0.6, zorder=1)
plt.axhline(median_salary, color='#9CA3AF', linestyle='--', linewidth=1.2, alpha=0.6, zorder=1)

top_cities = city_stats.nlargest(12, '岗位数量')
for _, row in top_cities.iterrows():
    plt.annotate(row['城市'],
                 (row['岗位数量'], row['平均薪资']),
                 xytext=(8, 5), textcoords='offset points',
                 fontsize=9.5, color='#1F2937', zorder=3)

from matplotlib.patches import Patch

legend_elements = [
    Patch(facecolor=C_GREEN, alpha=0.2, label='高薪多岗区（理想就业地）'),
    Patch(facecolor=C_ORANGE, alpha=0.2, label='高薪少岗区（竞争小机会少）'),
    Patch(facecolor=C_PURPLE, alpha=0.2, label='低薪多岗区（机会多薪资低）'),
    Patch(facecolor=C_GRAY, alpha=0.1, label='低薪少岗区（不推荐）'),
]
plt.legend(handles=legend_elements, loc='lower right', fontsize=9, frameon=True, fancybox=True)

plt.title('城市就业吸引力四象限分析（岗位数量 vs 平均薪资）', fontsize=14, pad=15)
plt.xlabel('岗位数量（个）', fontsize=11)
plt.ylabel('平均薪资（元/月）', fontsize=11)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.grid(True, alpha=0.15, linestyle=':')
sns.despine()

plt.tight_layout()
plt.savefig('图3-1_城市四象限分析.png', bbox_inches='tight')
plt.close()

# ==================== 图2：四大区域对比 ====================
print("✓ 图2：四大区域对比")
plt.figure(figsize=(9, 6.5))

region_order = ['东部地区', '中部地区', '西部地区', '东北地区']
region_stats = df.groupby('区域').agg(
    岗位数量=('职位名称', 'count'),
    平均薪资=('平均薪资', 'mean')
).reindex(region_order)

x = np.arange(len(region_order))
width = 0.55

ax1 = plt.gca()
colors_region = [C_BLUE, C_ORANGE, C_GREEN, C_PURPLE]
bars = ax1.bar(x, region_stats['岗位数量'], width, color=colors_region, alpha=0.75)
ax1.set_xticks(x)
ax1.set_xticklabels(region_stats.index, fontsize=10)
ax1.set_ylabel('岗位数量（个）', fontsize=11, color='#374151')

for i, bar in enumerate(bars):
    height = bar.get_height()
    pct = height / total * 100
    ax1.text(bar.get_x() + bar.get_width() / 2, height + 15,
             f'{pct:.1f}%', ha='center', va='bottom', fontsize=10, color=colors_region[i])

ax2 = ax1.twinx()
ax2.plot(x, region_stats['平均薪资'], 'o-', color=C_RED,
         linewidth=2.5, markersize=10, markerfacecolor='white', markeredgewidth=2.5)
ax2.set_ylabel('平均薪资（元/月）', fontsize=11, color=C_RED)
ax2.tick_params(axis='y', labelcolor=C_RED)

for i, salary in enumerate(region_stats['平均薪资']):
    ax2.text(i + 0.15, salary + 300, f'{int(salary)}', ha='left', va='bottom',
             fontsize=10, color=C_RED)

plt.title('四大区域数据分析师岗位数量与平均薪资对比', fontsize=14, pad=15)
ax1.set_ylim(0, max(region_stats['岗位数量']) * 1.25)
ax2.set_ylim(0, max(region_stats['平均薪资']) * 1.2)
plt.tight_layout()
plt.savefig('图3-2_四大区域对比.png', bbox_inches='tight')
plt.close()

# ==================== 图3：城市等级薪资范围 ====================
print("✓ 图3：城市等级薪资范围")
plt.figure(figsize=(9, 6.5))

city_tier_order = ['一线城市', '新一线城市', '二线及以下']

tier_stats = df.groupby('城市等级').agg(
    岗位数量=('职位名称', 'count'),
    平均薪资=('平均薪资', 'mean'),
    最低薪资=('平均薪资', 'min'),
    最高薪资=('平均薪资', 'max')
).reindex(city_tier_order)

x = np.arange(len(city_tier_order))
width = 0.55

colors_tier = [C_RED, C_ORANGE, C_GREEN]

bars = plt.bar(x, tier_stats['平均薪资'], width, color=colors_tier, alpha=0.75,
               edgecolor='white', linewidth=1.2)

plt.errorbar(x, tier_stats['平均薪资'],
             yerr=[tier_stats['平均薪资'] - tier_stats['最低薪资'],
                   tier_stats['最高薪资'] - tier_stats['平均薪资']],
             fmt='none', ecolor='#374151', elinewidth=1.5, capsize=8, capthick=1.5)

for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 1500,
             f'{tier_stats["岗位数量"].iloc[i]}个岗位',
             ha='center', va='bottom', fontsize=10, color=colors_tier[i])
    plt.text(bar.get_x() + bar.get_width() / 2, height / 2,
             f'{int(height)}',
             ha='center', va='center', fontsize=11, color='white')

for i in range(len(city_tier_order)):
    plt.text(x[i] + 0.35, tier_stats['最高薪资'].iloc[i],
             f'最高{int(tier_stats["最高薪资"].iloc[i])}',
             ha='left', va='center', fontsize=8.5, color='#6B7280')
    plt.text(x[i] + 0.35, tier_stats['最低薪资'].iloc[i],
             f'最低{int(tier_stats["最低薪资"].iloc[i])}',
             ha='left', va='center', fontsize=8.5, color='#6B7280')

plt.xticks(x, city_tier_order, fontsize=10)
plt.title('不同等级城市的数据分析师薪资范围对比', fontsize=14, pad=15)
plt.ylabel('薪资（元/月）', fontsize=11)
plt.ylim(0, tier_stats['最高薪资'].max() * 1.15)
plt.grid(axis='y', alpha=0.2, linestyle=':')
sns.despine()

plt.tight_layout()
plt.savefig('图3-3_城市等级薪资范围.png', bbox_inches='tight')
plt.close()

# ==================== 图4：薪资分布直方图 ====================
print("✓ 图4：薪资分布直方图")
plt.figure(figsize=(10, 6))

salary_data = df['平均薪资'].dropna()
mean_salary = salary_data.mean()
median_salary = salary_data.median()

n, bins, patches = plt.hist(salary_data, bins=35, color=C_BLUE, alpha=0.6,
                            edgecolor='white', linewidth=0.8, density=True)

sns.kdeplot(salary_data, color=C_ORANGE, linewidth=2.5, label='核密度估计')

plt.axvline(mean_salary, color=C_RED, linestyle='--', linewidth=2,
            label=f'平均值: {mean_salary:.0f}元')
plt.axvline(median_salary, color=C_GREEN, linestyle='--', linewidth=2,
            label=f'中位数: {median_salary:.0f}元')

plt.title('数据分析师平均薪资分布', fontsize=14, pad=15)
plt.xlabel('平均薪资（元/月）', fontsize=11)
plt.ylabel('密度', fontsize=11)
plt.legend(fontsize=10, frameon=True, fancybox=True)
plt.grid(axis='y', alpha=0.2, linestyle=':')
sns.despine()

plt.tight_layout()
plt.savefig('图3-4_薪资分布直方图.png', bbox_inches='tight')
plt.close()

# ==================== 图5：各城市平均薪资TOP10 ====================
print("✓ 图5：各城市平均薪资TOP10")
plt.figure(figsize=(10, 6.5))

city_salary = df.groupby('城市').agg(
    平均薪资=('平均薪资', 'mean'),
    岗位数量=('职位名称', 'count')
).query('岗位数量 >= 10').sort_values('平均薪资', ascending=True).tail(10)

colors_bar = plt.cm.Reds(np.linspace(0.4, 0.85, len(city_salary)))
bars = plt.barh(city_salary.index, city_salary['平均薪资'],
                color=colors_bar, height=0.6, edgecolor='white', linewidth=0.8)

plt.title('各城市数据分析师平均薪资TOP10（样本数≥10）', fontsize=14, pad=15)
plt.xlabel('平均薪资（元/月）', fontsize=11)
plt.ylabel('城市', fontsize=11)

for i, bar in enumerate(bars):
    width = bar.get_width()
    job_count = city_salary['岗位数量'].iloc[i]
    plt.text(width + 200, bar.get_y() + bar.get_height() / 2,
             f'{int(width)}元（{job_count}个岗位）',
             ha='left', va='center', fontsize=9.5, color='#374151')

plt.grid(axis='x', alpha=0.2, linestyle=':')
sns.despine()
plt.tight_layout()
plt.savefig('图3-5_城市平均薪资TOP10.png', bbox_inches='tight')
plt.close()

# ==================== 图6：工作经验与薪资（分学历） ====================
print("✓ 图6：工作经验与薪资（分学历）")
plt.figure(figsize=(10, 6.5))


def merge_exp(exp):
    if exp in ['5-10年', '10年以上']:
        return '5年以上'
    return exp


df['经验合并'] = df['工作经验'].apply(merge_exp)
exp_order = ['不限', '1年以内', '1-3年', '3-5年', '5年以上']

edu_list = ['大专', '本科', '硕士']
colors_edu = [C_GREEN, C_BLUE, C_ORANGE]
markers_edu = ['s', 'o', '^']

for edu, color, marker in zip(edu_list, colors_edu, markers_edu):
    edu_data = df[df['学历要求'] == edu]
    exp_salary = edu_data.groupby('经验合并')['平均薪资'].mean().reindex(exp_order).dropna()
    exp_count = edu_data.groupby('经验合并')['职位名称'].count().reindex(exp_order).dropna()

    valid_exp = exp_count[exp_count >= 5].index
    exp_salary_valid = exp_salary[valid_exp]

    if len(exp_salary_valid) >= 2:
        plt.plot(exp_salary_valid.index, exp_salary_valid.values, marker=marker,
                 linewidth=2.5, markersize=9, label=f'{edu}学历',
                 color=color, markerfacecolor='white', markeredgewidth=2.5)

plt.title('不同学历的工作经验与薪资增长路径对比', fontsize=14, pad=15)
plt.xlabel('工作经验要求', fontsize=11)
plt.ylabel('平均薪资（元/月）', fontsize=11)
plt.legend(fontsize=10, frameon=True, fancybox=True)
plt.grid(True, alpha=0.2, linestyle=':')
sns.despine()

plt.tight_layout()
plt.savefig('图3-6_经验薪资曲线分学历.png', bbox_inches='tight')
plt.close()
