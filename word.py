import warnings

warnings.filterwarnings('ignore', message='findfont')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import matplotlib.font_manager as fm

# ==================== 字体设置 ====================
font_names = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS', 'PingFang SC']
available_fonts = [f.name for f in fm.fontManager.ttflist]
chinese_font = next((f for f in font_names if f in available_fonts), 'DejaVu Sans')

sns.set(font=chinese_font, style="white", font_scale=1.1)
rcParams['font.sans-serif'] = [chinese_font]
rcParams['axes.unicode_minus'] = False
rcParams['figure.dpi'] = 150
rcParams['savefig.dpi'] = 300

# ==================== 配色 ====================
C_BLUE = '#2563EB'
C_ORANGE = '#EA580C'
C_GREEN = '#059669'
C_RED = '#DC2626'
C_GRAY = '#6B7280'
C_PURPLE = '#7C3AED'

# ==================== 读取统计结果 ====================
try:
    df_skill = pd.read_excel("技能需求统计结果.xlsx", sheet_name="技能统计")
except:
    import re

    df = pd.read_excel("BOSS直聘-详情采集.xlsx", sheet_name="Sheet1")

    skill_categories = [
        ("编程语言", ["SQL", "Python", "R语言", "Java", "C++", "Scala"]),
        ("办公工具", ["Excel", "PPT", "Word", "Office"]),
        ("可视化工具", ["Tableau", "Power BI", "PowerBI", "FineBI", "BI工具", "ECharts", "可视化"]),
        ("数据库", ["MySQL", "Oracle", "SQL Server", "Redis", "MongoDB", "PostgreSQL"]),
        ("大数据工具", ["Hadoop", "Spark", "Hive", "Flink", "Kafka"]),
        ("统计学", ["统计学", "统计分析", "假设检验", "回归分析", "概率论"]),
        ("机器学习", ["机器学习", "算法", "预测模型", "聚类", "分类", "深度学习"]),
        ("数据处理", ["数据挖掘", "数据仓库", "ETL", "数据清洗", "爬虫"]),
        ("软技能", ["沟通能力", "逻辑思维", "团队协作", "学习能力", "抗压能力"]),
    ]


    def count_skill(text, skill):
        if pd.isna(text) or not text:
            return 0
        return str(text).lower().count(skill.lower())


    results = []
    for category, skills in skill_categories:
        for skill in skills:
            count = sum(1 for _, row in df.iterrows()
                        if skill.lower() in str(row.get('职位描述', '')).lower())
            percentage = count / len(df) * 100
            results.append({
                "技能类别": category,
                "技能名称": skill,
                "出现岗位数": count,
                "占比(%)": round(percentage, 1),
            })

    df_skill = pd.DataFrame(results)
    df_skill = df_skill.sort_values(by="出现岗位数", ascending=False).reset_index(drop=True)
    df_skill.to_excel("技能需求统计结果.xlsx", index=False, sheet_name="技能统计")

print(f"共统计 {len(df_skill)} 个技能")

# ==================== 图1：TOP15技能占比条形图 ====================
print("\n正在生成图1：TOP15技能占比条形图...")

top15 = df_skill.head(15).copy()
top15 = top15.sort_values("占比(%)", ascending=True)

category_colors = {
    "编程语言": C_BLUE,
    "办公工具": C_GREEN,
    "可视化工具": C_ORANGE,
    "数据库": C_PURPLE,
    "大数据工具": C_RED,
    "统计学": '#0891B2',
    "机器学习": '#BE185D',
    "数据处理": '#65A30D',
    "软技能": C_GRAY,
}

bar_colors = [category_colors.get(cat, C_GRAY) for cat in top15['技能类别']]

plt.figure(figsize=(12, 8))
bars = plt.barh(top15['技能名称'], top15['占比(%)'], color=bar_colors, height=0.7, edgecolor='white', linewidth=1.5)

for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.5, bar.get_y() + bar.get_height() / 2,
             f'{width:.1f}%', ha='left', va='center', fontsize=10)

plt.title('数据分析师岗位核心技能需求TOP15', fontsize=16, pad=20)
plt.xlabel('岗位提及占比（%）', fontsize=12)
plt.ylabel('技能名称', fontsize=12)
plt.grid(True, axis='x', alpha=0.2, linestyle=':')
plt.xlim(0, max(top15['占比(%)']) * 1.15)

from matplotlib.patches import Patch

legend_elements = [Patch(facecolor=color, label=cat)
                   for cat, color in category_colors.items()
                   if cat in top15['技能类别'].values]
plt.legend(handles=legend_elements, title='技能类别', loc='lower right', fontsize=9, title_fontsize=10)

sns.despine()
plt.tight_layout()
plt.savefig('图3-7_技能需求TOP15.png', bbox_inches='tight')
plt.close()
print("✓ 已保存：图3-7_技能需求TOP15.png")

# ==================== 图2：技能类别需求对比 ====================
print("\n正在生成图2：技能类别需求对比...")

category_max = df_skill.groupby("技能类别")["占比(%)"].max().reset_index()
category_max = category_max.sort_values("占比(%)", ascending=False)

plt.figure(figsize=(10, 6))

cat_colors = [category_colors.get(cat, C_GRAY) for cat in category_max['技能类别']]

bars = plt.bar(category_max['技能类别'], category_max['占比(%)'],
               color=cat_colors, width=0.6, edgecolor='white', linewidth=1.5)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 1,
             f'{height:.1f}%', ha='center', va='bottom', fontsize=10)

plt.title('不同技能类别的需求程度对比', fontsize=15, pad=15)
plt.xlabel('技能类别', fontsize=11)
plt.ylabel('最高提及占比（%）', fontsize=11)
plt.grid(True, axis='y', alpha=0.2, linestyle=':')
plt.xticks(rotation=30, ha='right')
plt.ylim(0, max(category_max['占比(%)']) * 1.15)

sns.despine()
plt.tight_layout()
plt.savefig('图3-8_技能类别对比.png', bbox_inches='tight')
plt.close()
print("✓ 已保存：图3-8_技能类别对比.png")

# ==================== 总结 ====================
print("\n" + "=" * 50)
print("🎉 可视化完成！")
print("=" * 50)
print("生成的图片：")
print("  1. 图3-7_技能需求TOP15.png")
print("  2. 图3-8_技能类别对比.png")
print("\n生成的表格：")
print("  技能需求统计结果.xlsx")