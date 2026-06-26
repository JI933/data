import pandas as pd
import numpy as np

# 读取数据
df = pd.read_excel("数据分析师招聘信息.xlsx", sheet_name="招聘数据")

print(f"原始数据量: {len(df)} 条")

# ==================== 数据清洗 ====================

# 1. 删除异常薪资数据（平均薪资低于2000的）
df = df[df['平均薪资'] >= 2000].copy()
print(f"删除异常薪资后: {len(df)} 条")

# 2. 填充工作经验空白值
df['工作经验'] = df['工作经验'].fillna('1年以内')
df['工作经验'] = df['工作经验'].replace('', '1年以内')

# 3. 删除城市为空的数据
df = df.dropna(subset=['城市'])
df = df[df['城市'] != '']
print(f"删除城市为空后: {len(df)} 条")

# 4. 岗位名称清洗（去除多余空格和特殊字符）
df['职位名称'] = df['职位名称'].str.strip()  # 去除首尾空格
df['职位名称'] = df['职位名称'].str.replace(r'\s+', ' ', regex=True)  # 多个空格变一个

# 5. 再次去重（按公司+职位+城市）
df = df.drop_duplicates(subset=['公司名称', '职位名称', '城市'])
print(f"去重后: {len(df)} 条")

# ==================== 特征构造 ====================

# 1. 薪资等级特征
def salary_level(salary):
    if salary < 8000:
        return '低薪'
    elif salary < 15000:
        return '中低薪'
    elif salary < 25000:
        return '中高薪'
    else:
        return '高薪'

df['薪资等级'] = df['平均薪资'].apply(salary_level)

# 2. 学历数值化
edu_map = {'不限': 0, '大专': 1, '本科': 2, '硕士': 3, '博士': 4}
df['学历数值'] = df['学历要求'].map(edu_map).fillna(0)

# 3. 工作经验数值化
exp_map = {
    '不限': 0,
    '1年以内': 0.5,
    '1-3年': 2,
    '3-5年': 4,
    '5-10年': 7.5,
    '10年以上': 12
}
df['经验数值'] = df['工作经验'].map(exp_map).fillna(0)

# 4. 城市等级特征
first_tier = ['北京', '上海', '广州', '深圳']
new_first_tier = ['杭州', '成都', '武汉', '西安', '重庆', '苏州', '南京',
                  '天津', '郑州', '长沙', '东莞', '宁波', '佛山', '合肥', '青岛']

def city_tier(city):
    for c in first_tier:
        if c in city:
            return '一线城市'
    for c in new_first_tier:
        if c in city:
            return '新一线城市'
    return '二线及以下'

df['城市等级'] = df['城市'].apply(city_tier)

# 5. 区域划分特征
east = ['北京', '天津', '河北', '上海', '江苏', '浙江', '福建', '山东', '广东', '海南']
central = ['山西', '安徽', '江西', '河南', '湖北', '湖南']
west = ['内蒙古', '广西', '重庆', '四川', '贵州', '云南', '西藏', '陕西', '甘肃', '青海', '宁夏', '新疆']
northeast = ['辽宁', '吉林', '黑龙江']

def region(province):
    if province in east:
        return '东部地区'
    elif province in central:
        return '中部地区'
    elif province in west:
        return '西部地区'
    elif province in northeast:
        return '东北地区'
    else:
        return '其他'

df['区域'] = df['省份'].apply(region)

# 保存清洗后的数据
df.to_excel("数据分析师招聘信息_清洗后.xlsx", sheet_name="清洗后数据", index=False)

print(f"\n清洗完成，最终数据量: {len(df)} 条")
print(f"字段列表: {', '.join(df.columns)}")
print("\n薪资等级分布:")
print(df['薪资等级'].value_counts().sort_index())
print("\n城市等级分布:")
print(df['城市等级'].value_counts())
print("\n区域分布:")
print(df['区域'].value_counts())