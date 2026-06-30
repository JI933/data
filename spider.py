import requests
import json
import time
import random
import re
import pandas as pd
from urllib.parse import quote


class BaiduJobSpider:
    def __init__(self, keyword="数据分析师"):
        self.keyword = keyword
        self.base_url = "https://yiqifu.baidu.com/g/aqc"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "X-Requested-With": "XMLHttpRequest",
            "Referer": "https://yiqifu.baidu.com/g/aqc/joblist?q=%E6%95%B0%E6%8D%AE%E5%88%86%E6%9E%90%E5%B8%88",
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }
        self.provinces = {}
        self.all_jobs = []
        self.job_ids = set()

    def get_province_list(self):
        """获取所有省份代码列表"""
        url = f"{self.base_url}/getDistrictAjax"
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.encoding = 'utf-8'
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == 0:
                    self.provinces = data["data"]["provinceCode"]
                    if "100000" in self.provinces:
                        del self.provinces["100000"]
                    return True
        except Exception as e:
            print(f"获取省份列表失败: {e}")
        return False

    def parse_salary(self, salary_str):
        """解析薪资字符串"""
        try:
            salary_str = salary_str.replace("元/月", "").replace("元/天", "").replace("元/小时", "")
            if "-" in salary_str:
                low, high = salary_str.split("-")
                low = int(low)
                high = int(high)
                avg = (low + high) / 2
                return low, high, avg
            else:
                return None, None, None
        except:
            return None, None, None

    def clean_job_name(self, job_name):
        """清理职位名称中的HTML标签"""
        clean = re.sub(r'<[^>]+>', '', job_name)
        return clean

    def fetch_province_jobs(self, province_code, province_name):
        """采集单个省份的所有职位"""
        province_jobs = []
        page = 1
        max_pages = 100
        keyword_encoded = quote(self.keyword)

        while page <= max_pages:
            url = f"{self.base_url}/joblist/getDataAjax?q={keyword_encoded}&page={page}&pagesize=20&district={province_code}"

            try:
                response = requests.get(url, headers=self.headers, timeout=10)
                response.encoding = 'utf-8'

                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == 0:
                        result = data["data"]
                        job_list = result.get("list", [])
                        page_count = result.get("pageCount", 1)

                        if not job_list:
                            break

                        for job in job_list:
                            job_id = job.get("jobId", "")

                            # 去重
                            if job_id in self.job_ids:
                                continue
                            self.job_ids.add(job_id)

                            # 解析薪资
                            salary_low, salary_high, salary_avg = self.parse_salary(job.get("salary", ""))

                            # 只保留需要的字段
                            job_info = {
                                "职位名称": self.clean_job_name(job.get("jobName", "")),
                                "薪资范围": job.get("salary", ""),
                                "最低薪资": salary_low,
                                "最高薪资": salary_high,
                                "平均薪资": salary_avg,
                                "城市": job.get("city", ""),
                                "省份": province_name,
                                "公司名称": job.get("company", ""),
                                "学历要求": job.get("edu", ""),
                                "工作经验": job.get("exp", ""),
                            }

                            province_jobs.append(job_info)
                            self.all_jobs.append(job_info)

                        if page >= page_count:
                            break

                        page += 1
                        time.sleep(random.uniform(0.3, 0.8))
                    else:
                        break
                else:
                    break

            except Exception as e:
                time.sleep(2)
                try:
                    response = requests.get(url, headers=self.headers, timeout=10)
                    response.encoding = 'utf-8'
                    if response.status_code == 200:
                        continue
                except:
                    pass
                break

        return province_jobs

    def crawl(self):
        """开始爬取所有省份的数据"""
        print(f"开始采集「{self.keyword}」岗位信息")

        print("\n第1步：获取省份列表")
        if not self.get_province_list():
            print("获取省份列表失败")
            self.provinces = {
                "110000": "北京", "120000": "天津", "130000": "河北", "140000": "山西",
                "150000": "内蒙古", "210000": "辽宁", "220000": "吉林", "230000": "黑龙江",
                "310000": "上海", "320000": "江苏", "330000": "浙江", "340000": "安徽",
                "350000": "福建", "360000": "江西", "370000": "山东", "410000": "河南",
                "420000": "湖北", "430000": "湖南", "440000": "广东", "450000": "广西",
                "460000": "海南", "500000": "重庆", "510000": "四川", "520000": "贵州",
                "530000": "云南", "540000": "西藏", "610000": "陕西", "620000": "甘肃",
                "630000": "青海", "640000": "宁夏", "650000": "新疆",
            }

        print(f"\n第2步：按省份采集数据（共 {len(self.provinces)} 个省份）")

        for i, (code, name) in enumerate(self.provinces.items(), 1):
            print(f"[{i}/{len(self.provinces)}] 正在采集 {name}...", end=" ")
            jobs = self.fetch_province_jobs(code, name)
            if len(jobs) > 0:
                print(f"完成 获取 {len(jobs)} 条")
            else:
                print(f"完成 无数据")
            time.sleep(random.uniform(0.5, 1.5))

        print(f"全部采集完成，共获取 {len(self.all_jobs)} 条有效数据")

        return self.all_jobs

    def save_to_excel(self, filename="数据分析师招聘信息.xlsx"):
        """保存到Excel文件"""
        if not self.all_jobs:
            print("没有数据可保存")
            return

        df = pd.DataFrame(self.all_jobs)
        df = df.sort_values(by=["省份", "平均薪资"], ascending=[True, False])

        # 保存Excel（只有招聘数据一个sheet）
        df.to_excel(filename, sheet_name="招聘数据", index=False, engine="openpyxl")

        print(f"\n数据已保存到: {filename}")
        print(f"共 {len(df)} 条数据，{len(df.columns)} 个字段")
        print(f"字段列表: {', '.join(df.columns)}")

    def run(self, output_file="数据分析师招聘信息.xlsx"):
        """运行完整流程"""
        self.crawl()
        self.save_to_excel(output_file)
        return self.all_jobs


if __name__ == "__main__":
    spider = BaiduJobSpider(keyword="数据分析师")
    spider.run(output_file="数据分析师招聘信息.xlsx")