import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
import os
import calendar

# 配置
plt.rcParams.update({'font.size': 10, 'figure.dpi': 150})
sns.set_style("whitegrid")

class DatasetDeepDive:
    def __init__(self):
        self.data = None
        self.output_dir = './dataset_analysis_report'
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_data(self):
        print("Step 1: Loading Dataset...")
        try:
            dataset = load_dataset("Dingdong-Inc/FreshRetailNet-50K")
            self.data = dataset['train'].to_pandas()
            self.data['dt'] = pd.to_datetime(self.data['dt'])
            
            # 定义缺货：假设 stock_hour6_22_cnt > 0 代表该日存在缺货时段
            # 注意：具体定义需根据数据集文档，此处沿用之前代码逻辑
            self.data['is_oos'] = self.data['stock_hour6_22_cnt'] > 0
            
            print(f"  - Loaded {len(self.data):,} rows.")
            print(f"  - Time range: {self.data['dt'].min().date()} to {self.data['dt'].max().date()}")
        except Exception as e:
            print(f"  [Error] Failed to load data: {e}")
            exit(1)

    def analyze_temporal_patterns(self):
        """维度一：时间序列分析 (Trend & Seasonality)"""
        print("Step 2: Analyzing Temporal Patterns...")
        df = self.data.copy()
        df['day_of_week'] = df['dt'].dt.day_name()
        df['month'] = df['dt'].dt.to_period('M')
        
        fig = plt.figure(figsize=(20, 12))
        
        # 1. 全局日销量趋势
        ax1 = plt.subplot(2, 2, 1)
        daily_sales = df.groupby('dt')['sale_amount'].sum()
        ax1.plot(daily_sales.index, daily_sales.values, color='#1f77b4', linewidth=1)
        ax1.set_title('Global Daily Sales Trend')
        ax1.set_ylabel('Total Sales')
        
        # 2. 周度模式 (Boxplot) - 核心：看周末是否比工作日高
        ax2 = plt.subplot(2, 2, 2)
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        # 归一化处理：为了避免某些商品销量极大拉偏整体，我们看每个store-product的周度系数
        # 这里简化处理：直接看总量的分布
        sns.boxplot(x='day_of_week', y='sale_amount', data=df.sample(min(50000, len(df))), 
                    order=days_order, ax=ax2, showfliers=False, palette="Set2")
        ax2.set_title('Sales Distribution by Day of Week (No Outliers)')
        
        # 3. 缺货率的周度变化
        ax3 = plt.subplot(2, 2, 3)
        oos_weekly = df.groupby('day_of_week')['is_oos'].mean().reindex(days_order) * 100
        sns.barplot(x=oos_weekly.index, y=oos_weekly.values, ax=ax3, palette="Reds")
        ax3.set_title('Stockout Rate by Day of Week (%)')
        ax3.set_ylabel('OOS Rate (%)')
        
        # 4. 月度季节性热力图 (Year vs Month)
        ax4 = plt.subplot(2, 2, 4)
        df['year'] = df['dt'].dt.year
        df['month_val'] = df['dt'].dt.month
        pivot = df.groupby(['year', 'month_val'])['sale_amount'].sum().unstack()
        sns.heatmap(pivot, annot=False, cmap='YlOrRd', ax=ax4)
        ax4.set_title('Monthly Sales Heatmap')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '1_temporal_patterns.png'))
        plt.close()

    def analyze_entity_distribution(self):
        """维度二：实体分析 (Store & Product Heterogeneity) - 帕累托法则"""
        print("Step 3: Analyzing Entity Distributions (Pareto)...")
        
        fig = plt.figure(figsize=(20, 10))
        
        # 1. 商品长尾分布 (Pareto Curve)
        ax1 = plt.subplot(1, 2, 1)
        prod_sales = self.data.groupby('product_id')['sale_amount'].sum().sort_values(ascending=False)
        
        # 计算累计百分比
        cum_pct = prod_sales.cumsum() / prod_sales.sum() * 100
        x_pct = np.arange(len(prod_sales)) / len(prod_sales) * 100
        
        ax1.plot(x_pct, cum_pct, color='purple', linewidth=2)
        ax1.fill_between(x_pct, cum_pct, alpha=0.1, color='purple')
        
        # 找到 80% 销量对应的商品比例
        eighty_pct_idx = np.where(cum_pct >= 80)[0][0]
        eighty_pct_x = x_pct[eighty_pct_idx]
        
        ax1.axvline(eighty_pct_x, color='red', linestyle='--')
        ax1.axhline(80, color='red', linestyle='--')
        ax1.text(eighty_pct_x + 2, 75, f'Top {eighty_pct_x:.1f}% Products\ncontribute 80% Sales', color='red')
        
        ax1.set_title('Product Pareto Analysis (Long Tail)')
        ax1.set_xlabel('% of Products')
        ax1.set_ylabel('Cumulative % of Sales')
        ax1.grid(True)
        
        # 2. 店铺规模分布
        ax2 = plt.subplot(1, 2, 2)
        store_sales = self.data.groupby('store_id')['sale_amount'].sum().sort_values()
        store_sales.plot(kind='barh', ax=ax2, color='teal', width=0.8)
        ax2.set_title('Total Sales by Store (Store Heterogeneity)')
        ax2.set_xlabel('Total Sales Amount')
        ax2.set_yticks([]) # 隐藏店铺ID，太密了
        ax2.text(0.5, 0.5, f'Total Stores: {len(store_sales)}', transform=ax2.transAxes, ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '2_entity_distribution.png'))
        plt.close()

    def analyze_censorship_characteristics(self):
        """维度三：缺货特性分析 (Censorship/OOS)"""
        print("Step 4: Analyzing Censorship (Stockouts)...")
        
        oos_data = self.data[self.data['is_oos']]
        normal_data = self.data[~self.data['is_oos']]
        
        fig = plt.figure(figsize=(20, 12))
        
        # 1. 缺货与非缺货的销量分布对比 (KDE)
        ax1 = plt.subplot(2, 2, 1)
        sns.kdeplot(normal_data['sale_amount'], label='Full Stock Days', fill=True, alpha=0.3, color='green', ax=ax1)
        sns.kdeplot(oos_data['sale_amount'], label='OOS Days (Censored)', fill=True, alpha=0.3, color='red', ax=ax1)
        ax1.set_title('Sales Density: Full Stock vs OOS')
        ax1.set_xlim(0, self.data['sale_amount'].quantile(0.99)) # 去除极端值影响显示
        ax1.legend()
        
        # 2. 什么样的商品更容易缺货？(销量高 vs 缺货频次)
        ax2 = plt.subplot(2, 2, 2)
        prod_stats = self.data.groupby('product_id').agg({
            'sale_amount': 'mean',  # 平均日销量
            'is_oos': 'mean'        # 缺货概率
        })
        
        sns.scatterplot(data=prod_stats, x='sale_amount', y='is_oos', alpha=0.4, ax=ax2, color='orange')
        sns.regplot(data=prod_stats, x='sale_amount', y='is_oos', scatter=False, ax=ax2, color='red')
        ax2.set_title('Product Popularity vs OOS Probability')
        ax2.set_xlabel('Avg Daily Sales')
        ax2.set_ylabel('OOS Probability')
        
        # 3. 缺货严重程度的热力图 (Store x DayOfWeek)
        ax3 = plt.subplot(2, 2, 3)
        self.data['day_of_week_idx'] = self.data['dt'].dt.dayofweek
        oos_heatmap = self.data.groupby(['store_id', 'day_of_week_idx'])['is_oos'].mean().unstack()
        days_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        oos_heatmap.columns = days_labels
        
        # 只取 Top 20 缺货最严重的店铺展示，避免图太长
        top_oos_stores = self.data.groupby('store_id')['is_oos'].mean().nlargest(20).index
        sns.heatmap(oos_heatmap.loc[top_oos_stores], cmap='Reds', annot=False, ax=ax3)
        ax3.set_title('OOS Intensity Heatmap (Top 20 High-OOS Stores)')
        
        # 4. 连续缺货分析 (Run Length Encoding idea)
        # 这里简单统计：每个Store-Product组合的总缺货天数分布
        ax4 = plt.subplot(2, 2, 4)
        sp_oos_counts = self.data.groupby(['store_id', 'product_id'])['is_oos'].sum()
        sns.histplot(sp_oos_counts[sp_oos_counts > 0], bins=50, kde=False, color='brown', ax=ax4)
        ax4.set_title('Distribution of Total OOS Days per SKU')
        ax4.set_xlabel('Total Days Out of Stock')
        ax4.set_yscale('log') # 对数坐标，因为长尾明显
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '3_censorship_analysis.png'))
        plt.close()

    def generate_text_report(self):
        """生成文本摘要报告"""
        print("Step 5: Generating Text Report...")
        
        total_rows = len(self.data)
        total_sales = self.data['sale_amount'].sum()
        oos_rows = self.data['is_oos'].sum()
        n_stores = self.data['store_id'].nunique()
        n_products = self.data['product_id'].nunique()
        
        # 计算基尼系数 (Gini Coefficient) 衡量商品不平等度
        prod_sales = self.data.groupby('product_id')['sale_amount'].sum().sort_values().values
        lorenz = np.cumsum(prod_sales) / prod_sales.sum()
        lorenz = np.insert(lorenz, 0, 0) 
        gini = 1 - 2 * np.trapz(lorenz, dx=1/len(prod_sales))
        
        report = f"""
==========================================================
FRESH RETAIL DATASET ANALYSIS REPORT
==========================================================

1. BASIC STATISTICS
-------------------
Total Records:      {total_rows:,}
Date Range:         {self.data['dt'].min().date()} to {self.data['dt'].max().date()}
Total Stores:       {n_stores}
Total Products:     {n_products}
Total Sales Volume: {total_sales:,.2f}

2. CENSORSHIP / OOS METRICS
---------------------------
Total OOS Events:   {oos_rows:,}
Global OOS Rate:    {oos_rows / total_rows * 100:.2f}%
Avg Sales (Normal): {self.data[~self.data['is_oos']]['sale_amount'].mean():.2f}
Avg Sales (OOS):    {self.data[self.data['is_oos']]['sale_amount'].mean():.2f}
(Note: OOS sales being lower/higher depends on whether the stockout happened early or late in the day)

3. PARETO & INEQUALITY
----------------------
Product Gini Coeff: {gini:.4f} (0=Equal, 1=Extreme Inequality)
Top 10% Products Share: {self.data.groupby('product_id')['sale_amount'].sum().nlargest(int(n_products*0.1)).sum() / total_sales * 100:.2f}% of Total Sales
Top 10% Stores Share:   {self.data.groupby('store_id')['sale_amount'].sum().nlargest(int(n_stores*0.1)).sum() / total_sales * 100:.2f}% of Total Sales

4. TEMPORAL INSIGHTS
--------------------
Busiest Day of Week: {self.data.groupby(self.data['dt'].dt.day_name())['sale_amount'].sum().idxmax()}
Highest OOS Day:     {self.data.groupby(self.data['dt'].dt.day_name())['is_oos'].mean().idxmax()}

==========================================================
Generated by Analysis Script
"""
        with open(os.path.join(self.output_dir, 'dataset_summary_report.txt'), 'w') as f:
            f.write(report)
        
        print(report)

    def run(self):
        self.load_data()
        self.analyze_temporal_patterns()
        self.analyze_entity_distribution()
        self.analyze_censorship_characteristics()
        self.generate_text_report()
        print(f"\nAnalysis Complete! Report saved to: {self.output_dir}/")

if __name__ == "__main__":
    analyzer = DatasetDeepDive()
    analyzer.run()