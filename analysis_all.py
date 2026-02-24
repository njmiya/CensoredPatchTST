import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from scipy import stats
import os
import glob
import warnings

# 忽略警告
warnings.filterwarnings('ignore')
plt.rcParams.update({'font.size': 12, 'figure.dpi': 150})
sns.set_style("whitegrid")

class ModelComparison:
    def __init__(self, base_path='./latent_demand_recovery/exp/demand'):
        self.base_path = base_path
        self.original_data = None
        self.model_results = {}
        self.output_dir = './model_comparison_v2'
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.load_data()
    
    def load_data(self):
        """加载原始数据和所有模型结果"""
        print("Step 1: Loading Data...")
        # 加载原始数据
        try:
            dataset = load_dataset("Dingdong-Inc/FreshRetailNet-50K")
            self.original_data = dataset['train'].to_pandas()
            # 确保日期格式正确
            self.original_data['dt'] = pd.to_datetime(self.original_data['dt'])
            self.original_data = self.original_data.sort_values(by=['store_id', 'product_id', 'dt'])
            print(f"  - Original data loaded: {len(self.original_data)} records")
        except Exception as e:
            print(f"  [Error] Failed to load HuggingFace dataset: {e}")
            return

        # 加载模型结果
        model_files = glob.glob(os.path.join(self.base_path, '*_demand.parquet'))
        if not model_files:
            print(f"  [Warning] No model files found in {self.base_path}")
            
        for f in model_files:
            model_name = os.path.basename(f).replace('_demand.parquet', '')
            try:
                df = pd.read_parquet(f)
                df['dt'] = pd.to_datetime(df['dt'])
                # 仅保留必要的列以节省内存
                self.model_results[model_name] = df[['store_id', 'product_id', 'dt', 'sale_amount_pred']]
                print(f"  - Model loaded: {model_name} ({len(df)} records)")
            except Exception as e:
                print(f"  [Error] Failed to load {model_name}: {e}")

    def calculate_statistics(self):
        """计算统计指标：不仅仅是总量，更关注分布特性"""
        print("Step 2: Calculating Statistics...")
        metrics = {}
        
        # 预计算非缺货日的统计特征（作为基准）
        normal_sales = self.original_data[self.original_data['stock_hour6_22_cnt'] == 0]['sale_amount']
        normal_mean = normal_sales.mean()
        normal_std = normal_sales.std()

        for model_name, pred_df in self.model_results.items():
            # 合并数据
            merged = self.original_data.merge(
                pred_df, on=['store_id', 'product_id', 'dt'], how='inner'
            )
            
            # 识别缺货日
            oos_mask = merged['stock_hour6_22_cnt'] > 0
            oos_data = merged[oos_mask]
            
            if len(oos_data) == 0:
                continue

            # 1. 基础恢复指标
            recovered_amt = oos_data['sale_amount_pred'].sum()
            original_amt = oos_data['sale_amount'].sum()
            increase_pct = (recovered_amt - original_amt) / original_amt * 100 if original_amt > 0 else 0
            
            # 2. 分布合理性 (Plausibility)
            # 计算缺货日恢复后销量 vs 非缺货日销量的 Wasserstein 距离
            # 距离越小，说明恢复的分布并没有偏离正常销售模式太远（这通常是好事，除非发生了剧烈抢购）
            ws_dist = stats.wasserstein_distance(oos_data['sale_amount_pred'], normal_sales)
            
            # 3. 异常值检测
            # 检查恢复值是否超过了正常均值的 3倍标准差 (3-sigma rule)，计算异常高值的比例
            threshold = normal_mean + 3 * normal_std
            outlier_ratio = (oos_data['sale_amount_pred'] > threshold).mean() * 100

            metrics[model_name] = {
                'Total Recovery ($)': recovered_amt - original_amt,
                'Demand Uplift (%)': increase_pct,
                'Avg Predicted / Normal Avg': oos_data['sale_amount_pred'].mean() / normal_mean,
                'Distribution Distance (Wasserstein)': ws_dist,
                'Extreme Value Ratio (%)': outlier_ratio
            }
            
        return pd.DataFrame(metrics).T

    def analyze_correlations(self):
        """分析不同模型预测结果之间的相关性"""
        print("Step 3: Analyzing Model Correlations...")
        # 选取所有缺货日的数据进行对比
        common_idx = ['store_id', 'product_id', 'dt']
        
        # 找出所有模型都预测了的缺货样本
        base_df = self.original_data[self.original_data['stock_hour6_22_cnt'] > 0][common_idx].copy()
        
        for model_name, df in self.model_results.items():
            base_df = base_df.merge(df.rename(columns={'sale_amount_pred': model_name}), 
                                  on=common_idx, how='inner')
        
        if len(base_df) > 0:
            corr_matrix = base_df[list(self.model_results.keys())].corr()
            return corr_matrix
        return None

    def plot_comprehensive_analysis(self, metrics_df, corr_matrix):
        """生成综合分析图表"""
        print("Step 4: Generating Visualizations...")
        models = metrics_df.index.tolist()
        n_models = len(models)
        
        fig = plt.figure(figsize=(20, 12))
        
        # 1. 需求提升百分比 (Demand Uplift)
        ax1 = plt.subplot(2, 3, 1)
        sns.barplot(x=metrics_df.index, y='Demand Uplift (%)', data=metrics_df, ax=ax1, palette='viridis')
        ax1.set_title('Demand Uplift by Model (%)')
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
        ax1.set_ylabel('Increase over Censored Sales (%)')

        # 2. 分布距离 (Wasserstein Distance) - 越小可能越“保守/安全”
        ax2 = plt.subplot(2, 3, 2)
        sns.barplot(x=metrics_df.index, y='Distribution Distance (Wasserstein)', data=metrics_df, ax=ax2, palette='rocket')
        ax2.set_title('Distribution Distance vs Normal Sales\n(Lower usually implies conservative recovery)')
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)

        # 3. 极端值比例
        ax3 = plt.subplot(2, 3, 3)
        sns.barplot(x=metrics_df.index, y='Extreme Value Ratio (%)', data=metrics_df, ax=ax3, palette='magma')
        ax3.set_title('Extreme Value Ratio (> 3-sigma of Normal)')
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)

        # 4. 销量分布密度对比 (KDE)
        ax4 = plt.subplot(2, 3, 4)
        
        # 绘制非缺货日的真实分布
        normal_sales = self.original_data[self.original_data['stock_hour6_22_cnt'] == 0]['sale_amount']
        sns.kdeplot(normal_sales, label='Normal Days (Observed)', fill=True, color='grey', alpha=0.3, ax=ax4)
        
        for model in models:
            # 获取该模型的缺货日预测值
            pred_df = self.model_results[model]
            merged = self.original_data[self.original_data['stock_hour6_22_cnt'] > 0].merge(
                pred_df, on=['store_id', 'product_id', 'dt']
            )
            sns.kdeplot(merged['sale_amount_pred'], label=model, ax=ax4, linewidth=2)
            
        ax4.set_xlim(0, normal_sales.quantile(0.99)) # 限制X轴范围避免长尾影响视觉
        ax4.set_title('Density Distribution: Normal vs Recovered (OOS)')
        ax4.set_xlabel('Daily Sales Amount')
        ax4.legend()

        # 5. 模型相关性热力图
        ax5 = plt.subplot(2, 3, 5)
        if corr_matrix is not None:
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=0, vmax=1, ax=ax5)
            ax5.set_title('Inter-Model Prediction Correlation')
        
        # 6. 平均每日需求恢复量对比 (Boxplot)
        ax6 = plt.subplot(2, 3, 6)
        plot_data = []
        labels = []
        for model in models:
            pred_df = self.model_results[model]
            merged = self.original_data[self.original_data['stock_hour6_22_cnt'] > 0].merge(
                pred_df, on=['store_id', 'product_id', 'dt']
            )
            # 恢复增量
            diff = merged['sale_amount_pred'] - merged['sale_amount']
            plot_data.append(diff)
            labels.append(model)
        
        ax6.boxplot(plot_data, labels=labels, showfliers=False) # 隐藏异常值以便看清中位数
        ax6.set_title('Daily Recovery Amount Distribution (No Outliers)')
        ax6.set_xticklabels(labels, rotation=45)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'model_performance_overview.png'))
        print(f"  - Saved overview plot to {self.output_dir}")

    def plot_intelligent_case_study(self, top_k=3):
        """
        智能选择案例：选择缺货天数较多且总销量较高的头部商品进行展示。
        而不是随机选择第0个。
        """
        print("Step 5: Generating Case Studies...")
        
        # 1. 筛选高价值且高频缺货的 SKU
        stats_df = self.original_data.groupby(['store_id', 'product_id']).agg({
            'sale_amount': 'sum',
            'stock_hour6_22_cnt': lambda x: (x > 0).sum(),
            'dt': 'count'
        }).reset_index()
        
        # 筛选至少有 30 天数据的商品
        stats_df = stats_df[stats_df['dt'] > 30]
        # 按 (缺货天数 * 总销量) 排序，找到最有代表性的案例
        stats_df['score'] = stats_df['sale_amount'] * stats_df['stock_hour6_22_cnt']
        top_cases = stats_df.sort_values('score', ascending=False).head(top_k)
        
        for i, (_, row) in enumerate(top_cases.iterrows()):
            store_id = row['store_id']
            prod_id = row['product_id']
            
            # 准备画图
            fig, ax = plt.subplots(figsize=(15, 6))
            
            # 获取原始数据
            raw_ts = self.original_data[
                (self.original_data['store_id'] == store_id) & 
                (self.original_data['product_id'] == prod_id)
            ].sort_values('dt')
            
            # 截取一段数据展示 (例如最近60天或者缺货最集中的时段)
            # 这里简单起见展示全部，如果太长可以截断
            if len(raw_ts) > 100:
                raw_ts = raw_ts.tail(100)
            
            # 画原始销量
            ax.plot(raw_ts['dt'], raw_ts['sale_amount'], 'k.-', label='Observed Sales', alpha=0.6, linewidth=1.5)
            
            # 标记缺货区域 (背景色)
            oos_dates = raw_ts[raw_ts['stock_hour6_22_cnt'] > 0]['dt']
            for date in oos_dates:
                ax.axvspan(date, date + pd.Timedelta(days=1), color='red', alpha=0.1, label='_nolegend_')

            # 画各个模型的预测
            colors = sns.color_palette("husl", len(self.model_results))
            for j, (model, res_df) in enumerate(self.model_results.items()):
                model_ts = res_df[
                    (res_df['store_id'] == store_id) & 
                    (res_df['product_id'] == prod_id) &
                    (res_df['dt'].isin(raw_ts['dt']))
                ].sort_values('dt')
                
                # 只画缺货那几天的预测点，连接线可以虚线
                if not model_ts.empty:
                    # 为了连贯性，我们可以merge一下
                    merged_plot = raw_ts[['dt']].merge(model_ts, on='dt', how='left')
                    ax.plot(merged_plot['dt'], merged_plot['sale_amount_pred'], 
                            linestyle='--', marker='x', markersize=5, 
                            color=colors[j], label=f'{model}', alpha=0.8)

            ax.set_title(f'Case {i+1}: Store {store_id}, Product {prod_id} (OOS Days: {row["stock_hour6_22_cnt"]})')
            ax.set_ylabel('Sales Amount')
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
            ax.grid(True, alpha=0.3)
            
            # 只在图例显示一次“OOS Zone”
            import matplotlib.patches as mpatches
            red_patch = mpatches.Patch(color='red', alpha=0.1, label='OOS Zone')
            handles, labels = ax.get_legend_handles_labels()
            handles.append(red_patch)
            labels.append("OOS Zone")
            ax.legend(handles=handles, labels=labels)

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'case_study_{i+1}.png'))
            plt.close()
            print(f"  - Saved Case Study {i+1}")

    def run(self):
        if self.original_data is None or not self.model_results:
            print("Data not loaded properly.")
            return
            
        # 1. 计算指标
        metrics_df = self.calculate_statistics()
        print("\nModel Metrics Summary:")
        print(metrics_df)
        metrics_df.to_csv(os.path.join(self.output_dir, 'model_metrics.csv'))
        
        # 2. 计算相关性
        corr_matrix = self.analyze_correlations()
        
        # 3. 绘图
        self.plot_comprehensive_analysis(metrics_df, corr_matrix)
        
        # 4. 案例研究
        self.plot_intelligent_case_study()
        
        print(f"\nDone! All results saved to {self.output_dir}/")

if __name__ == "__main__":
    comparator = ModelComparison()
    comparator.run()