import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
import os
import glob
import warnings

# 设置绘图风格
plt.rcParams.update({'font.size': 10, 'figure.dpi': 150})
sns.set_style("whitegrid")
warnings.filterwarnings('ignore')

class BatchModelAnalyzer:
    def __init__(self, result_dir='./latent_demand_recovery/exp/demand'):
        self.result_dir = result_dir
        self.original_data = None
        self.output_base = './detailed_model_reports'
        
        # 创建基础输出目录
        os.makedirs(self.output_base, exist_ok=True)
        self.load_base_data()
    
    def load_base_data(self):
        """加载一次原始数据（Ground Truth/Context）"""
        print("Step 1: Loading Original Dataset...")
        try:
            dataset = load_dataset("Dingdong-Inc/FreshRetailNet-50K")
            self.original_data = dataset['train'].to_pandas()
            self.original_data['dt'] = pd.to_datetime(self.original_data['dt'])
            self.original_data = self.original_data.sort_values(by=['store_id', 'product_id', 'dt'])
            print(f"  - Loaded {len(self.original_data):,} records successfully.")
        except Exception as e:
            print(f"  [Error] Failed to load dataset: {e}")
            exit(1)

    def process_all_models(self):
        """扫描并处理所有模型文件"""
        files = glob.glob(os.path.join(self.result_dir, '*_demand.parquet'))
        if not files:
            print(f"[Warning] No model results found in {self.result_dir}")
            return

        print(f"\nStep 2: Found {len(files)} models. Starting batch analysis...")
        
        for file_path in files:
            model_name = os.path.basename(file_path).replace('_demand.parquet', '')
            print(f"\n--- Analyzing Model: {model_name} ---")
            
            try:
                # 1. 加载模型预测数据
                pred_df = pd.read_parquet(file_path)
                pred_df['dt'] = pd.to_datetime(pred_df['dt'])
                
                # 2. 创建该模型的专属输出目录
                model_out_dir = os.path.join(self.output_base, model_name)
                os.makedirs(model_out_dir, exist_ok=True)
                
                # 3. 合并数据
                merged_data = self.original_data.merge(
                    pred_df[['store_id', 'product_id', 'dt', 'sale_amount_pred']],
                    on=['store_id', 'product_id', 'dt'],
                    how='inner'
                )
                
                # 标记缺货
                merged_data['is_oos'] = merged_data['stock_hour6_22_cnt'] > 0
                
                # 4. 生成分析
                self.generate_dashboard(merged_data, model_name, model_out_dir)
                self.generate_case_studies(merged_data, model_name, model_out_dir)
                self.save_statistics(merged_data, model_name, model_out_dir)
                
                print(f"  -> Report saved to: {model_out_dir}/")
                
            except Exception as e:
                print(f"  [Error] Failed to process {model_name}: {e}")

    def generate_dashboard(self, df, model_name, out_dir):
        """生成 3x3 综合分析仪表盘"""
        oos_data = df[df['is_oos']]
        if len(oos_data) == 0:
            return

        fig = plt.figure(figsize=(20, 15))
        fig.suptitle(f'Model Analysis Report: {model_name}', fontsize=16, y=0.92)

        # 1. 销售分布密度对比 (KDE)
        ax1 = plt.subplot(3, 3, 1)
        sns.kdeplot(df['sale_amount'], label='Observed (All)', color='grey', fill=True, alpha=0.3, ax=ax1)
        sns.kdeplot(oos_data['sale_amount_pred'], label='Recovered (OOS Only)', color='blue', ax=ax1)
        ax1.set_title('Density: Observed vs Recovered Demand')
        ax1.set_xlabel('Sales Amount')
        ax1.legend()

        # 2. 缺货日销售箱线图
        ax2 = plt.subplot(3, 3, 2)
        data_to_plot = [oos_data['sale_amount'], oos_data['sale_amount_pred']]
        ax2.boxplot(data_to_plot, labels=['Observed\n(Censored)', 'Recovered\n(Predicted)'], 
                    patch_artist=True, boxprops=dict(facecolor="lightblue"))
        ax2.set_title('Sales Volume Comparison on OOS Days')
        ax2.set_ylabel('Sales Amount')

        # 3. 恢复增量分布
        ax3 = plt.subplot(3, 3, 3)
        recovery_amt = oos_data['sale_amount_pred'] - oos_data['sale_amount']
        sns.histplot(recovery_amt, bins=50, kde=True, color='green', ax=ax3)
        ax3.axvline(recovery_amt.mean(), color='red', linestyle='--', label=f'Mean: {recovery_amt.mean():.1f}')
        ax3.set_title('Distribution of Recovery Amount (Delta)')
        ax3.set_xlabel('Added Sales ($)')
        ax3.legend()

        # 4. 周度模式 (Weekly Pattern) - 新增维度！
        ax4 = plt.subplot(3, 3, 4)
        df['day_of_week'] = df['dt'].dt.day_name()
        # 按周一到周日排序
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekly_pattern = df.groupby('day_of_week')[['sale_amount', 'sale_amount_pred']].mean().reindex(days_order)
        
        x = np.arange(len(days_order))
        width = 0.35
        ax4.bar(x - width/2, weekly_pattern['sale_amount'], width, label='Observed', color='grey', alpha=0.7)
        ax4.bar(x + width/2, weekly_pattern['sale_amount_pred'], width, label='Recovered', color='orange', alpha=0.7)
        ax4.set_xticks(x)
        ax4.set_xticklabels(days_order, rotation=45)
        ax4.set_title('Weekly Seasonality Check')
        ax4.legend()

        # 5. 月度趋势
        ax5 = plt.subplot(3, 3, 5)
        monthly = df.resample('M', on='dt')[['sale_amount', 'sale_amount_pred']].sum()
        ax5.plot(monthly.index, monthly['sale_amount'], 'o-', label='Observed', color='grey')
        ax5.plot(monthly.index, monthly['sale_amount_pred'], 's--', label='Recovered', color='blue')
        ax5.set_title('Monthly Total Sales Trend')
        ax5.legend()
        plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45)

        # 6. 恢复率 vs 缺货频率 (散点图)
        ax6 = plt.subplot(3, 3, 6)
        store_stats = oos_data.groupby('store_id').agg({
            'sale_amount': 'sum',
            'sale_amount_pred': 'sum',
            'dt': 'count' # OOS days count
        })
        store_stats['recovery_pct'] = (store_stats['sale_amount_pred'] - store_stats['sale_amount']) / store_stats['sale_amount']
        
        sns.scatterplot(data=store_stats, x='dt', y='recovery_pct', alpha=0.6, ax=ax6, size='sale_amount', sizes=(20, 200))
        ax6.set_title('Recovery Rate vs OOS Frequency (Store Level)')
        ax6.set_xlabel('Number of OOS Days')
        ax6.set_ylabel('Recovery Rate (%)')

        # 7. 预测值 vs 原始值 (Log Scale)
        ax7 = plt.subplot(3, 3, 7)
        # 随机采样避免卡顿
        sample = oos_data.sample(min(2000, len(oos_data)))
        ax7.scatter(sample['sale_amount'], sample['sale_amount_pred'], alpha=0.4, s=10)
        # 画对角线
        lims = [0, max(sample['sale_amount'].max(), sample['sale_amount_pred'].max())]
        ax7.plot(lims, lims, 'r--', alpha=0.75, zorder=0)
        ax7.set_title('Pred vs Orig on OOS Days')
        ax7.set_xlabel('Original Sales')
        ax7.set_ylabel('Recovered Demand')

        # 8. 异常检测 (Z-Score)
        ax8 = plt.subplot(3, 3, 8)
        # 计算恢复后的 Z-Score
        z_scores = (oos_data['sale_amount_pred'] - oos_data['sale_amount_pred'].mean()) / oos_data['sale_amount_pred'].std()
        sns.histplot(z_scores, bins=30, ax=ax8, color='purple')
        ax8.set_title('Z-Score Distribution of Predictions')
        ax8.set_xlabel('Z-Score')
        ax8.set_ylabel('Count')

        # 9. 关键指标文本
        ax9 = plt.subplot(3, 3, 9)
        total_rec = (oos_data['sale_amount_pred'] - oos_data['sale_amount']).sum()
        uplift = total_rec / df['sale_amount'].sum() * 100
        
        text_str = f"""
        MODEL SUMMARY: {model_name}
        ---------------------------
        Total Records: {len(df):,}
        OOS Days: {len(oos_data):,} ({len(oos_data)/len(df)*100:.1f}%)
        
        Total Recovery: ${total_rec:,.0f}
        Global Uplift: +{uplift:.2f}%
        
        Avg Sales (Normal): ${df[~df['is_oos']]['sale_amount'].mean():.2f}
        Avg Sales (OOS Orig): ${oos_data['sale_amount'].mean():.2f}
        Avg Sales (OOS Pred): ${oos_data['sale_amount_pred'].mean():.2f}
        """
        ax9.text(0.1, 0.5, text_str, fontsize=12, family='monospace', va='center')
        ax9.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'{model_name}_dashboard.png'))
        plt.close()

    def generate_case_studies(self, df, model_name, out_dir):
        """生成精选案例（Top 3 高价值缺货商品）"""
        oos_data = df[df['is_oos']]
        if len(oos_data) == 0: return

        # 寻找最有展示价值的案例：缺货天数多 且 总恢复量大
        # 评分 = 缺货天数 * log(总恢复金额)
        case_score = oos_data.groupby(['store_id', 'product_id']).agg({
            'sale_amount': 'sum',
            'sale_amount_pred': 'sum',
            'dt': 'count'
        })
        case_score['recovery_amt'] = case_score['sale_amount_pred'] - case_score['sale_amount']
        case_score['score'] = case_score['dt'] * np.log1p(case_score['recovery_amt'])
        
        top_cases = case_score.nlargest(3, 'score').reset_index()

        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        if len(top_cases) < 3: axes = [axes] # Handle edge case

        for i, (_, row) in enumerate(top_cases.iterrows()):
            ax = axes[i] if len(top_cases) > 1 else axes
            
            store_id, prod_id = row['store_id'], row['product_id']
            
            # 获取该商品全量数据
            prod_ts = df[(df['store_id'] == store_id) & (df['product_id'] == prod_id)].sort_values('dt')
            
            # 只展示最近 60 天或有数据的时段
            plot_data = prod_ts.tail(60)
            
            ax.plot(plot_data['dt'], plot_data['sale_amount'], 'o-', color='grey', label='Observed', alpha=0.6)
            ax.plot(plot_data['dt'], plot_data['sale_amount_pred'], 'x--', color='blue', label='Recovered', linewidth=1.5)
            
            # 高亮缺货日
            oos_dates = plot_data[plot_data['is_oos']]['dt']
            for d in oos_dates:
                ax.axvline(d, color='red', alpha=0.15, linewidth=10) # 粗线作为背景
            
            ax.set_title(f"Case {i+1}: Store {store_id} Product {prod_id} (OOS Days: {row['dt']}, Recovery: +${row['recovery_amt']:.0f})")
            ax.set_ylabel('Sales')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'{model_name}_cases.png'))
        plt.close()

    def save_statistics(self, df, model_name, out_dir):
        """保存详细统计指标到CSV"""
        oos_data = df[df['is_oos']]
        
        # 1. 店铺维度的统计
        store_metrics = oos_data.groupby('store_id').agg({
            'sale_amount': 'sum',
            'sale_amount_pred': 'sum',
            'dt': 'count'
        }).reset_index()
        store_metrics.columns = ['store_id', 'orig_sales', 'pred_sales', 'oos_days']
        store_metrics['uplift'] = store_metrics['pred_sales'] - store_metrics['orig_sales']
        store_metrics['uplift_pct'] = store_metrics['uplift'] / store_metrics['orig_sales']
        
        store_metrics.sort_values('uplift', ascending=False).to_csv(
            os.path.join(out_dir, f'{model_name}_store_metrics.csv'), index=False
        )

# 运行分析
if __name__ == "__main__":
    analyzer = BatchModelAnalyzer()
    analyzer.process_all_models()
    
    print("\n" + "="*60)
    print("BATCH ANALYSIS COMPLETED")
    print(f"Check the output folder: ./detailed_model_reports/")
    print("Structure:")
    print("  ├── DLinear/")
    print("  │   ├── DLinear_dashboard.png")
    print("  │   ├── DLinear_cases.png")
    print("  │   └── DLinear_store_metrics.csv")
    print("  ├── TimesNet/...")
    print("  └── ...")
    print("="*60)