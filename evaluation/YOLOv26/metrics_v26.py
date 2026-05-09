import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_yolov26_experiments(results_dir):
    """
    تحليل ومقارنة نتائج تجارب YOLOv26 فقط.
    """
    print("📊 Evaluating YOLOv26 Experiments...")
    all_metrics = []
    
    exp_folders = ['exp1', 'exp2', 'exp3']
    
    for exp in exp_folders:
        csv_path = os.path.join(results_dir, exp, 'results.csv')
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                df.columns = [c.strip() for c in df.columns]
                
                last_row = df.iloc[-1].copy()
                last_row['experiment'] = exp
                all_metrics.append(last_row)
            except Exception as e:
                print(f"⚠️ Error reading {exp}: {e}")

    if not all_metrics:
        print("❌ No YOLOv26 results found.")
        return

    comparison_df = pd.DataFrame(all_metrics)
    cols = ['experiment', 'metrics/mAP50(B)', 'metrics/precision(B)', 'metrics/recall(B)']
    available_cols = [c for c in cols if c in comparison_df.columns]
    
    summary = comparison_df[available_cols]
    print("\n--- YOLOv26 Performance Summary ---")
    print(summary.to_string(index=False))
    
    plt.figure(figsize=(10, 5))
    melted = summary.melt(id_vars='experiment', var_name='Metric', value_name='Score')
    sns.barplot(data=melted, x='Metric', y='Score', hue='experiment', palette='magma')
    plt.title('YOLOv26 Experiment Comparison (Baseline vs Preprocessed vs Tuned)')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', alpha=0.3)
    
    plt.savefig(os.path.join(results_dir, 'v26_comparison.png'))
    print(f"\n✅ Chart saved to {results_dir}/v26_comparison.png")
    plt.show()

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    v26_results_path = os.path.join(project_root, 'results', 'YOLOv26')
    
    evaluate_yolov26_experiments(v26_results_path)
