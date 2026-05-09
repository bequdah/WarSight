import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def get_best_metrics(results_dir):
    """
    البحث عن أفضل تجربة (أعلى mAP) داخل مجلد نتائج معين.
    """
    best_map = -1
    best_row = None
    best_exp = ""
    
    exp_folders = [f for f in os.listdir(results_dir) if f.startswith('exp')]
    for exp in exp_folders:
        csv_path = os.path.join(results_dir, exp, 'results.csv')
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                df.columns = [c.strip() for c in df.columns]
                last_row = df.iloc[-1]
                
                # نعتمد mAP50 كمعيار للأفضل
                current_map = last_row['metrics/mAP50(B)']
                if current_map > best_map:
                    best_map = current_map
                    best_row = last_row.copy()
                    best_exp = exp
            except:
                continue
    
    if best_row is not None:
        best_row['exp_name'] = best_exp
    return best_row

def compare_v8_vs_v26(v8_dir, v26_dir):
    """
    مقارنة "أفضل ما في v8" ضد "أفضل ما في v26".
    """
    print("🏆 Initiating Final Model Showdown: YOLOv8 vs YOLOv26")
    
    best_v8 = get_best_metrics(v8_dir)
    best_v26 = get_best_metrics(v26_dir)
    
    results = []
    if best_v8 is not None:
        best_v8['Model'] = f"YOLOv8 ({best_v8['exp_name']})"
        results.append(best_v8)
    
    if best_v26 is not None:
        best_v26['Model'] = f"YOLOv26 ({best_v26['exp_name']})"
        results.append(best_v26)
        
    if not results:
        print("❌ No results found to compare.")
        return

    df = pd.DataFrame(results)
    
    # اختيار المقاييس للمقارنة
    metrics = ['metrics/mAP50(B)', 'metrics/precision(B)', 'metrics/recall(B)']
    available_metrics = [m for m in metrics if m in df.columns]
    
    print("\n📊 --- FINAL COMPARISON TABLE ---")
    print(df[['Model'] + available_metrics].to_string(index=False))
    
    # رسم المخطط النهائي
    plt.figure(figsize=(12, 7))
    melted = df.melt(id_vars='Model', value_vars=available_metrics, var_name='Metric', value_name='Score')
    
    sns.set_style("darkgrid")
    sns.barplot(data=melted, x='Metric', y='Score', hue='Model', palette='rocket')
    
    plt.title('Final Showdown: YOLOv8 vs YOLOv26 (Best Experiments Only)', fontsize=14, fontweight='bold')
    plt.ylim(0, 1.0)
    plt.legend(title='Tactical Model')
    
    # حفظ النتيجة في مجلد الـ evaluation
    output_path = os.path.join(os.path.dirname(v8_dir), 'evaluation', 'final_model_comparison.png')
    plt.savefig(output_path)
    print(f"\n🔥 Final comparison chart saved to: {output_path}")
    plt.show()

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    v8_path = os.path.join(project_root, 'results', 'YOLOv8')
    v26_path = os.path.join(project_root, 'results', 'YOLOv26')
    
    compare_v8_vs_v26(v8_path, v26_path)
