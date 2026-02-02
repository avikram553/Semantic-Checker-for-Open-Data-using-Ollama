"""
Visualization Script for Semantic Checker Analysis Results
Creates interactive and static plots comparing Levenshtein, Jaro-Winkler, and SBERT
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
import warnings
import sys
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Column name mapping for different analysis methods
COLUMN_MAPPING = {
    'levenshtein': 'similarity_score',
    'jaro_winkler': 'jaro_winkler_similarity',
    'sbert': 'sbert_similarity'
}

def get_similarity_column(key):
    """Get the correct similarity column name for a given method."""
    return COLUMN_MAPPING.get(key, f'{key}_similarity')

def load_results():
    """Load all analysis results."""
    data_dir = Path("data")
    
    results = {}
    
    # Load Levenshtein results
    lev_file = data_dir / "levenshtein_results.csv"
    if lev_file.exists():
        results['levenshtein'] = pd.read_csv(lev_file)
        print(f"✅ Loaded Levenshtein results: {len(results['levenshtein'])} pairs")
    else:
        print(f"⚠️  Levenshtein results not found: {lev_file}")
    
    # Load Jaro-Winkler results
    jw_file = data_dir / "jaro_winkler_results.csv"
    if jw_file.exists():
        results['jaro_winkler'] = pd.read_csv(jw_file)
        print(f"✅ Loaded Jaro-Winkler results: {len(results['jaro_winkler'])} pairs")
    else:
        print(f"⚠️  Jaro-Winkler results not found: {jw_file}")
    
    # Load SBERT results
    sbert_file = data_dir / "sbert_results.csv"
    if sbert_file.exists():
        results['sbert'] = pd.read_csv(sbert_file)
        print(f"✅ Loaded SBERT results: {len(results['sbert'])} pairs")
    else:
        print(f"⚠️  SBERT results not found: {sbert_file}")
    
    return results

def create_distribution_plots(results):
    """Create distribution plots for similarity scores."""
    print("\n📊 Creating distribution plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Similarity Score Distributions by Method and Match Status', fontsize=16, fontweight='bold')
    
    method_names = ['levenshtein', 'jaro_winkler', 'sbert']
    display_names = ['Levenshtein', 'Jaro-Winkler', 'SBERT']
    
    for idx, (key, name) in enumerate(zip(method_names, display_names)):
        if key not in results:
            continue
        
        df = results[key]
        col = get_similarity_column(key)
        
        # Overall distribution
        ax = axes[0, idx]
        ax.hist(df[col], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax.set_xlabel('Similarity Score')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{name} - Overall Distribution')
        ax.axvline(df[col].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df[col].mean():.3f}')
        ax.legend()
        
        # Distribution by match status
        ax = axes[1, idx]
        matches = df[df['match'] == True][col]
        non_matches = df[df['match'] == False][col]
        
        ax.hist(matches, bins=30, alpha=0.6, color='green', label='Matches (True)', edgecolor='black')
        ax.hist(non_matches, bins=30, alpha=0.6, color='red', label='Non-matches (False)', edgecolor='black')
        ax.set_xlabel('Similarity Score')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{name} - By Match Status')
        ax.legend()
        ax.axvline(matches.mean(), color='darkgreen', linestyle='--', linewidth=2)
        ax.axvline(non_matches.mean(), color='darkred', linestyle='--', linewidth=2)
    
    plt.tight_layout()
    output_file = "visualizations/distribution_plots.png"
    Path("visualizations").mkdir(exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   Saved: {output_file}")
    plt.close()

def create_comparison_boxplot(results):
    """Create boxplot comparing all methods."""
    print("\n📊 Creating comparison boxplot...")
    
    # Prepare data
    plot_data = []
    
    method_info = [
        ('levenshtein', 'Levenshtein'),
        ('jaro_winkler', 'Jaro-Winkler'),
        ('sbert', 'SBERT')
    ]
    
    for key, name in method_info:
        if key in results:
            df = results[key]
            col = get_similarity_column(key)
            for match in [True, False]:
                subset = df[df['match'] == match]
                for score in subset[col]:
                    plot_data.append({
                        'Method': name,
                        'Match': 'Match' if match else 'Non-match',
                        'Similarity': score
                    })
    
    plot_df = pd.DataFrame(plot_data)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.boxplot(data=plot_df, x='Method', y='Similarity', hue='Match', ax=ax, palette=['green', 'red'])
    ax.set_title('Similarity Score Comparison Across Methods', fontsize=16, fontweight='bold')
    ax.set_ylabel('Similarity Score', fontsize=12)
    ax.set_xlabel('Method', fontsize=12)
    ax.legend(title='Ground Truth', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_file = "visualizations/comparison_boxplot.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   Saved: {output_file}")
    plt.close()

def create_confusion_matrices(results):
    """Create confusion matrices for different thresholds."""
    print("\n📊 Creating confusion matrices...")
    
    methods = [
        ('levenshtein', 'Levenshtein', 0.5),
        ('jaro_winkler', 'Jaro-Winkler', 0.5),
        ('sbert', 'SBERT', 0.7)
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Confusion Matrices at Optimal Thresholds', fontsize=16, fontweight='bold')
    
    for idx, (key, name, threshold) in enumerate(methods):
        if key not in results:
            continue
        
        df = results[key]
        col = get_similarity_column(key)
        df['predicted'] = df[col] >= threshold
        
        # Calculate confusion matrix
        tp = len(df[(df['predicted'] == True) & (df['match'] == True)])
        fp = len(df[(df['predicted'] == True) & (df['match'] == False)])
        tn = len(df[(df['predicted'] == False) & (df['match'] == False)])
        fn = len(df[(df['predicted'] == False) & (df['match'] == True)])
        
        cm = np.array([[tn, fp], [fn, tp]])
        
        ax = axes[idx]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                   xticklabels=['Predicted No', 'Predicted Yes'],
                   yticklabels=['Actual No', 'Actual Yes'],
                   cbar=True, square=True, linewidths=2)
        
        accuracy = (tp + tn) / len(df)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        ax.set_title(f'{name} (Threshold: {threshold})\nF1: {f1:.3f} | Acc: {accuracy:.3f}', 
                    fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    output_file = "visualizations/confusion_matrices.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   Saved: {output_file}")
    plt.close()

def create_roc_curves(results):
    """Create ROC-like curves showing precision, recall, and F1 at different thresholds."""
    print("\n📊 Creating threshold performance curves...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Performance Metrics Across Thresholds', fontsize=16, fontweight='bold')
    
    methods = [
        ('levenshtein', 'Levenshtein'),
        ('jaro_winkler', 'Jaro-Winkler'),
        ('sbert', 'SBERT')
    ]
    
    for idx, (key, name) in enumerate(methods):
        if key not in results:
            continue
        
        df = results[key]
        col = get_similarity_column(key)
        thresholds = np.arange(0.0, 1.01, 0.01)
        
        precisions = []
        recalls = []
        f1_scores = []
        accuracies = []
        
        for threshold in thresholds:
            df['predicted'] = df[col] >= threshold
            
            tp = len(df[(df['predicted'] == True) & (df['match'] == True)])
            fp = len(df[(df['predicted'] == True) & (df['match'] == False)])
            tn = len(df[(df['predicted'] == False) & (df['match'] == False)])
            fn = len(df[(df['predicted'] == False) & (df['match'] == True)])
            
            accuracy = (tp + tn) / len(df) if len(df) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
            accuracies.append(accuracy)
        
        ax = axes[idx]
        ax.plot(thresholds, precisions, label='Precision', linewidth=2, color='blue')
        ax.plot(thresholds, recalls, label='Recall', linewidth=2, color='green')
        ax.plot(thresholds, f1_scores, label='F1-Score', linewidth=2, color='red')
        ax.plot(thresholds, accuracies, label='Accuracy', linewidth=2, color='orange', linestyle='--')
        
        # Mark best F1
        best_f1_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_f1_idx]
        best_f1 = f1_scores[best_f1_idx]
        ax.axvline(best_threshold, color='red', linestyle=':', alpha=0.5)
        ax.scatter([best_threshold], [best_f1], color='red', s=100, zorder=5)
        ax.text(best_threshold, best_f1, f'  Best: {best_threshold:.2f}', fontsize=9)
        
        ax.set_xlabel('Threshold', fontsize=11)
        ax.set_ylabel('Score', fontsize=11)
        ax.set_title(f'{name}', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    output_file = "visualizations/threshold_curves.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   Saved: {output_file}")
    plt.close()

def create_interactive_dashboard(results):
    """Create an interactive HTML dashboard with Plotly."""
    print("\n📊 Creating interactive dashboard...")
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Method Comparison - Matches vs Non-matches',
            'Similarity Score Distributions',
            'F1-Score vs Threshold',
            'Precision vs Recall Trade-off',
            'Performance Metrics Summary',
            'Category-wise Performance'
        ),
        specs=[
            [{"type": "box"}, {"type": "violin"}],
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "bar"}, {"type": "bar"}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    colors = {'levenshtein': '#1f77b4', 'jaro_winkler': '#ff7f0e', 'sbert': '#2ca02c'}
    method_names = {'levenshtein': 'Levenshtein', 'jaro_winkler': 'Jaro-Winkler', 'sbert': 'SBERT'}
    
    # 1. Box plot comparison
    for key in results:
        df = results[key]
        col = get_similarity_column(key)
        
        for match in [True, False]:
            subset = df[df['match'] == match]
            fig.add_trace(
                go.Box(
                    y=subset[col],
                    name=f"{method_names[key]} ({'Match' if match else 'Non-match'})",
                    marker_color=colors[key],
                    opacity=0.7 if match else 0.4
                ),
                row=1, col=1
            )
    
    # 2. Violin plot distributions
    for key in results:
        df = results[key]
        col = get_similarity_column(key)
        
        fig.add_trace(
            go.Violin(
                y=df[col],
                name=method_names[key],
                marker_color=colors[key],
                box_visible=True,
                meanline_visible=True
            ),
            row=1, col=2
        )
    
    # 3. F1-Score vs Threshold
    for key in results:
        df = results[key]
        col = get_similarity_column(key)
        
        thresholds = np.arange(0.0, 1.01, 0.02)
        f1_scores = []
        
        for threshold in thresholds:
            df['predicted'] = df[col] >= threshold
            tp = len(df[(df['predicted'] == True) & (df['match'] == True)])
            fp = len(df[(df['predicted'] == True) & (df['match'] == False)])
            fn = len(df[(df['predicted'] == False) & (df['match'] == True)])
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            f1_scores.append(f1)
        
        fig.add_trace(
            go.Scatter(
                x=thresholds,
                y=f1_scores,
                name=method_names[key],
                mode='lines',
                line=dict(color=colors[key], width=3)
            ),
            row=2, col=1
        )
    
    # 4. Precision-Recall curve
    for key in results:
        df = results[key]
        col = get_similarity_column(key)
        
        thresholds = np.arange(0.0, 1.01, 0.02)
        precisions = []
        recalls = []
        
        for threshold in thresholds:
            df['predicted'] = df[col] >= threshold
            tp = len(df[(df['predicted'] == True) & (df['match'] == True)])
            fp = len(df[(df['predicted'] == True) & (df['match'] == False)])
            fn = len(df[(df['predicted'] == False) & (df['match'] == True)])
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            precisions.append(precision)
            recalls.append(recall)
        
        fig.add_trace(
            go.Scatter(
                x=recalls,
                y=precisions,
                name=method_names[key],
                mode='lines',
                line=dict(color=colors[key], width=3)
            ),
            row=2, col=2
        )
    
    # 5. Performance summary bar chart
    metrics_data = []
    for key in results:
        df = results[key]
        col = get_similarity_column(key)
        
        # Find best F1 threshold
        best_f1 = 0
        best_metrics = {}
        
        for threshold in np.arange(0.0, 1.01, 0.01):
            df['predicted'] = df[col] >= threshold
            tp = len(df[(df['predicted'] == True) & (df['match'] == True)])
            fp = len(df[(df['predicted'] == True) & (df['match'] == False)])
            tn = len(df[(df['predicted'] == False) & (df['match'] == False)])
            fn = len(df[(df['predicted'] == False) & (df['match'] == True)])
            
            accuracy = (tp + tn) / len(df)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            if f1 > best_f1:
                best_f1 = f1
                best_metrics = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
        
        for metric, value in best_metrics.items():
            metrics_data.append({'Method': method_names[key], 'Metric': metric.title(), 'Value': value})
    
    metrics_df = pd.DataFrame(metrics_data)
    
    for metric in ['Accuracy', 'Precision', 'Recall', 'F1']:
        subset = metrics_df[metrics_df['Metric'] == metric]
        fig.add_trace(
            go.Bar(
                x=subset['Method'],
                y=subset['Value'],
                name=metric,
                text=[f"{v:.3f}" for v in subset['Value']],
                textposition='auto'
            ),
            row=3, col=1
        )
    
    # 6. Category-wise performance (using first available result)
    first_key = list(results.keys())[0]
    if 'category' in results[first_key].columns:
        category_data = []
        for key in results:
            df = results[key]
            col = get_similarity_column(key)
            
            for category in df['category'].unique():
                cat_subset = df[df['category'] == category]
                match_subset = cat_subset[cat_subset['match'] == True]
                avg_sim = match_subset[col].mean() if len(match_subset) > 0 else 0
                
                category_data.append({
                    'Method': method_names[key],
                    'Category': category,
                    'Avg Similarity': avg_sim
                })
        
        cat_df = pd.DataFrame(category_data)
        
        for key in results:
            subset = cat_df[cat_df['Method'] == method_names[key]]
            fig.add_trace(
                go.Bar(
                    x=subset['Category'],
                    y=subset['Avg Similarity'],
                    name=method_names[key],
                    marker_color=colors[key]
                ),
                row=3, col=2
            )
    
    # Update layout
    fig.update_layout(
        title_text="Semantic Checker Analysis Dashboard",
        title_font_size=24,
        showlegend=True,
        height=1400,
        template="plotly_white"
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Threshold", row=2, col=1)
    fig.update_yaxes(title_text="F1-Score", row=2, col=1)
    
    fig.update_xaxes(title_text="Recall", row=2, col=2)
    fig.update_yaxes(title_text="Precision", row=2, col=2)
    
    fig.update_yaxes(title_text="Similarity Score", row=1, col=1)
    fig.update_yaxes(title_text="Similarity Score", row=1, col=2)
    
    # Save
    output_file = "visualizations/interactive_dashboard.html"
    fig.write_html(output_file)
    print(f"   Saved: {output_file}")
    print(f"   🌐 Open in browser to interact with the dashboard!")

def create_summary_report(results):
    """Create a summary comparison table."""
    print("\n📊 Creating summary report...")
    
    summary_data = []
    
    for key in results:
        df = results[key]
        col = get_similarity_column(key)
        
        # Calculate statistics
        matches = df[df['match'] == True]
        non_matches = df[df['match'] == False]
        
        # Find best F1 threshold
        best_f1 = 0
        best_threshold = 0
        best_metrics = {}
        
        for threshold in np.arange(0.0, 1.01, 0.01):
            df['predicted'] = df[col] >= threshold
            tp = len(df[(df['predicted'] == True) & (df['match'] == True)])
            fp = len(df[(df['predicted'] == True) & (df['match'] == False)])
            tn = len(df[(df['predicted'] == False) & (df['match'] == False)])
            fn = len(df[(df['predicted'] == False) & (df['match'] == True)])
            
            accuracy = (tp + tn) / len(df)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_metrics = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                }
        
        summary_data.append({
            'Method': key.replace('_', ' ').title(),
            'Avg Similarity (All)': df[col].mean(),
            'Avg Similarity (Matches)': matches[col].mean(),
            'Avg Similarity (Non-matches)': non_matches[col].mean(),
            'Separation Gap': matches[col].mean() - non_matches[col].mean(),
            'Best Threshold': best_threshold,
            'Best F1-Score': best_metrics['f1'],
            'Precision @ Best': best_metrics['precision'],
            'Recall @ Best': best_metrics['recall'],
            'Accuracy @ Best': best_metrics['accuracy']
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Create styled table
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.axis('tight')
    ax.axis('off')
    
    # Format data for display
    display_data = summary_df.copy()
    for col in display_data.columns:
        if col != 'Method':
            display_data[col] = display_data[col].apply(lambda x: f"{x:.4f}" if isinstance(x, float) else x)
    
    table = ax.table(cellText=display_data.values,
                    colLabels=display_data.columns,
                    cellLoc='center',
                    loc='center',
                    colColours=['lightgray'] * len(display_data.columns))
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Color best values
    for i in range(len(summary_data)):
        # Highlight best F1 score
        best_f1_idx = summary_data.index(max(summary_data, key=lambda x: x['Best F1-Score']))
        if i == best_f1_idx:
            for j in range(len(display_data.columns)):
                table[(i+1, j)].set_facecolor('#90EE90')
    
    plt.title('Performance Summary Comparison', fontsize=16, fontweight='bold', pad=20)
    
    output_file = "visualizations/summary_report.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   Saved: {output_file}")
    plt.close()
    
    # Also save as CSV
    csv_file = "visualizations/summary_report.csv"
    summary_df.to_csv(csv_file, index=False)
    print(f"   Saved: {csv_file}")

def create_simple_tables(results, methods=None, tough_threshold: float = 0.7):
    """Create simple CSV tables and console summaries for:
    - True positives (correct positives)
    - False negatives (missed matches)
    - Tough matches (true matches with low similarity)

    Saves CSVs into `visualizations/` and prints brief tables to console.
    """
    print("\n📋 Creating simple tables (TP / FN / Tough matches) ...")
    out_dir = Path("visualizations")
    out_dir.mkdir(exist_ok=True)

    if methods is None:
        methods = list(results.keys())

    default_thresholds = {'sbert': 0.7, 'jaro_winkler': 0.5, 'levenshtein': 0.5}

    for key in methods:
        if key not in results:
            print(f" - Skipping {key}: results not available")
            continue

        df = results[key].copy()
        col = get_similarity_column(key)

        if col not in df.columns:
            print(f" - Skipping {key}: similarity column '{col}' not found")
            continue

        threshold = default_thresholds.get(key, 0.5)
        df['predicted'] = df[col] >= threshold

        tp_df = df[(df['predicted'] == True) & (df['match'] == True)].copy()
        fn_df = df[(df['predicted'] == False) & (df['match'] == True)].copy()
        tough_df = df[(df['match'] == True) & (df[col] < tough_threshold)].copy()

        tp_csv = out_dir / f"{key}_true_positives.csv"
        fn_csv = out_dir / f"{key}_false_negatives.csv"
        tough_csv = out_dir / f"{key}_tough_matches.csv"

        tp_df.to_csv(tp_csv, index=False, encoding='utf-8')
        fn_df.to_csv(fn_csv, index=False, encoding='utf-8')
        tough_df.to_csv(tough_csv, index=False, encoding='utf-8')

        # Print concise tables (top 10 rows) to console
        display_cols = [c for c in ['id', 'attribute1', 'attribute2', col, 'category', 'reasoning'] if c in df.columns]

        print(f"\nMethod: {key}  (threshold={threshold}, tough<{tough_threshold})")
        print(f"  True Positives: {len(tp_df)} -> {tp_csv}")
        if len(tp_df) > 0:
            print(tp_df[display_cols].head(10).to_string(index=False))

        print(f"\n  False Negatives (missed matches): {len(fn_df)} -> {fn_csv}")
        if len(fn_df) > 0:
            print(fn_df[display_cols].head(10).to_string(index=False))

        print(f"\n  Tough Matches (true matches with low similarity): {len(tough_df)} -> {tough_csv}")
        if len(tough_df) > 0:
            print(tough_df[display_cols].head(10).to_string(index=False))

    print("\n✅ Simple tables generated in 'visualizations/'")

def main():
    """Main function to generate all visualizations."""
    print("=" * 80)
    print("SEMANTIC CHECKER VISUALIZATION GENERATOR")
    print("=" * 80)
    print()
    
    # Create output directory
    Path("visualizations").mkdir(exist_ok=True)
    
    # Load results
    print("Loading analysis results...")
    results = load_results()
    
    if not results:
        print("\n❌ No results found! Please run the analysis scripts first:")
        print("   - python levenshtein_analysis.py")
        print("   - python jaro_winkler_analysis.py")
        print("   - python sbert_analysis.py")
        return
    
    print(f"\n✅ Loaded {len(results)} result files")
    print()
    
    # If user requested only simple tables, produce them and exit
    if '--tables' in sys.argv:
        create_simple_tables(results)
        return

    # Generate visualizations
    try:
        create_distribution_plots(results)
        create_comparison_boxplot(results)
        create_confusion_matrices(results)
        create_roc_curves(results)
        create_summary_report(results)
        create_interactive_dashboard(results)
    except Exception as e:
        print(f"\n❌ Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 80)
    print("✅ VISUALIZATION COMPLETE!")
    print("=" * 80)
    print("\nGenerated files in 'visualizations/' folder:")
    print("  📊 distribution_plots.png - Score distributions")
    print("  📊 comparison_boxplot.png - Method comparison")
    print("  📊 confusion_matrices.png - Classification performance")
    print("  📊 threshold_curves.png - Performance vs threshold")
    print("  📊 summary_report.png - Performance summary table")
    print("  📊 summary_report.csv - Summary data (CSV)")
    print("  🌐 interactive_dashboard.html - Interactive dashboard")
    print("\n💡 Open 'interactive_dashboard.html' in your browser for best experience!")
    print("=" * 80)

if __name__ == "__main__":
    main()
