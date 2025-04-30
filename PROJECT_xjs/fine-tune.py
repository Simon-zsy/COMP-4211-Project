import pandas as pd
from collections import defaultdict

def compare_ner_tags(original_csv_path, predicted_csv_path, output_csv_path):
    """
    对比原始标注和模型预测的NER标签
    
    参数:
        original_csv_path: 原始标注文件路径（含id和NER Tag列）
        predicted_csv_path: 模型预测文件路径（含id和NER Tag列）
        output_csv_path: 差异分析结果输出路径
    """
    # 读取数据
    orig_df = pd.read_csv(original_csv_path)
    pred_df = pd.read_csv(predicted_csv_path)
    
    # 合并数据
    merged_df = pd.merge(
        orig_df.rename(columns={"NER Tag": "Original Tag"}),
        pred_df.rename(columns={"NER Tag": "Predicted Tag"}),
        on="id",
        how="inner"
    )
    
    # 转换为列表格式（假设原始数据是用字符串表示的列表）
    merged_df["Original Tag"] = merged_df["Original Tag"].apply(eval)
    merged_df["Predicted Tag"] = merged_df["Predicted Tag"].apply(eval)
    
    # 对比分析
    results = []
    for _, row in merged_df.iterrows():
        orig_tags = row["Original Tag"]
        pred_tags = row["Predicted Tag"]
        
        # 检查长度一致性
        if len(orig_tags) != len(pred_tags):
            results.append({
                "id": row["id"],
                "error_type": "LENGTH_MISMATCH",
                "details": f"Original length {len(orig_tags)} != Predicted length {len(pred_tags)}",
                "original": str(orig_tags),
                "predicted": str(pred_tags)
            })
            continue
        
        # 逐token对比
        diff_positions = []
        for i, (o_tag, p_tag) in enumerate(zip(orig_tags, pred_tags)):
            if o_tag != p_tag:
                diff_positions.append({
                    "position": i,
                    "original": o_tag,
                    "predicted": p_tag,
                    "error_type": "TAG_MISMATCH"
                })
        
        # 记录差异
        if diff_positions:
            results.append({
                "id": row["id"],
                "error_type": "TAG_DIFFERENCES",
                "details": f"{len(diff_positions)} tag differences",
                "diff_positions": str(diff_positions),
                "original": str(orig_tags),
                "predicted": str(pred_tags)
            })
    
    # 转换为DataFrame并保存
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_csv_path, index=False)
    print(f"对比结果已保存到: {output_csv_path}")
    
    # 生成统计报告
    generate_statistics(result_df)
    
    return result_df

def generate_statistics(result_df):
    """生成差异统计报告"""
    stats = {
        "total_samples": len(result_df),
        "perfect_matches": len(result_df[result_df["error_type"] == "MATCH"]),
        "length_mismatches": len(result_df[result_df["error_type"] == "LENGTH_MISMATCH"]),
        "tag_differences": len(result_df[result_df["error_type"] == "TAG_DIFFERENCES"]),
    }
    
    stats["match_rate"] = stats["perfect_matches"] / stats["total_samples"]
    
    print("\n===== 差异统计报告 =====")
    for k, v in stats.items():
        print(f"{k.replace('_', ' ').title()}: {v}")
    print("="*30)

# 使用示例
if __name__ == "__main__":
    original_csv = "train.csv"  # 替换为你的原始标注文件路径
    predicted_csv = "submission.csv"  # 替换为模型预测文件路径
    output_csv = "comparison_results.csv"  # 输出文件路径
    
    comparison_results = compare_ner_tags(original_csv, predicted_csv, output_csv)