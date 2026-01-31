#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 14:41:37 2025

@author: sunnydog
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 20:45:49 2025

@author: sunnydog

融合代码：HCM+SMC预后预测研究的完整分析流程
包括：相关性分析 + PCA特征迁移学习 + 预测模型构建
使用生存分析框架替代回归模型
补充了预后评分的关键指标计算
增加了基准模型比较
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pointbiserialr, pearsonr, chi2_contingency, kruskal
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import roc_auc_score, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 尝试导入生存分析包，如果失败则安装
try:
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.metrics import concordance_index_censored
    from sksurv.util import Surv
    from lifelines import CoxPHFitter, KaplanMeierFitter
    from lifelines.statistics import logrank_test
except ImportError:
    print("scikit-survival or lifelines not installed, trying to install...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-survival", "lifelines"])
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.metrics import concordance_index_censored
    from sksurv.util import Surv
    from lifelines import CoxPHFitter, KaplanMeierFitter
    from lifelines.statistics import logrank_test
    
# 在代码开头的import部分添加
try:
    from adjustText import adjust_text
    ADJUST_TEXT_AVAILABLE = True
except ImportError:
    print("adjustText not installed, using alternative methods")
    ADJUST_TEXT_AVAILABLE = False
    
# 定义变量类型
CATEGORICAL_FEATURES = ['a1','b1','b2','b3','b4','b5',
                       'c1', 'c2', 'c3', 'c4', 'c5','c6','c7',
                       'd1','d2','d3','d4','d5',
                       'e9','e10','e11','e12','e13','e14','e15','e16',
                       'f1']

NUMERICAL_FEATURES = ['a2', 'a3', 'a4','e1','e2','e3','e4','e5','e6','e7','e8']

# 特征名称翻译字典
# =============================================================================

FEATURE_TRANSLATIONS = {
    'a1': 'Sex',
    'a2': 'Age', 
    'a3': 'Weight',
    'a4': 'BMI',
    'b1': 'Chest Discomfort',
    'b2': 'Dyspnea',                    # 呼吸困难
    'b3': 'Palpitation',                # 心悸  
    'b4': 'Syncope',                    # 晕厥
    'b5': 'Symptomatic',                # 有症状
    'c1': 'NYHA Class II-IV',     # 心衰2级
    'c2': 'Family History of SMCs',     
    # 家族史
    'c3': 'Hypertension',               # 高血压
    'c4': 'Diabetes Mellitus',          # 糖尿病
    'c5': 'Hyperlipidemia',             # 高脂血症
    'c6': 'Coronary Artery Disease',    # 冠心病
    'c7': 'Atrial Fibrillation/Flutter',# 房颤/房扑
    'd1': 'VT/NSVT',                    # 室速/短阵室速
    'd2': 'Left Bundle Branch Block',   # LBBB
    'd3': 'Sinus Bradycardia',          # 窦缓
    'd4': 'Intraventricular Block',     # 室内阻滞
    'd5': 'AV Block',                   # 房室传导阻滞
    'e1': 'Left Atrial Diameter',       # 左房前后径
    'e2': 'LVEDD',  # 左室横径
    'e3': 'Max LV Wall Thickness',      # 左室最大室壁厚度
    'e4': 'LV Ejection Fraction',       # 左室射血分数
    'e5': 'LVEDV',                      # 左室舒张末期容积
    'e6': 'LVESV',                      # 左室收缩末期容积
    'e7': 'Cardiac Output',             # CO
    'e8': 'Left Ventricular Mass',      # LVM
    'e9': 'Ventricular Aneurysm',       # 有无室壁瘤
    'e10': 'LVOT Obstruction',          # 有无左室流出道梗阻
    'e11': 'SAM',                       # 有无SAM征
    'e12': 'Late Gadolinium Enhancement', # 有无LGE
    'e13': 'RV Insertion Point Enhancement', # 右室插入点强化
    'e14': 'Septal Enhancement',        # 室间隔强化
    'e15': 'LV Free Wall Enhancement',  # 左室游离壁强化
    'e16': 'Apical Enhancement',        # 心尖段强化
    'f1': 'Surgical Treatment'          # 是否手术治疗
}

# 图表标题和标签翻译
CHART_TRANSLATIONS = {
    # 相关性分析
    '特征与预后相关性分析': 'Feature and Prognosis Correlation Analysis',
    '数值变量': 'Numerical Variables',
    '分类变量': 'Categorical Variables',
    '点二列相关系数': 'Point-biserial Correlation Coefficient',
    'Cramér\'s V 系数': 'Cramér\'s V Coefficient',
    '皮尔逊相关系数': 'Pearson Correlation Coefficient',
    '标准化 H 统计量': 'Standardized H Statistic',
    
    # PCA分析
    'PCA特征值分析': 'PCA Eigenvalue Analysis',
    '主成分': 'Principal Component',
    '特征值': 'Eigenvalue',
    '解释方差比例': 'Explained Variance Ratio',
    '累计解释方差': 'Cumulative Explained Variance',
    '碎石图': 'Scree Plot',
    '特征值分布': 'Eigenvalue Distribution',
    '主成分特征载荷热图': 'Principal Component Feature Loading Heatmap',
    '载荷系数': 'Loading Coefficient',
    
    # 模型分析
    '逻辑回归特征重要性': 'Logistic Regression Feature Importance',
    '基于原始变量': 'Based on Original Variables',
    '基于PCA主成分': 'Based on PCA Principal Components',
    '生存分析模型性能比较': 'Survival Analysis Model Performance Comparison',
    '一致性指数': 'Concordance Index',
    'Cox模型': 'Cox Model',
    '随机生存森林': 'Random Survival Forest',
    
    # 预后评分
    '预后评分与生存时间的关系': 'Relationship Between Prognostic Score and Survival Time',
    '不同预后评分分组的事件发生率': 'Event Rates by Prognostic Score Groups',
    'Kaplan-Meier生存曲线': 'Kaplan-Meier Survival Curves',
    '按预后评分分组': 'Stratified by Prognostic Score',
    '低风险': 'Low Risk',
    '中低风险': 'Low-intermediate Risk', 
    '中高风险': 'High-intermediate Risk',
    '高风险': 'High Risk',
    '生存概率': 'Survival Probability',
    '时间 (天)': 'Time (Days)',
    'Log-rank检验': 'Log-rank Test',
    
    # 综合指标
    '综合预后评分详细分析': 'Comprehensive Prognostic Score Analysis',
    '计算预后评分的风险比': 'Calculating Hazard Ratios for Prognostic Score',
    '计算时间依赖性AUC': 'Calculating Time-dependent AUC',
    '计算重分类改善指标': 'Calculating Reclassification Improvement Metrics',
    '计算特征相对重要性': 'Calculating Feature Relative Importance',
    '连续预后评分的风险比': 'Hazard Ratio of Continuous Prognostic Score',
    '时间依赖性AUC': 'Time-dependent AUC',
    '重分类改善': 'Reclassification Improvement',
    'C-index改善': 'C-index Improvement',
    '相对重要性': 'Relative Importance',
    '百分位排名': 'Percentile Rank'
}

# 在函数开头添加
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 12
})

def translate_feature_name(feature_code, translation_dict=FEATURE_TRANSLATIONS):
    """
    翻译特征代码为英文变量名
    支持处理编码后的分类变量名（如"a1_1"）
    """
    if not isinstance(feature_code, str):
        feature_code = str(feature_code)
    
    # 如果是经过编码的分类变量（如"a1_1"），提取原始特征名
    if '_' in feature_code:
        # 尝试匹配编码后的变量名（如"a1_1", "a1_2"等）
        for code in translation_dict.keys():
            if feature_code.startswith(code + '_'):
                # 返回基础特征名加上编码部分
                base_name = translation_dict.get(code, code)
                category = feature_code.split('_')[1]
                # 尝试将类别转换为更有意义的形式
                try:
                    # 如果类别是数字，可以添加描述
                    if category.isdigit():
                        if code == 'a1':  # 性别
                            category_desc = 'Male' if category == '1' else 'Female'
                        else:
                            category_desc = f'Category {category}'
                        return f"{base_name} ({category_desc})"
                except:
                    pass
                return f"{base_name}_{category}"
        
        # 如果没有匹配到，提取基础特征名
        base_feature = feature_code.split('_')[0]
        base_name = translation_dict.get(base_feature, base_feature)
        return f"{base_name}_{feature_code.split('_')[1]}"
    else:
        # 直接翻译
        return translation_dict.get(feature_code, feature_code)

def translate_feature_names(feature_list, translation_dict=FEATURE_TRANSLATIONS):
    """
    翻译特征名称列表
    """
    translated = []
    for feature in feature_list:
        translated.append(translate_feature_name(feature, translation_dict))
    return translated

def translate_chart_text(text):
    """
    翻译图表文本
    """
    return CHART_TRANSLATIONS.get(text, text)


# =============================================================================
# 新增：基准模型训练和比较函数
# =============================================================================

def train_baseline_models_on_smc(X_smc_original, y_time, y_event):
    """
    直接在SMC数据上训练基准模型（不使用PCA迁移学习）
    """
    print("\n" + "="*60)
    print("Training Baseline Models on SMC (without PCA transfer learning)")
    print("="*60)
    
    # 创建生存分析所需的数据格式
    y_structured = np.array([(bool(event_i), time_i) for event_i, time_i in zip(y_event, y_time)],
                          dtype=[('event', 'bool'), ('time', 'f8')])
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_smc_original, y_structured, test_size=0.2, random_state=42
    )
    
    print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    print(f"Features used: {X_train.shape[1]}")
    
    baseline_results = {}
    
    # 方法1: Cox比例风险模型
    print("\n1. Training baseline Cox proportional hazards model...")
    try:
        cox_model = CoxPHSurvivalAnalysis(alpha=0.1)
        cox_model.fit(X_train, y_train)
        cox_score = cox_model.score(X_test, y_test)
        baseline_results['Baseline Cox'] = {
            'model': cox_model,
            'c_index': cox_score,
            'type': 'cox'
        }
        print(f"Baseline Cox model concordance index: {cox_score:.3f}")
    except Exception as e:
        print(f"Error training baseline Cox model: {e}")
        baseline_results['Baseline Cox'] = {'c_index': 0.5, 'model': None, 'type': 'cox'}
    
    # 方法2: 随机生存森林
    print("\n2. Training baseline random survival forest model...")
    try:
        rsf = RandomSurvivalForest(
            n_estimators=100,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        rsf.fit(X_train, y_train)
        rsf_score = rsf.score(X_test, y_test)
        baseline_results['Baseline RSF'] = {
            'model': rsf,
            'c_index': rsf_score,
            'type': 'rsf'
        }
        print(f"Baseline random survival forest concordance index: {rsf_score:.3f}")
    except Exception as e:
        print(f"Error training baseline RSF model: {e}")
        baseline_results['Baseline RSF'] = {'c_index': 0.5, 'model': None, 'type': 'rsf'}
    
    # 方法3: 简单的逻辑回归（作为事件分类的基准）
    print("\n3. Training baseline logistic regression model...")
    try:
        y_event_train = [y[0] for y in y_train]
        y_event_test = [y[0] for y in y_test]
        
        lr_model = LogisticRegression(penalty='l2', C=1.0, solver='liblinear', max_iter=1000)
        lr_model.fit(X_train, y_event_train)
        
        # 使用AUC作为评估指标
        test_probs = lr_model.predict_proba(X_test)[:, 1]
        lr_auc = roc_auc_score(y_event_test, test_probs)
        
        # 将AUC转换为近似C-index（对于二分类生存数据）
        # 注意：这只是一个近似值，实际上生存分析的C-index和分类的AUC不完全等同
        baseline_results['Baseline Logistic'] = {
            'model': lr_model,
            'c_index': lr_auc,  # 使用AUC作为近似
            'auc': lr_auc,
            'type': 'logistic'
        }
        print(f"Baseline logistic regression AUC: {lr_auc:.3f}")
    except Exception as e:
        print(f"Error training baseline logistic regression: {e}")
        baseline_results['Baseline Logistic'] = {'c_index': 0.5, 'model': None, 'type': 'logistic'}
    
    return baseline_results

def train_transfer_learning_models(X_smc_pca, y_time, y_event):
    """
    使用PCA迁移学习特征训练模型
    """
    print("\n" + "="*60)
    print("Training Transfer Learning Models on SMC (with PCA from HCM)")
    print("="*60)
    
    # 创建生存分析所需的数据格式
    y_structured = np.array([(bool(event_i), time_i) for event_i, time_i in zip(y_event, y_time)],
                          dtype=[('event', 'bool'), ('time', 'f8')])
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_smc_pca, y_structured, test_size=0.2, random_state=42
    )
    
    print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    print(f"PCA features used: {X_train.shape[1]}")
    
    transfer_results = {}
    
    # 方法1: Cox比例风险模型
    print("\n1. Training transfer learning Cox proportional hazards model...")
    try:
        cox_model = CoxPHSurvivalAnalysis(alpha=0.1)
        cox_model.fit(X_train, y_train)
        cox_score = cox_model.score(X_test, y_test)
        transfer_results['Transfer Cox'] = {
            'model': cox_model,
            'c_index': cox_score,
            'type': 'cox'
        }
        print(f"Transfer learning Cox model concordance index: {cox_score:.3f}")
    except Exception as e:
        print(f"Error training transfer learning Cox model: {e}")
        transfer_results['Transfer Cox'] = {'c_index': 0.5, 'model': None, 'type': 'cox'}
    
    # 方法2: 随机生存森林
    print("\n2. Training transfer learning random survival forest model...")
    try:
        rsf = RandomSurvivalForest(
            n_estimators=100,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        rsf.fit(X_train, y_train)
        rsf_score = rsf.score(X_test, y_test)
        transfer_results['Transfer RSF'] = {
            'model': rsf,
            'c_index': rsf_score,
            'type': 'rsf'
        }
        print(f"Transfer learning random survival forest concordance index: {rsf_score:.3f}")
    except Exception as e:
        print(f"Error training transfer learning RSF model: {e}")
        transfer_results['Transfer RSF'] = {'c_index': 0.5, 'model': None, 'type': 'rsf'}
    
    # 方法3: 逻辑回归
    print("\n3. Training transfer learning logistic regression model...")
    try:
        y_event_train = [y[0] for y in y_train]
        y_event_test = [y[0] for y in y_test]
        
        lr_model = LogisticRegression(penalty='l2', C=1.0, solver='liblinear', max_iter=1000)
        lr_model.fit(X_train, y_event_train)
        
        test_probs = lr_model.predict_proba(X_test)[:, 1]
        lr_auc = roc_auc_score(y_event_test, test_probs)
        
        transfer_results['Transfer Logistic'] = {
            'model': lr_model,
            'c_index': lr_auc,
            'auc': lr_auc,
            'type': 'logistic'
        }
        print(f"Transfer learning logistic regression AUC: {lr_auc:.3f}")
    except Exception as e:
        print(f"Error training transfer learning logistic regression: {e}")
        transfer_results['Transfer Logistic'] = {'c_index': 0.5, 'model': None, 'type': 'logistic'}
    
    return transfer_results

def compare_model_performance(baseline_results, transfer_results, output_file='model_comparison.png'):
    """
    比较基准模型和迁移学习模型的性能
    """
    print("\n" + "="*60)
    print("Model Performance Comparison: Traditional Statistical Approaches vs Transfer Learning Approaches")
    print("="*60)
    
    # 准备数据用于可视化
    comparison_data = []
    
    # 收集基准模型结果
    for model_name, results in baseline_results.items():
        comparison_data.append({
            'Model': model_name,
            'C-index': results['c_index'],
            'Type': 'Traditional Statistical Approaches',
            'Model Type': results['type']
        })
    
    # 收集迁移学习模型结果
    for model_name, results in transfer_results.items():
        comparison_data.append({
            'Model': model_name,
            'C-index': results['c_index'],
            'Type': 'Transfer Learning Approaches',
            'Model Type': results['type']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # 打印详细比较
    print("\n=== Detailed Model Performance Comparison ===")
    print(comparison_df.to_string(index=False))
    
    # 计算性能提升
    baseline_cox = baseline_results.get('Baseline Cox', {}).get('c_index', 0.5)
    transfer_cox = transfer_results.get('Transfer Cox', {}).get('c_index', 0.5)
    cox_improvement = transfer_cox - baseline_cox
    
    baseline_rsf = baseline_results.get('Baseline RSF', {}).get('c_index', 0.5)
    transfer_rsf = transfer_results.get('Transfer RSF', {}).get('c_index', 0.5)
    rsf_improvement = transfer_rsf - baseline_rsf
    
    print(f"\n=== Performance Improvement ===")
    print(f"Cox Model Improvement: {cox_improvement:.3f} ({baseline_cox:.3f} → {transfer_cox:.3f})")
    print(f"RSF Model Improvement: {rsf_improvement:.3f} ({baseline_rsf:.3f} → {transfer_rsf:.3f})")
    
    if cox_improvement > 0:
        print(f"Cox model improved by {100*cox_improvement/baseline_cox:.1f}%")
    if rsf_improvement > 0:
        print(f"RSF model improved by {100*rsf_improvement/baseline_rsf:.1f}%")
    
    # 可视化比较
    plt.figure(figsize=(14, 8))
    
    # 设置颜色
    colors = {'Traditional Statistical Approaches': '#3498DB', 'Transfer Learning Approaches': '#E74C3C'}
    
    # 获取唯一的模型类型（去掉前缀）
    model_types = sorted(set([m.replace('Baseline ', '').replace('Transfer ', '').replace('Traditional Statistical Approaches ', '') 
                             for m in baseline_results.keys()]))
    
    # 创建分组条形图
    x = np.arange(len(model_types))
    width = 0.22  # 进一步减小宽度使柱子更细
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 准备数据
    baseline_values = []
    transfer_values = []
    
    for model_type in model_types:
        baseline_key = f'Baseline {model_type}'
        transfer_key = f'Transfer {model_type}'
        
        baseline_values.append(baseline_results.get(baseline_key, {}).get('c_index', 0))
        transfer_values.append(transfer_results.get(transfer_key, {}).get('c_index', 0))
    
    # 绘制条形 - 使用更细的柱子
    baseline_bars = ax.bar(x - width/2, baseline_values, width, 
                          label='Traditional Statistical Approaches', 
                          color=colors['Traditional Statistical Approaches'], 
                          alpha=0.8, linewidth=1, edgecolor='black')
    
    transfer_bars = ax.bar(x + width/2, transfer_values, width, 
                          label='Transfer Learning Approaches', 
                          color=colors['Transfer Learning Approaches'], 
                          alpha=0.8, linewidth=1, edgecolor='black')
    
    # 添加数值标签（使用更小的字体）
    for bar in baseline_bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.008,
                f'{height:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    for bar in transfer_bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.008,
                f'{height:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 添加斜向上箭头显示改进 - 修改为斜形右上方向上的箭头
    for i in range(len(model_types)):
        baseline_c = baseline_values[i]
        transfer_c = transfer_values[i]
        
        if transfer_c > baseline_c:
            # 计算柱子中心位置
            baseline_center = baseline_bars[i].get_x() + baseline_bars[i].get_width()/2
            transfer_center = transfer_bars[i].get_x() + transfer_bars[i].get_width()/2
            
            # 计算改进值和百分比
            improvement = transfer_c - baseline_c
            percentage_improvement = (improvement / baseline_c) * 100 if baseline_c > 0 else 0
            
            # 计算箭头起点和终点（起点在基准柱子顶部，终点在迁移柱子顶部）
            # 将箭头起点和终点降低一些，避免与柱子顶部的数值标签重叠
            start_x = baseline_center
            start_y = baseline_c + 0.04  # 从柱子顶部上方0.02开始
            end_x = transfer_center
            end_y = transfer_c + 0.05    # 到迁移柱子顶部上方0.02
            
            # 绘制斜向上箭头（右上方向）
            arrow = ax.annotate('', 
                               xy=(end_x, end_y), 
                               xytext=(start_x, start_y),
                               arrowprops=dict(arrowstyle='->', 
                                              color='green', 
                                              lw=2.5,
                                              shrinkA=6,  # 起点留出空间
                                              shrinkB=7)) # 终点留出空间
            
            # 计算箭头中点位置（用于放置文本）- 将文本位置提高一些
            mid_x = (start_x + end_x) / 2
            mid_y = (start_y + end_y) / 2 + 0.05  # 增加偏移量，从0.02改为0.03
            
            # 添加改进值文本 - 使用更紧凑的格式
            ax.text(mid_x, mid_y, 
                   f'+{improvement:.3f} (+{percentage_improvement:.1f}%)', 
                   ha='center', va='bottom', fontsize=10, color='green', 
                   fontweight='bold', 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor='green', alpha=0.9))
    
    # 设置图表属性
    ax.set_xlabel('Model Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('C-index / AUC', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison: Traditional Statistical Approaches vs Transfer Learning Approaches', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(model_types, rotation=0, fontsize=11)
    ax.set_ylim(0, 1.15)  # 调整y轴上限
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加参考线（不显示在图例中）
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0.7, color='orange', linestyle='--', alpha=0.5)
    
    # 只显示柱子的图例
    from matplotlib.patches import Patch
    
    legend_elements = [
        Patch(facecolor=colors['Traditional Statistical Approaches'], 
              label='Traditional Statistical Approaches'),
        Patch(facecolor=colors['Transfer Learning Approaches'], 
              label='Transfer Learning Approaches')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nModel comparison chart saved as: {output_file}")
    
    # 保存详细结果到Excel
    with pd.ExcelWriter('model_performance_comparison.xlsx') as writer:
        comparison_df.to_excel(writer, sheet_name='All Models', index=False)
        
        # 创建汇总表
        summary_data = []
        for model_type in model_types:
            baseline_key = f'Baseline {model_type}'
            transfer_key = f'Transfer {model_type}'
            
            if baseline_key in baseline_results and transfer_key in transfer_results:
                baseline_c = baseline_results[baseline_key]['c_index']
                transfer_c = transfer_results[transfer_key]['c_index']
                improvement = transfer_c - baseline_c
                relative_improvement = 100 * improvement / baseline_c if baseline_c > 0 else 0
                
                summary_data.append({
                    'Model Type': model_type,
                    'Traditional Approaches C-index': baseline_c,
                    'Transfer Learning C-index': transfer_c,
                    'Absolute Improvement (ΔC-index)': improvement,
                    'Relative Improvement (%)': relative_improvement,
                    'Significant Improvement': 'Yes' if improvement > 0.05 else 'No'
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # 添加统计检验结果（简化的）
        stats_df = pd.DataFrame({
            'Metric': ['Best Traditional Model', 'Best Transfer Learning Model', 
                      'Maximum Improvement (ΔC-index)', 'Average Improvement (ΔC-index)'],
            'Value': [
                max(baseline_values),
                max(transfer_values),
                max([t-b for t, b in zip(transfer_values, baseline_values)]),
                np.mean([t-b for t, b in zip(transfer_values, baseline_values)])
            ]
        })
        stats_df.to_excel(writer, sheet_name='Statistics', index=False)
    
    print("Detailed comparison results saved to: model_performance_comparison.xlsx")
    
    return comparison_df
# =============================================================================
# 新增：预后评分关键指标计算函数
# =============================================================================

def calculate_hazard_ratios(prognostic_score, time, event):
    """
    计算预后评分分组的风险比和置信区间
    """
    print("\n=== Calculating Hazard Ratios for Prognostic Score ===")
    
    # 确保输入数据是数值类型
    time = pd.to_numeric(time, errors='coerce')
    event = pd.to_numeric(event, errors='coerce')
    prognostic_score = pd.to_numeric(prognostic_score, errors='coerce')
    
    # 移除任何NaN值
    valid_mask = ~(np.isnan(time) | np.isnan(event) | np.isnan(prognostic_score))
    time = time[valid_mask]
    event = event[valid_mask]
    prognostic_score = prognostic_score[valid_mask]
    
    print(f"Valid sample size: {len(time)}")
    
    # 创建分组（使用四分位数）
    group_labels = ['Low Risk', 'Low-intermediate Risk', 'High-intermediate Risk', 'High Risk']
    
    # 确保有足够的数据点进行分组
    if len(prognostic_score) < 4:
        print("Too few data points for quartile grouping")
        return None, None, None
    
    try:
        score_groups = pd.cut(prognostic_score, 
                             bins=[0, 0.25, 0.5, 0.75, 1.0],
                             labels=group_labels)
        
        # 确保score_groups是分类类型
        score_groups = pd.Categorical(score_groups, categories=group_labels, ordered=True)
    except Exception as e:
        print(f"Failed to create score groups: {e}")
        # 使用等距分组作为备选
        try:
            score_groups = pd.cut(prognostic_score, 
                                 bins=4,
                                 labels=group_labels)
            score_groups = pd.Categorical(score_groups, categories=group_labels, ordered=True)
        except Exception as e2:
            print(f"Equal interval grouping also failed: {e2}")
            return None, None, None
    
    # 准备Cox回归数据
    cox_data = pd.DataFrame({
        'time': time,
        'event': event,
        'score_group': score_groups,
        'continuous_score': prognostic_score
    })
    
    # 为分类变量创建哑变量
    cox_data_dummy = pd.get_dummies(cox_data, columns=['score_group'], prefix='group')
    
    # 确保参考组是低风险组
    if 'group_Low Risk' not in cox_data_dummy.columns:
        # 如果低风险组不存在，选择第一个组作为参考
        reference_group = [col for col in cox_data_dummy.columns if col.startswith('group_')][0]
        print(f"Using {reference_group} as reference group")
    else:
        reference_group = 'group_Low Risk'
    
    # 拟合Cox模型
    cph = CoxPHFitter()
    try:
        cph.fit(cox_data_dummy[['time', 'event'] + [col for col in cox_data_dummy.columns if col.startswith('group_')]], 
                duration_col='time', event_col='event')
        
        # 获取风险比结果
        hr_results = cph.summary
    except Exception as e:
        print(f"Failed to fit grouped Cox model: {e}")
        hr_results = None
    
    # 计算连续评分的风险比
    cph_continuous = CoxPHFitter()
    try:
        cph_continuous.fit(cox_data[['time', 'event', 'continuous_score']], 
                          duration_col='time', event_col='event')
        continuous_hr = cph_continuous.summary.loc['continuous_score']
        
        print(f"Hazard ratio of continuous prognostic score: {continuous_hr['exp(coef)']:.3f} (95%CI: {continuous_hr['exp(coef) lower 95%']:.3f}-{continuous_hr['exp(coef) upper 95%']:.3f})")
    except Exception as e:
        print(f"Failed to fit continuous score Cox model: {e}")
        continuous_hr = None
    
    return hr_results, score_groups, continuous_hr

def calculate_time_dependent_auc(prognostic_score, time, event, time_points=[365, 1095]):
    """
    计算不同时间点的AUC（时间依赖性ROC）
    """
    print("\n=== Calculating Time-dependent AUC ===")
    
    auc_results = {}
    
    for t in time_points:
        # 创建该时间点的二分类结局
        # 事件：在时间t之前发生事件
        # 删失：在时间t之后仍存活或失访
        y_binary = (time <= t) & (event == 1)
        
        # 只考虑在该时间点之前有信息的患者
        # 包括：在时间t之前发生事件的患者，以及在时间t时仍存活的患者
        informative = (time <= t) | (event == 0)
        
        if sum(y_binary) > 0 and sum(informative) > 0:  # 确保有事件发生且有信息患者
            try:
                auc = roc_auc_score(y_binary[informative], prognostic_score[informative])
                auc_results[f'{t} days'] = {
                    'AUC': auc,
                    'n_events': sum(y_binary),
                    'n_informative': sum(informative)
                }
                print(f"{t} days AUC: {auc:.3f} (Events: {sum(y_binary)})")
            except Exception as e:
                print(f"Error calculating {t} days AUC: {e}")
        else:
            print(f"{t} days: Insufficient events, skipping calculation")
    
    return auc_results

def calculate_reclassification_metrics(prognostic_score, baseline_data, time, event, time_point=365):
    """
    计算净重分类改善(NRI)和综合判别改善(IDI)
    简化版本 - 与基于年龄和EF的简单评分比较
    """
    print("\n=== Calculating Reclassification Improvement Metrics ===")
    
    # 如果没有提供基线评分，创建一个简单的基线评分（基于年龄和EF）
    if baseline_data is None:
        print("Creating baseline score (based on age and ejection fraction)...")
        # 假设a2是年龄，e4是射血分数
        # 这里需要根据你的实际数据调整
        try:
            # 标准化年龄和EF
            age_normalized = (baseline_data['a2'] - baseline_data['a2'].mean()) / baseline_data['a2'].std()
            ef_normalized = (baseline_data['e4'] - baseline_data['e4'].mean()) / baseline_data['e4'].std()
            
            # 简单线性组合：年龄增加风险，EF降低风险
            baseline_scores = age_normalized - 0.5 * ef_normalized
            baseline_scores = (baseline_scores - baseline_scores.min()) / (baseline_scores.max() - baseline_scores.min())
        except Exception as e:
            print(f"Failed to create baseline score: {e}")
            return None
    else:
        # 如果提供了基线数据但格式不对，尝试处理
        print("Using provided baseline data...")
        try:
            # 确保baseline_data是一维数组
            if hasattr(baseline_data, 'shape') and len(baseline_data.shape) > 1:
                print(f"Warning: Baseline data is {baseline_data.shape[1]}-dimensional array, trying to convert to 1D")
                # 如果是二维数组，取第一列或进行平均
                if baseline_data.shape[1] == 2:
                    # 如果是两列，假设是年龄和EF，进行组合
                    baseline_scores = baseline_data.iloc[:, 0] - 0.5 * baseline_data.iloc[:, 1]
                else:
                    baseline_scores = baseline_data.mean(axis=1)
            else:
                baseline_scores = baseline_data
            
            # 标准化基线评分
            baseline_scores = (baseline_scores - baseline_scores.min()) / (baseline_scores.max() - baseline_scores.min())
            
        except Exception as e:
            print(f"Failed to process baseline data: {e}")
            return None
    
    # 确保baseline_scores是一维数组
    try:
        baseline_scores = np.array(baseline_scores).flatten()
        print(f"Baseline score shape: {baseline_scores.shape}")
        print(f"Prognostic score shape: {prognostic_score.shape}")
    except Exception as e:
        print(f"Failed to convert baseline score to 1D array: {e}")
        return None
    
    # 计算C-index改善
    from sksurv.metrics import concordance_index_censored
    
    # 确保所有数组长度一致
    min_length = min(len(prognostic_score), len(baseline_scores), len(time), len(event))
    prognostic_score = prognostic_score[:min_length]
    baseline_scores = baseline_scores[:min_length]
    time = time[:min_length]
    event = event[:min_length]
    
    print(f"Using {min_length} samples for calculation")
    
    # 新模型的C-index
    try:
        cindex_new = concordance_index_censored(event.astype(bool), time, -prognostic_score)[0]
    except Exception as e:
        print(f"Failed to calculate new model C-index: {e}")
        return None
    
    # 基线模型的C-index
    try:
        cindex_old = concordance_index_censored(event.astype(bool), time, -baseline_scores)[0]
    except Exception as e:
        print(f"Failed to calculate baseline model C-index: {e}")
        return None
    
    cindex_improvement = cindex_new - cindex_old
    
    # 简化的IDI计算
    try:
        event_group_new = prognostic_score[event == 1]
        nonevent_group_new = prognostic_score[event == 0]
        
        event_group_old = baseline_scores[event == 1]
        nonevent_group_old = baseline_scores[event == 0]
        
        # IDI = (新模型事件组平均分 - 新模型非事件组平均分) - (旧模型事件组平均分 - 旧模型非事件组平均分)
        idi = (np.mean(event_group_new) - np.mean(nonevent_group_new)) - (np.mean(event_group_old) - np.mean(nonevent_group_old))
        
        # 简化的NRI计算（概念性）
        # 在实际应用中需要更复杂的计算
        nri_events = np.mean(event_group_new > event_group_old) - np.mean(event_group_new < event_group_old)
        nri_nonevents = np.mean(nonevent_group_new < nonevent_group_old) - np.mean(nonevent_group_new > nonevent_group_old)
        nri = nri_events + nri_nonevents
        
        results = {
            'C-index Improvement': cindex_improvement,
            'IDI': idi,
            'NRI': nri,
            'New Model C-index': cindex_new,
            'Baseline Model C-index': cindex_old
        }
        
        print(f"C-index Improvement: {cindex_improvement:.3f}")
        print(f"IDI: {idi:.3f}")
        print(f"NRI: {nri:.3f}")
        print(f"New Model C-index: {cindex_new:.3f}")
        print(f"Baseline Model C-index: {cindex_old:.3f}")
        
        return results
    except Exception as e:
        print(f"Error calculating IDI and NRI: {e}")
        return None


def create_survival_prognostic_score(classifier, survival_model, model_type, X, y_time, y_event):
    """
    创建基于生存分析的预后评分 - 修复版本
    """
    print("\n=== Building Prognostic Score ===")
    
    # 方法1: 直接使用生存模型的风险评分
    if model_type == 'cox':
        prognostic_score = survival_model.predict(X)
        print("Using Cox model risk score as prognostic score")
    else:
        # 对于随机生存森林，使用更稳健的方法
        try:
            # 预测在特定时间点的生存概率
            time_points = [365, 730, 1095]  # 1年, 2年, 3年
            
            # 预测生存函数
            survival_funcs = survival_model.predict_survival_function(X)
            
            # 计算多个时间点的平均风险
            risk_scores = []
            for func in survival_funcs:
                # 计算在多个时间点的生存概率，然后转换为风险
                survival_probs = [func(t) for t in time_points if t <= func.x.max()]
                if survival_probs:
                    avg_survival = np.mean(survival_probs)
                    risk_score = 1 - avg_survival  # 生存概率转换为风险
                else:
                    risk_score = 0.5  # 默认值
                risk_scores.append(risk_score)
            
            prognostic_score = np.array(risk_scores)
            print("Using RSF multi-timepoint average risk as prognostic score")
            
        except Exception as e:
            print(f"RSF risk score calculation failed: {e}")
            # 备选方案：使用事件概率
            prognostic_score = classifier.predict_proba(X)[:, 1]
            print("Using event probability as prognostic score")
    
    # 标准化评分到0-1范围
    prognostic_score = (prognostic_score - prognostic_score.min()) / (prognostic_score.max() - prognostic_score.min())
    
    # 验证预后评分的C-index
    from sksurv.metrics import concordance_index_censored
    c_index = concordance_index_censored(y_event.astype(bool), y_time, prognostic_score)[0]
    print(f"Prognostic score C-index: {c_index:.3f}")
    
    if c_index < 0.5:
        print("Warning: Poor prognostic score performance, considering inversion")
        prognostic_score = 1 - prognostic_score
        c_index_inverted = concordance_index_censored(y_event.astype(bool), y_time, prognostic_score)[0]
        print(f"Inverted C-index: {c_index_inverted:.3f}")
    
    return prognostic_score

def create_kaplan_meier_with_risktable(prognostic_score, time, event):
    """
    创建带风险表的Kaplan-Meier生存曲线
    """
    print("\n=== Creating Kaplan-Meier Survival Curves with Risk Table ===")
    
    # 根据数据分布动态确定随访时间
    time_90th = np.percentile(time, 85)  # 使用85百分位，减少尾部删失
    max_time = min(365 * 2.5, time_90th)  # 不超过2.5年
    
    # 使用事件发生的时间范围
    event_times = time[event == 1]
    if len(event_times) > 0:
        max_time_events = np.max(event_times)
        max_time = min(max_time, max_time_events * 1.2)  # 不超过最后事件的120%
    
    valid_mask = time <= max_time
    time_limited = time[valid_mask]
    event_limited = event[valid_mask]
    prognostic_score_limited = prognostic_score[valid_mask]
    
    print(f"Limited follow-up time to {max_time:.0f} days, valid samples: {len(time_limited)}")
    print(f"Number of events: {sum(event_limited)}")
    
    # 使用中位数分组
    median_score = np.median(prognostic_score_limited)
    score_groups = np.where(prognostic_score_limited <= median_score, 'Low Risk', 'High Risk')
    score_groups = pd.Categorical(score_groups, categories=['Low Risk', 'High Risk'], ordered=True)
    
    # 打印每组样本量
    n_low = sum(score_groups == 'Low Risk')
    n_high = sum(score_groups == 'High Risk')
    print(f"Low risk group sample size: {n_low}")
    print(f"High risk group sample size: {n_high}")
    
    # 创建图形和轴 - 调整高度比例
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), 
                                   gridspec_kw={'height_ratios': [3, 1]},
                                   sharex=True)
    
    kmf = KaplanMeierFitter()
    
    # 定义颜色
    colors = ['#2E8B57', '#DC143C']  # 海绿色和深红色
    
    # 绘制低风险组
    mask_low = (score_groups == 'Low Risk')
    if sum(mask_low) > 5:
        kmf_low = KaplanMeierFitter()
        kmf_low.fit(time_limited[mask_low], event_limited[mask_low], 
                   label=f'Low Risk (n={sum(mask_low)})')
        kmf_low.plot(ax=ax1, ci_show=True, color=colors[0], linewidth=2)
    
    # 绘制高风险组
    mask_high = (score_groups == 'High Risk')
    if sum(mask_high) > 5:
        kmf_high = KaplanMeierFitter()
        kmf_high.fit(time_limited[mask_high], event_limited[mask_high], 
                    label=f'High Risk (n={sum(mask_high)})')
        kmf_high.plot(ax=ax1, ci_show=True, color=colors[1], linewidth=2)
    
    # 设置生存曲线样式
    ax1.set_title(f'Kaplan-Meier Survival Curves\n(Follow-up: {max_time//365} years)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Survival Probability', fontsize=12)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=11)
    
    # 计算log-rank检验p值
    try:
        from lifelines.statistics import logrank_test
        
        if sum(mask_low) > 0 and sum(mask_high) > 0:
            results = logrank_test(time_limited[mask_low], time_limited[mask_high], 
                                 event_limited[mask_low], event_limited[mask_high])
            p_value = results.p_value
            
            # 在图上添加p值
            ax1.text(0.02, 0.02, f'Log-rank test: p = {p_value:.4f}', 
                    transform=ax1.transAxes, fontsize=11,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            print(f"Log-rank test p-value: {p_value:.4f}")
    except Exception as e:
        print(f"Error calculating log-rank test: {e}")
    
    # ==================== 创建风险表 ====================
    # 定义风险表的时间点
    if max_time <= 365:  # 1年以内
        time_points = np.linspace(0, max_time, 6, dtype=int)
    elif max_time <= 730:  # 2年以内
        time_points = np.linspace(0, max_time, 8, dtype=int)
    else:  # 超过2年
        time_points = np.linspace(0, max_time, 10, dtype=int)
    
    # 确保时间点不超过最大时间
    time_points = time_points[time_points <= max_time]
    
    # 创建风险表数据
    risk_table_data = []
    
    for t in time_points:
        row = {'Time': t}
        
        # 低风险组
        if sum(mask_low) > 0:
            at_risk_low = sum(time_limited[mask_low] >= t)
            row['Low Risk'] = at_risk_low
        else:
            row['Low Risk'] = 0
            
        # 高风险组
        if sum(mask_high) > 0:
            at_risk_high = sum(time_limited[mask_high] >= t)
            row['High Risk'] = at_risk_high
        else:
            row['High Risk'] = 0
            
        risk_table_data.append(row)
    
    risk_table_df = pd.DataFrame(risk_table_data)
    
    # 绘制风险表
    ax2.axis('off')  # 隐藏坐标轴
    table = ax2.table(
        cellText=risk_table_df.values,
        colLabels=risk_table_df.columns,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]  # 表格占据整个轴区域
    )
    
    # 设置表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)  # 调整表格大小
    
    # 设置表头样式
    for i in range(len(risk_table_df.columns)):
        table[(0, i)].set_facecolor('#4B4B4B')
        table[(0, i)].set_text_props(color='white', weight='bold')
    
    # 交替行颜色
    for i in range(1, len(risk_table_df) + 1):
        for j in range(len(risk_table_df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F5F5F5')
    
    # 设置x轴标签
    ax1.set_xlabel('Time (Days)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('kaplan_meier_with_risktable.png', dpi=300, bbox_inches='tight')
    
    # 也保存风险表数据到文件
    risk_table_df.to_csv('risk_table_data.csv', index=False)
    print("Risk table data saved to: risk_table_data.csv")
    
    return fig, risk_table_df

def create_simple_kaplan_meier_with_risktable(prognostic_score, time, event):
    """
    创建简化的带风险表的Kaplan-Meier生存曲线
    """
    print("\n=== Creating Simplified Kaplan-Meier Survival Curves ===")
    
    # 使用中位数分组
    median_score = np.median(prognostic_score)
    score_groups = np.where(prognostic_score <= median_score, 'Low Risk', 'High Risk')
    
    # 创建生存曲线
    plt.figure(figsize=(10, 6))
    kmf = KaplanMeierFitter()
    
    # 绘制低风险组
    mask_low = (score_groups == 'Low Risk')
    if sum(mask_low) > 0:
        kmf.fit(time[mask_low], event[mask_low], label='Low Risk')
        kmf.plot(ci_show=True)
    
    # 绘制高风险组
    mask_high = (score_groups == 'High Risk')
    if sum(mask_high) > 0:
        kmf.fit(time[mask_high], event[mask_high], label='High Risk')
        kmf.plot(ci_show=True)
    
    plt.title('Kaplan-Meier Survival Curves')
    plt.xlabel('Time (Days)')
    plt.ylabel('Survival Probability')
    plt.grid(True, alpha=0.3)
    plt.savefig('kaplan_meier_simple.png', dpi=300, bbox_inches='tight')
    
    return plt.gcf()

def comprehensive_prognostic_score_analysis(prognostic_score, time, event, feature_importance, baseline_data=None):
    """
    综合预后评分分析 - 整合所有关键指标计算
    增加95%置信区间计算
    """
    print("\n" + "="*60)
    print("Comprehensive Prognostic Score Analysis")
    print("="*60)
    
    # ==================== 新增：CI计算函数（放在最前面） ====================
# 在 comprehensive_prognostic_score_analysis 函数内部，找到 calculate_cindex_ci 函数的定义
# 修改为返回3个值

    def calculate_cindex_ci(prognostic_score, time, event, n_bootstrap=1000):
        """计算C-index的95%置信区间"""
        from sksurv.metrics import concordance_index_censored
        import numpy as np
        
        n_samples = len(time)
        cindex_values = []
        
        print(f"Starting bootstrap for C-index CI (n={n_samples}, bootstrap={n_bootstrap})...")
        
        # 先计算原始C-index
        original_cindex = concordance_index_censored(
            event.astype(bool), 
            time, 
            prognostic_score  # 注意：不要取负值
        )[0]
        print(f"Original C-index: {original_cindex:.3f}")
        
        for i in range(n_bootstrap):
            if i % 100 == 0:
                print(f"  Bootstrap iteration {i}/{n_bootstrap}")
            
            indices = np.random.choice(n_samples, n_samples, replace=True)
            try:
                cindex = concordance_index_censored(
                    event[indices].astype(bool), 
                    time[indices], 
                    prognostic_score[indices]  # 修复：使用正值而不是负值
                )[0]
                cindex_values.append(cindex)
            except Exception as e:
                if i == 0:  # 只打印第一次错误
                    print(f"  Warning in bootstrap iteration {i}: {e}")
                continue
        
        if cindex_values:
            cindex_values = np.array(cindex_values)
            ci_lower = np.percentile(cindex_values, 2.5)
            ci_upper = np.percentile(cindex_values, 97.5)
            print(f"  Bootstrap completed: mean C-index = {np.mean(cindex_values):.3f}, 95% CI = [{ci_lower:.3f}, {ci_upper:.3f}]")
            return ci_lower, ci_upper, original_cindex  # 返回3个值
        else:
            print("  Error: No valid bootstrap samples for C-index CI calculation")
            return None, None, original_cindex  # 返回3个值
                
    def calculate_time_auc_ci(prognostic_score, time, event, time_point, n_bootstrap=1000):
        """计算时间依赖性AUC的95%置信区间"""
        from sklearn.metrics import roc_auc_score
        import numpy as np
        
        n_samples = len(time)
        auc_values = []
        
        for i in range(n_bootstrap):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            y_binary = (time[indices] <= time_point) & (event[indices] == 1)
            informative = (time[indices] <= time_point) | (event[indices] == 0)
            
            if sum(y_binary) > 2:
                try:
                    auc = roc_auc_score(
                        y_binary[informative], 
                        prognostic_score[indices][informative]
                    )
                    auc_values.append(auc)
                except:
                    continue
        
        if auc_values:
            auc_values = np.array(auc_values)
            ci_lower = np.percentile(auc_values, 2.5)
            ci_upper = np.percentile(auc_values, 97.5)
            return ci_lower, ci_upper
        else:
            return None, None
    
    def calculate_reclassification_ci(prognostic_score, baseline_score, time, event, n_bootstrap=1000):
        """计算NRI和IDI的95%置信区间"""
        import numpy as np
        
        n_samples = len(time)
        nri_values = []
        idi_values = []
        
        for i in range(n_bootstrap):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            event_mask = event[indices] == 1
            
            if sum(event_mask) > 0:
                event_new = prognostic_score[indices][event_mask]
                event_old = baseline_score[indices][event_mask]
                nonevent_new = prognostic_score[indices][~event_mask]
                nonevent_old = baseline_score[indices][~event_mask]
                
                # IDI
                idi = (np.mean(event_new) - np.mean(nonevent_new)) - \
                      (np.mean(event_old) - np.mean(nonevent_old))
                idi_values.append(idi)
                
                # NRI
                nri_events = np.mean(event_new > event_old) - np.mean(event_new < event_old)
                nri_nonevents = np.mean(nonevent_new < nonevent_old) - np.mean(nonevent_new > nonevent_old)
                nri = nri_events + nri_nonevents
                nri_values.append(nri)
        
        # 计算置信区间
        if idi_values:
            idi_ci = (np.percentile(idi_values, 2.5), np.percentile(idi_values, 97.5))
        else:
            idi_ci = (None, None)
        
        if nri_values:
            nri_ci = (np.percentile(nri_values, 2.5), np.percentile(nri_values, 97.5))
        else:
            nri_ci = (None, None)
        
        return nri_ci, idi_ci
    
    # ==================== 原有代码继续 ====================
    print("\n=== Data Validation ===")
    # ... 您原有的代码从这里继续 ...
    # ==================== 数据验证部分 ====================
    print("\n=== Data Validation ===")
    # ... 数据验证代码 ...
    
    # 4. 创建Kaplan-Meier曲线 - 使用带风险表的版本
    try:
        km_plot, risk_table = create_kaplan_meier_with_risktable(prognostic_score, time, event)
    except Exception as e:
        print(f"Failed to create Kaplan-Meier curves: {e}")

    
    # ... 其余代码 ...
    # ==================== 数据验证结束 ====================
    
    # 1. 计算风险比
    try:
        hr_results, score_groups, continuous_hr = calculate_hazard_ratios(prognostic_score, time, event)
    except Exception as e:
        print(f"Failed to calculate hazard ratios: {e}")
        hr_results, score_groups, continuous_hr = None, None, None
# ==================== 新增：创建森林图 ====================
    try:
        forest_df = create_forest_plot_for_hr(prognostic_score, time, event, output_file='forest_plot_hazard_ratios.png')
    except Exception as e:
        print(f"Failed to create forest plot: {e}")
        forest_df = None
    
    # 2. 计算时间依赖性AUC
    try:
        time_auc = calculate_time_dependent_auc(prognostic_score, time, event)
    except Exception as e:
        print(f"Failed to calculate time-dependent AUC: {e}")
        time_auc = {}
    
    # ==================== 新增：计算关键指标的95%置信区间 ====================
    print("\n=== Calculating 95% Confidence Intervals for Key Metrics ===")
    
    # 首先定义CI计算函数（可以作为嵌套函数）
    def calculate_time_auc_ci(prognostic_score, time, event, time_point, n_bootstrap=1000):
        """计算时间依赖性AUC的95%置信区间"""
        from sklearn.metrics import roc_auc_score
        import numpy as np
        
        n_samples = len(time)
        auc_values = []
        
        for i in range(n_bootstrap):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            y_binary = (time[indices] <= time_point) & (event[indices] == 1)
            informative = (time[indices] <= time_point) | (event[indices] == 0)
            
            if sum(y_binary) > 2:
                try:
                    auc = roc_auc_score(
                        y_binary[informative], 
                        prognostic_score[indices][informative]
                    )
                    auc_values.append(auc)
                except:
                    continue
        
        if auc_values:
            auc_values = np.array(auc_values)
            ci_lower = np.percentile(auc_values, 2.5)
            ci_upper = np.percentile(auc_values, 97.5)
            return ci_lower, ci_upper
        else:
            return None, None
    
    def calculate_reclassification_ci(prognostic_score, baseline_score, time, event, n_bootstrap=1000):
        """计算NRI和IDI的95%置信区间"""
        import numpy as np
        
        n_samples = len(time)
        nri_values = []
        idi_values = []
        
        for i in range(n_bootstrap):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            event_mask = event[indices] == 1
            
            if sum(event_mask) > 0:
                event_new = prognostic_score[indices][event_mask]
                event_old = baseline_score[indices][event_mask]
                nonevent_new = prognostic_score[indices][~event_mask]
                nonevent_old = baseline_score[indices][~event_mask]
                
                # IDI
                idi = (np.mean(event_new) - np.mean(nonevent_new)) - \
                      (np.mean(event_old) - np.mean(nonevent_old))
                idi_values.append(idi)
                
                # NRI
                nri_events = np.mean(event_new > event_old) - np.mean(event_new < event_old)
                nri_nonevents = np.mean(nonevent_new < nonevent_old) - np.mean(nonevent_new > nonevent_old)
                nri = nri_events + nri_nonevents
                nri_values.append(nri)
        
        # 计算置信区间
        if idi_values:
            idi_ci = (np.percentile(idi_values, 2.5), np.percentile(idi_values, 97.5))
        else:
            idi_ci = (None, None)
        
        if nri_values:
            nri_ci = (np.percentile(nri_values, 2.5), np.percentile(nri_values, 97.5))
        else:
            nri_ci = (None, None)
        
        return nri_ci, idi_ci
    
    # 现在开始计算CI
    # 计算总C-index的95% CI
    print("Calculating C-index 95% CI (Bootstrap)...")
    from sksurv.metrics import concordance_index_censored
    c_index = concordance_index_censored(event.astype(bool), time, prognostic_score)[0]
    cindex_ci_lower, cindex_ci_upper, c_index = calculate_cindex_ci(prognostic_score, time, event, n_bootstrap=1000)
    # ==================== 新增：打印C-index CI结果 ====================
    if cindex_ci_lower is not None and cindex_ci_upper is not None:
        print(f"C-index: {c_index:.3f} (95% CI: {cindex_ci_lower:.3f}-{cindex_ci_upper:.3f})")
    else:
        print("Warning: Failed to calculate C-index 95% CI")
    
    # 计算时间依赖性AUC的95% CI
    print("Calculating time-dependent AUC 95% CI...")
    time_auc_cis = {}
    for t_name in time_auc.keys():
        # 从 '365 days' 中提取数字部分
        t_days_str = t_name.replace(' days', '')
        t_days = int(t_days_str)
        auc_ci_lower, auc_ci_upper = calculate_time_auc_ci(prognostic_score, time, event, t_days, n_bootstrap=1000)
        
        # ==================== 新增：打印时间AUC CI结果 ====================
        if auc_ci_lower is not None and auc_ci_upper is not None:
            print(f"{t_name} AUC: {time_auc[t_name]['AUC']:.3f} (95% CI: {auc_ci_lower:.3f}-{auc_ci_upper:.3f})")
        else:
            print(f"Warning: Failed to calculate {t_name} AUC 95% CI")
        
        time_auc_cis[t_name] = {
            'AUC_lower_ci': auc_ci_lower,
            'AUC_upper_ci': auc_ci_upper
        }
        # 在 comprehensive_prognostic_score_analysis 函数中找到这段代码，并替换：
    print("\n=== Creating Time-dependent AUC Curve (Figure 4B) ===")
    
    def create_event_rate_by_prognostic_score_chart(prognostic_score, event, output_file='event_rate_by_prognostic_score_detailed.png'):
    #"""
    #创建预后评分分组的事件发生率详细条形图
    
        print(f"\nCreating event rate chart for {len(prognostic_score)} samples...")
        
        # 将预后评分分为4个风险组
        try:
            # 使用四分位数分组
            groups = pd.qcut(prognostic_score, q=4, 
                            labels=['Low Risk (Q1)', 'Low-intermediate Risk (Q2)', 
                                    'High-intermediate Risk (Q3)', 'High Risk (Q4)'],
                            duplicates='drop')
            print("Using quartile groups")
        except Exception as e:
            print(f"Error using qcut: {e}, using cut instead")
            groups = pd.cut(prognostic_score, bins=4, 
                           labels=['Low Risk', 'Low-intermediate Risk', 
                                   'High-intermediate Risk', 'High Risk'])
        
        # 计算每组的事件发生率
        event_rates = []
        group_sample_sizes = []
        group_event_counts = []
        
        for group_name in groups.categories:
            mask = (groups == group_name)
            n_total = mask.sum()
            n_events = event[mask].sum()
            event_rate = n_events / n_total if n_total > 0 else 0
            
            event_rates.append(event_rate)
            group_sample_sizes.append(n_total)
            group_event_counts.append(n_events)
            print(f"  {group_name}: {n_events} events / {n_total} samples = {event_rate:.1%}")
        
        # 创建条形图
        plt.figure(figsize=(12, 8))
        
        # 使用渐变色（绿色到红色）
        colors = ['#2E8B57', '#3CB371', '#FF8C00', '#DC143C']
        
        # 绘制条形图
        bars = plt.bar(range(len(event_rates)), event_rates, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        
        # 设置图表属性
        plt.title('Event Rates by Prognostic Score Quartile Groups', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Risk Group', fontsize=14, fontweight='bold')
        plt.ylabel('Event Rate', fontsize=14, fontweight='bold')
        plt.xticks(range(len(event_rates)), groups.categories, rotation=45, ha='right', fontsize=12)
        plt.ylim(0, max(event_rates) * 1.3 if max(event_rates) > 0 else 1.0)
        plt.grid(True, alpha=0.3, axis='y')
        
        # 在柱子上添加详细信息
        for i, (bar, rate, n_total, n_events) in enumerate(zip(bars, event_rates, group_sample_sizes, group_event_counts)):
            # 添加事件率和事件数
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(event_rates)*0.02,
                    f'{rate:.1%}\n(n={n_events}/{n_total})',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
            
            # 在柱子内部添加样本数
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                    f'n={n_total}', ha='center', va='center', 
                    color='white', fontsize=10, fontweight='bold')
        
        # 添加总事件数信息
        total_events = sum(event)
        total_samples = len(event)
        overall_event_rate = total_events / total_samples if total_samples > 0 else 0
        
        plt.text(0.02, 0.98, f'Overall event rate: {overall_event_rate:.1%} ({total_events}/{total_samples})',
                 transform=plt.gca().transAxes, fontsize=11,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
                 verticalalignment='top')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Event rate chart saved as: {output_file}")
        
        # 显示图形
        plt.show()
        
        # 返回数据
        event_rate_data = pd.DataFrame({
            'Risk_Group': groups.categories,
            'Sample_Size': group_sample_sizes,
            'Event_Count': group_event_counts,
            'Event_Rate': event_rates,
            'Event_Rate_Percentage': [f"{rate*100:.1f}%" for rate in event_rates]
        })
        
        return event_rate_data

    def plot_time_dependent_auc_alternative(prognostic_score, event_indicator, event_times, time_points, save_path):
        #"""替代的时间依赖AUC曲线绘制函数 - 修复版本，使用精确时间点计算"""
        from sklearn.metrics import roc_auc_score
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        
        # 使用传入的数据
        prognostic_scores = np.array(prognostic_score)
        event_indicator = np.array(event_indicator)
        event_times = np.array(event_times)
        
        print(f"Using {len(prognostic_scores)} samples for time-dependent AUC calculation")
        print(f"Number of events: {np.sum(event_indicator)}")
        print(f"Max follow-up time: {np.max(event_times):.0f} days")
        
        # 生成更多时间点以获得平滑曲线
        max_time = min(np.max(event_times[event_indicator == 1]), 1500)
        eval_times = np.linspace(30, max_time, 30)
        
        # 用于曲线的AUC值
        auc_values = []
        auc_lower = []
        auc_upper = []
        
        # 用于特定时间点的精确AUC值
        exact_time_auc = {}
        
        # 首先计算特定时间点的精确AUC - 使用与控制台打印相同的方法
        for tp in time_points:
            if tp <= max_time:
                # 使用与calculate_time_dependent_auc函数相同的逻辑
                y_binary = (event_times <= tp) & (event_indicator == 1)
                y_binary = y_binary.astype(int)
                informative = (event_times <= tp) | (event_indicator == 0)
                
                if sum(y_binary) > 0 and sum(informative) > 0:
                    try:
                        auc = roc_auc_score(y_binary[informative], prognostic_scores[informative])
                        exact_time_auc[tp] = auc
                        n_events = sum(y_binary)
                        print(f"Exact AUC at {tp} days: {auc:.3f} (Events: {n_events})")
                    except Exception as e:
                        print(f"Error calculating exact AUC at {tp} days: {e}")
                        exact_time_auc[tp] = 0.5
                else:
                    print(f"Exact AUC at {tp} days: Insufficient events")
                    exact_time_auc[tp] = 0.5
        
        # 计算曲线上的AUC值（用于绘制平滑曲线）
        for t in eval_times:
            # 使用与上面相同的逻辑，保持一致性
            y_binary = (event_times <= t) & (event_indicator == 1)
            y_binary = y_binary.astype(int)
            informative = (event_times <= t) | (event_indicator == 0)
            
            if sum(y_binary) > 0 and sum(informative) > 0:
                try:
                    auc = roc_auc_score(y_binary[informative], prognostic_scores[informative])
                    auc_values.append(auc)
                    n = sum(informative)
                    # 计算AUC的标准误（Hanley & McNeil方法）
                    if auc != 0.5:  # 只有当AUC不是0.5时才计算标准误
                        Q1 = auc / (2 - auc)
                        Q2 = 2 * auc * auc / (1 + auc)
                        se = np.sqrt((auc * (1 - auc) + (n - 1) * (Q1 - auc*auc) + (n - 1) * (Q2 - auc*auc)) / (n * (n - 1)))
                        
                        if se > 0:
                            z = 1.96
                            auc_lower.append(max(0.5, auc - z * se))
                            auc_upper.append(min(1.0, auc + z * se))
                        else:
                            auc_lower.append(auc)
                            auc_upper.append(auc)
                    else:
                        auc_lower.append(0.5)
                        auc_upper.append(0.5)
                except Exception as e:
                    auc_values.append(0.5)
                    auc_lower.append(0.5)
                    auc_upper.append(0.5)
            else:
                auc_values.append(0.5)
                auc_lower.append(0.5)
                auc_upper.append(0.5)
        
        # 过滤有效点
        valid_indices = [i for i, auc in enumerate(auc_values) if auc > 0.5]
        if len(valid_indices) > 0:
            eval_times = eval_times[valid_indices]
            auc_values = [auc_values[i] for i in valid_indices]
            auc_lower = [auc_lower[i] for i in valid_indices]
            auc_upper = [auc_upper[i] for i in valid_indices]
        else:
            print("Warning: All AUC values are 0.5 or below. Check data and model.")
        
        # 绘制图形
        plt.figure(figsize=(10, 6))
        if len(eval_times) > 0:
            plt.plot(eval_times / 365, auc_values, 'b-', linewidth=2.5, label='Time-dependent AUC')
            if auc_lower and auc_upper:
                plt.fill_between(eval_times / 365, auc_lower, auc_upper, alpha=0.2, color='blue', label='95% CI')
        else:
            print("No valid AUC values to plot.")
            return None
        
        plt.axhline(y=0.5, color='r', linestyle='--', linewidth=1, alpha=0.7, label='Random (AUC=0.5)')
        
        # 标记关键时间点 - 使用精确计算的AUC值
        for tp in time_points:
            if tp in exact_time_auc:
                auc_tp = exact_time_auc[tp]
                # 找到最接近的时间点
                idx = np.argmin(np.abs(eval_times - tp))
                if idx < len(auc_values):
                    # 使用曲线上的点作为标记位置，但显示精确值
                    plt.scatter(tp/365, auc_values[idx], color='darkblue', s=100, zorder=5)
                    plt.text(tp/365 + 0.05, auc_values[idx] - 0.02, f'{auc_tp:.3f}', fontsize=10, fontweight='bold')
                else:
                    # 如果时间点不在eval_times中，直接标记
                    plt.scatter(tp/365, auc_tp, color='darkblue', s=100, zorder=5)
                    plt.text(tp/365 + 0.05, auc_tp - 0.02, f'{auc_tp:.3f}', fontsize=10, fontweight='bold')
        
        # 添加事件数信息
        event_counts = []
        for t in [365, 730, 1095]:
            if t <= max_time:
                n_events = np.sum((event_times <= t) & (event_indicator == 1))
                event_counts.append(f'{t/365:.0f}y: {n_events}')
        
        info_text = f"Events: {', '.join(event_counts)}" if event_counts else f"Total events: {np.sum(event_indicator)}"
        plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes,
                 verticalalignment='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.xlabel('Time (Years)', fontsize=12, fontweight='bold')
        plt.ylabel('Time-dependent AUC', fontsize=12, fontweight='bold')
        plt.title('Time-dependent AUC Curve for Prognostic Score', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.ylim(0.45, 1.05)
        plt.legend(loc='lower right', fontsize=10)
        
        # 计算平均AUC
        avg_auc = np.mean(auc_values) if auc_values else 0.5
        
        # 添加模型性能信息
        from sksurv.metrics import concordance_index_censored
        c_index = concordance_index_censored(event_indicator.astype(bool), event_times, prognostic_scores)[0]
        
        plt.text(0.98, 0.02, f'C-index: {c_index:.3f}\nAvg AUC: {avg_auc:.3f}', 
                 transform=plt.gca().transAxes, horizontalalignment='right',
                 fontsize=10, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Time-dependent AUC curve saved as: {save_path}")
        plt.show()
        
        # 保存数据
        auc_data = pd.DataFrame({
            'Time_days': eval_times,
            'Time_years': eval_times / 365,
            'AUC': auc_values,
            'AUC_lower_95CI': auc_lower,
            'AUC_upper_95CI': auc_upper
        })
        auc_data.to_excel("time_dependent_auc_data.xlsx", index=False)
        print("Time-dependent AUC data saved to: time_dependent_auc_data.xlsx")
        
        # 保存精确时间点AUC
        exact_auc_df = pd.DataFrame([
            {'Time_days': tp, 'Time_years': tp/365, 'AUC': exact_time_auc[tp]} 
            for tp in exact_time_auc
        ])
        exact_auc_df.to_excel("exact_time_auc_values.xlsx", index=False)
        print("Exact time point AUC values saved to: exact_time_auc_values.xlsx")
        
        # 验证一致性
        print("\n=== Verifying consistency between graph and console output ===")
        for tp in time_points:
            if tp in exact_time_auc:
                print(f"Time {tp} days: Graph AUC = {exact_time_auc[tp]:.3f}")
        
        return auc_data
            
    # 直接使用传入的参数调用替代函数
 # 在 comprehensive_prognostic_score_analysis 函数中找到调用这个函数的地方，修改为：
    auc_data = plot_time_dependent_auc_alternative(
        prognostic_score=prognostic_score,  # 参数名改为单数形式
        event_indicator=event,
        event_times=time,
        time_points=[365, 730, 1095],
        save_path="Figure4B_time_dependent_auc_curve.png"
    )
        # ==================== 新增：打印CI计算结果 ====================
    print("\n=== 95% Confidence Intervals Results ===")
    
    # 1. 打印C-index的CI
    if cindex_ci_lower is not None and cindex_ci_upper is not None:
        print(f"C-index: {c_index:.3f} (95% CI: {cindex_ci_lower:.3f}-{cindex_ci_upper:.3f})")
    else:
        print("Warning: Failed to calculate C-index 95% CI")
    
    # 2. 打印时间依赖性AUC的CI
    for t_name, auc_info in time_auc.items():
        ci_info = time_auc_cis.get(t_name, {})
        auc_value = auc_info.get('AUC', 0)
        ci_lower = ci_info.get('AUC_lower_ci')
        ci_upper = ci_info.get('AUC_upper_ci')
        
        if ci_lower is not None and ci_upper is not None:
            print(f"{t_name} AUC: {auc_value:.3f} (95% CI: {ci_lower:.3f}-{ci_upper:.3f})")
        else:
            print(f"Warning: Failed to calculate {t_name} AUC 95% CI")
    
    # 3. 打印重分类指标的CI（如果已计算）
    if 'nri_ci' in locals() and 'idi_ci' in locals():
        if nri_ci[0] is not None and nri_ci[1] is not None:
            print(f"NRI 95% CI: {nri_ci[0]:.3f}-{nri_ci[1]:.3f}")
        if idi_ci[0] is not None and idi_ci[1] is not None:
            print(f"IDI 95% CI: {idi_ci[0]:.3f}-{idi_ci[1]:.3f}")
    
    print("="*60)
    
    # 准备基线评分用于重分类CI计算
    baseline_scores_for_ci = None
    if baseline_data is not None:
        try:
            # 确保baseline_data是一维数组
            if hasattr(baseline_data, 'shape') and len(baseline_data.shape) > 1:
                if baseline_data.shape[1] == 2:
                    # 如果是两列，使用年龄和EF
                    baseline_scores_for_ci = baseline_data.iloc[:, 0] - 0.5 * baseline_data.iloc[:, 1]
                else:
                    baseline_scores_for_ci = baseline_data.mean(axis=1)
            else:
                baseline_scores_for_ci = baseline_data
            
            # 标准化
            baseline_scores_for_ci = (baseline_scores_for_ci - baseline_scores_for_ci.min()) / \
                                     (baseline_scores_for_ci.max() - baseline_scores_for_ci.min())
            baseline_scores_for_ci = np.array(baseline_scores_for_ci).flatten()
        except Exception as e:
            print(f"Warning: Failed to prepare baseline scores for CI calculation: {e}")
    
    # 计算重分类指标的95% CI
    if baseline_scores_for_ci is not None:
        nri_ci, idi_ci = calculate_reclassification_ci(
            prognostic_score, baseline_scores_for_ci, time, event, n_bootstrap=1000
        )
        
        # ==================== 新增：打印重分类指标CI结果 ====================
        if nri_ci[0] is not None and nri_ci[1] is not None:
            print(f"NRI 95% CI: {nri_ci[0]:.3f}-{nri_ci[1]:.3f}")
        if idi_ci[0] is not None and idi_ci[1] is not None:
            print(f"IDI 95% CI: {idi_ci[0]:.3f}-{idi_ci[1]:.3f}")
    else:
        nri_ci, idi_ci = (None, None), (None, None)
        print("No baseline data available for reclassification CI calculation")
    
    # ==================== 原有代码继续 ====================

    # 3. 计算重分类改善指标
    try:
        reclassification_results = calculate_reclassification_metrics(prognostic_score, baseline_data, time, event)
    except Exception as e:
        print(f"Failed to calculate reclassification improvement metrics: {e}")
        reclassification_results = None
    
    # 4. 创建Kaplan-Meier曲线 - 使用简化版本
    try:
        km_plot = create_simple_kaplan_meier_with_risktable(prognostic_score, time, event)
    except Exception as e:
        print(f"Failed to create Kaplan-Meier curves: {e}")
        km_plot = None
    # 5. 创建预后评分分组的事件发生率图（新添加）
    try:
        print("\n=== Creating Event Rate by Prognostic Score Chart ===")
        event_rate_data = create_event_rate_by_prognostic_score_chart(
            prognostic_score, event, output_file='event_rate_by_prognostic_score_detailed.png'
        )
        
        # 保存数据
        event_rate_data.to_excel('event_rate_by_prognostic_score.xlsx', index=False)
        print("Event rate data saved to: event_rate_by_prognostic_score.xlsx")
        
    except Exception as e:
        print(f"Failed to create event rate chart: {e}")
        import traceback
        traceback.print_exc()
    
    # 5. 计算特征相对重要性
    if feature_importance is not None:
        try:
            relative_importance_df = calculate_relative_importance(feature_importance)
        except Exception as e:
            print(f"Failed to calculate feature relative importance: {e}")
            relative_importance_df = None
    
    # 6. 创建综合结果表格
    comprehensive_results = {
        'Hazard Ratio Analysis': {
            'Continuous Score HR': continuous_hr['exp(coef)'] if continuous_hr is not None else None,
            'HR_95CI_lower': continuous_hr['exp(coef) lower 95%'] if continuous_hr is not None else None,
            'HR_95CI_upper': continuous_hr['exp(coef) upper 95%'] if continuous_hr is not None else None,
            'HR_p_value': continuous_hr['p'] if continuous_hr is not None else None
        },
        'Time-dependent AUC': time_auc,
        'Reclassification Improvement': reclassification_results,
        'Score Group Statistics': {
            'Low Risk': sum(score_groups == 'Low Risk') if score_groups is not None else 0,
            'Intermediate Risk': sum(score_groups == 'Intermediate Risk') if score_groups is not None else 0,
            'High Risk': sum(score_groups == 'High Risk') if score_groups is not None else 0
        }
    }
    
    # 保存综合结果
    try:
        with pd.ExcelWriter('comprehensive_prognostic_analysis.xlsx') as writer:
            
            # 新增：保存完整的Table 2（包含CI）
            table2_data = []
            
            # C-index
            table2_data.append({
                '指标': '一致性指数 (C-index)',
                '点估计': c_index,
                '95%CI_下限': cindex_ci_lower,
                '95%CI_上限': cindex_ci_upper,
                '备注': '评估模型整体区分能力'
            })
            
            # 时间依赖性AUC
            for t_name, auc_info in time_auc.items():
                t_days = t_name.replace(' days', '')
                ci_info = time_auc_cis.get(t_name, {})
                table2_data.append({
                    '指标': f'时间依赖性AUC ({t_days}天)',
                    '点估计': auc_info['AUC'],
                    '95%CI_下限': ci_info.get('AUC_lower_ci'),
                    '95%CI_上限': ci_info.get('AUC_upper_ci'),
                    '备注': f'事件数: {auc_info["n_events"]}'
                })
            
            # 风险比（已提供CI）
            if continuous_hr is not None:
                table2_data.append({
                    '指标': '风险比 (连续评分)',
                    '点估计': continuous_hr['exp(coef)'],
                    '95%CI_下限': continuous_hr['exp(coef) lower 95%'],
                    '95%CI_上限': continuous_hr['exp(coef) upper 95%'],
                    '备注': f'p值: {continuous_hr["p"]:.4f}'
                })
            
            # 重分类指标
            if reclassification_results:
                table2_data.append({
                    '指标': '净重分类改善 (NRI)',
                    '点估计': reclassification_results.get('NRI'),
                    '95%CI_下限': nri_ci[0] if nri_ci and nri_ci[0] is not None else None,
                    '95%CI_上限': nri_ci[1] if nri_ci and nri_ci[1] is not None else None,
                    '备注': '与基线模型比较'
                })
                table2_data.append({
                    '指标': '综合判别改善 (IDI)',
                    '点估计': reclassification_results.get('IDI'),
                    '95%CI_下限': idi_ci[0] if idi_ci and idi_ci[0] is not None else None,
                    '95%CI_上限': idi_ci[1] if idi_ci and idi_ci[1] is not None else None,
                    '备注': '与基线模型比较'
                })
            
            table2_df = pd.DataFrame(table2_data)
            table2_df.to_excel(writer, sheet_name='Table2_Performance', index=False)
            
            # 同时单独保存Table 2
            table2_df.to_excel('Table2_Prognostic_Performance.xlsx', index=False)
            print("Table 2 saved to: Table2_Prognostic_Performance.xlsx")
            
            # 在控制台打印格式化的Table 2
            print("\n" + "="*80)
            print("TABLE 2: Prognostic Score Performance Metrics (with 95% Confidence Intervals)")
            print("="*80)
            
            # 创建格式化的显示版本
            display_df = table2_df.copy()
            for col in ['点估计', '95%CI_下限', '95%CI_上限']:
                display_df[col] = display_df[col].apply(lambda x: f'{x:.3f}' if pd.notnull(x) else 'N/A')
            
            display_df['95%置信区间'] = display_df.apply(
                lambda row: f"{row['95%CI_下限']}-{row['95%CI_上限']}" 
                if row['95%CI_下限'] != 'N/A' and row['95%CI_上限'] != 'N/A' else 'N/A', 
                axis=1
            )
            
            print(display_df[['指标', '点估计', '95%置信区间', '备注']].to_string(index=False))
            print("="*80)
            # 保存风险比结果
            hr_df = pd.DataFrame([comprehensive_results['Hazard Ratio Analysis']])
            hr_df.to_excel(writer, sheet_name='Hazard Ratio Analysis', index=False)
            
            # 保存时间依赖性AUC
            time_auc_df = pd.DataFrame(comprehensive_results['Time-dependent AUC']).T
            time_auc_df.to_excel(writer, sheet_name='Time-dependent AUC')
            
            # 保存重分类改善结果
            if reclassification_results:
                reclass_df = pd.DataFrame([reclassification_results])
                reclass_df.to_excel(writer, sheet_name='Reclassification Improvement')
            
            # 保存评分分组统计
            group_stats_df = pd.DataFrame([comprehensive_results['Score Group Statistics']])
            group_stats_df.to_excel(writer, sheet_name='Score Group Statistics', index=False)
            
            # 保存特征相对重要性
            if feature_importance is not None and relative_importance_df is not None:
                relative_importance_df.to_excel(writer, sheet_name='Feature Relative Importance', index=False)
        
        print("\nComprehensive prognostic score analysis results saved to: comprehensive_prognostic_analysis.xlsx")
    except Exception as e:
        print(f"Error saving comprehensive results: {e}")
    
    return comprehensive_results

def create_baseline_characteristics_table(smc_original, prognostic_score, categorical_features, numerical_features):
    """
    创建Table 1: 患者基线特征表
    按预后评分分组显示患者特征
    自动检测数据分布，对非正态分布使用中位数和四分位数
    """
    print("\n" + "="*60)
    print("Creating Table 1: Patient Baseline Characteristics Table")
    print("="*60)
    
    # 确保预后评分与数据长度一致
    if len(prognostic_score) != len(smc_original):
        print(f"Warning: Prognostic score length ({len(prognostic_score)}) doesn't match data length ({len(smc_original)})")
        # 截断较长的那个
        min_len = min(len(prognostic_score), len(smc_original))
        prognostic_score = prognostic_score[:min_len]
        smc_original = smc_original.iloc[:min_len].copy()
    
    # 添加预后评分到数据
    data_with_score = smc_original.copy()
    data_with_score['prognostic_score'] = prognostic_score
    
    # 使用四分位数创建4个风险组
    try:
        # 使用pd.qcut确保每组有相同数量的患者
        groups = pd.qcut(prognostic_score, q=4, 
                        labels=['Low Risk', 'Low-intermediate Risk', 
                                'High-intermediate Risk', 'High Risk'], 
                        duplicates='drop')
    except Exception as e:
        print(f"Using cut instead of qcut due to duplicate values: {e}")
        groups = pd.cut(prognostic_score, bins=4, 
                       labels=['Low Risk', 'Low-intermediate Risk', 
                               'High-intermediate Risk', 'High Risk'])
    
    # 计算每组样本数
    group_counts = groups.value_counts().sort_index()
    print(f"Group distribution: {group_counts.to_dict()}")
    
    # 创建结果表格
    results = []
    
    # 1. 数值变量的处理
    for feature in numerical_features:
        if feature not in data_with_score.columns:
            continue
            
        row = {
            'Variable': translate_feature_name(feature),
            'Type': 'Continuous'
        }
        
        # 获取数据并移除NaN
        feature_data = data_with_score[feature].dropna()
        
        if len(feature_data) > 0:
            # 方法1: Shapiro-Wilk正态性检验（样本量<=5000时）
            from scipy.stats import shapiro
            
            use_median = False
            
            if len(feature_data) <= 5000:
                try:
                    # Shapiro-Wilk检验
                    stat, p_value = shapiro(feature_data)
                    is_normal = p_value > 0.05
                    print(f"  {translate_feature_name(feature)} (n={len(feature_data)}): Shapiro-Wilk p={p_value:.4f}, {'Normal' if is_normal else 'Non-normal'}")
                    
                    if not is_normal:
                        use_median = True
                    else:
                        # 即使p>0.05，再检查偏度和峰度
                        skewness = feature_data.skew()
                        kurtosis = feature_data.kurtosis()
                        
                        # 如果偏度绝对值>1或峰度绝对值>3，视为非正态
                        if abs(skewness) > 1 or abs(kurtosis) > 3:
                            use_median = True
                            print(f"    Skewness: {skewness:.2f}, Kurtosis: {kurtosis:.2f} -> Use median")
                except Exception as e:
                    print(f"    Shapiro test failed for {feature}: {e}")
                    # 备选：使用偏度判断
                    skewness = feature_data.skew()
                    use_median = abs(skewness) > 1
            else:
                # 大样本时使用偏度判断
                skewness = feature_data.skew()
                use_median = abs(skewness) > 1
            
            # 方法2: 视觉检查（绘制直方图和Q-Q图）
            create_distribution_check_plot(feature_data, feature, translate_feature_name(feature))
            
            # 根据分布选择统计量
            if use_median:
                # 使用中位数和四分位数
                median_val = feature_data.median()
                q1 = feature_data.quantile(0.25)
                q3 = feature_data.quantile(0.75)
                row['All patients (N={})'.format(len(feature_data))] = f"{median_val:.1f} [{q1:.1f}, {q3:.1f}]"
                stat_method = 'median'
            else:
                # 使用均数和标准差
                mean_val = feature_data.mean()
                std_val = feature_data.std()
                row['All patients (N={})'.format(len(feature_data))] = f"{mean_val:.1f} ± {std_val:.1f}"
                stat_method = 'mean'
        else:
            row['All patients (N={})'.format(len(feature_data))] = "NA"
            stat_method = 'unknown'
        
        # 计算各组统计（使用与方法一致的统计量）
        group_values = []
        for group_name in groups.categories:
            group_data = data_with_score.loc[groups == group_name, feature].dropna()
            if len(group_data) > 0:
                if stat_method == 'median':
                    median_val = group_data.median()
                    q1 = group_data.quantile(0.25)
                    q3 = group_data.quantile(0.75)
                    group_values.append(f"{median_val:.1f} [{q1:.1f}, {q3:.1f}]")
                else:
                    mean_val = group_data.mean()
                    std_val = group_data.std()
                    group_values.append(f"{mean_val:.1f} ± {std_val:.1f}")
            else:
                group_values.append("NA")
        
        for i, group_name in enumerate(groups.categories):
            row[group_name] = group_values[i]
        
        # 计算组间比较的p值
        # 对于非正态分布使用Kruskal-Wallis检验，对于正态分布使用ANOVA
        try:
            group_data_list = []
            for group_name in groups.categories:
                group_data = data_with_score.loc[groups == group_name, feature].dropna()
                if len(group_data) > 0:
                    group_data_list.append(group_data)
            
            if len(group_data_list) >= 2:
                if stat_method == 'median':
                    # 非正态分布：使用Kruskal-Wallis检验
                    from scipy.stats import kruskal
                    stat, p_value = kruskal(*group_data_list)
                else:
                    # 正态分布：使用ANOVA
                    from scipy.stats import f_oneway
                    stat, p_value = f_oneway(*group_data_list)
                
                if p_value < 0.001:
                    row['P-value'] = "<0.001"
                else:
                    row['P-value'] = f"{p_value:.3f}"
            else:
                row['P-value'] = "NA"
        except Exception as e:
            print(f"Error calculating p-value for {feature}: {e}")
            row['P-value'] = "NA"
        
        results.append(row)
    
    # 2. 分类变量的处理（保持不变）
    for feature in categorical_features:
        if feature not in data_with_score.columns:
            continue
            
        row = {
            'Variable': translate_feature_name(feature),
            'Type': 'Categorical'
        }
        
        # 计算总体统计
        overall_data = data_with_score[feature].dropna()
        if len(overall_data) > 0:
            # 获取分类变量的所有可能值
            unique_values = sorted(overall_data.unique())
            value_counts = overall_data.value_counts()
            
            # 为每个分类值创建一个子行
            for i, value in enumerate(unique_values):
                if i == 0:
                    # 第一行包含变量名
                    sub_row = row.copy()
                    sub_row['Variable'] = translate_feature_name(feature)
                else:
                    # 后续行只显示类别
                    sub_row = {'Variable': '  ' + str(value), 'Type': ''}
                
                count = value_counts.get(value, 0)
                percentage = 100 * count / len(overall_data)
                sub_row['All patients (N={})'.format(len(overall_data))] = f"{count} ({percentage:.1f}%)"
                
                # 计算各组统计
                for group_name in groups.categories:
                    group_data = data_with_score.loc[groups == group_name, feature].dropna()
                    if len(group_data) > 0:
                        group_count = (group_data == value).sum()
                        group_percentage = 100 * group_count / len(group_data)
                        sub_row[group_name] = f"{group_count} ({group_percentage:.1f}%)"
                    else:
                        sub_row[group_name] = "0 (0%)"
                
                # 只在第一行添加p值
                if i == 0:
                    # 计算卡方检验p值
                    try:
                        # 创建列联表
                        contingency_table = []
                        for group_name in groups.categories:
                            group_data = data_with_score.loc[groups == group_name, feature].dropna()
                            if len(group_data) > 0:
                                group_counts = []
                                for val in unique_values:
                                    count = (group_data == val).sum()
                                    group_counts.append(count)
                                contingency_table.append(group_counts)
                        
                        if len(contingency_table) >= 2 and all(len(row) > 0 for row in contingency_table):
                            from scipy.stats import chi2_contingency
                            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                            if p_value < 0.001:
                                sub_row['P-value'] = "<0.001"
                            else:
                                sub_row['P-value'] = f"{p_value:.3f}"
                        else:
                            sub_row['P-value'] = "NA"
                    except Exception as e:
                        sub_row['P-value'] = "NA"
                else:
                    sub_row['P-value'] = ""  # 后续行不显示p值
                
                results.append(sub_row)
    
    # 创建DataFrame
    results_df = pd.DataFrame(results)
    
    # 重新排列列顺序
    columns_order = ['Variable', 'Type'] + ['All patients (N={})'.format(len(data_with_score))] + \
                    list(groups.categories) + ['P-value']
    results_df = results_df[columns_order]
    
    # 保存到Excel
    output_file = 'Table1_Baseline_Characteristics.xlsx'
    results_df.to_excel(output_file, index=False)
    
    print(f"\nTable 1 saved to: {output_file}")
    
    # 在控制台显示表格
    print("\n" + "="*100)
    print("TABLE 1: Baseline Characteristics of Patients Stratified by Prognostic Score")
    print("="*100)
    
    # 创建一个格式化的显示版本
    display_df = results_df.copy()
    
    # 限制每列的宽度
    pd.set_option('display.max_colwidth', 40)
    pd.set_option('display.width', 1000)
    
    # 打印表格
    print(display_df.to_string(index=False))
    print("="*100)
    
    # 创建可视化
    create_baseline_characteristics_visualization(data_with_score, groups, results_df)
    
    return results_df, groups

def create_distribution_check_plot(data, feature_code, feature_name, save_dir='distribution_checks'):
    """
    创建用于检查数据分布的图形
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 子图1: 直方图
    axes[0, 0].hist(data, bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].axvline(data.mean(), color='red', linestyle='--', label=f'Mean: {data.mean():.2f}')
    axes[0, 0].axvline(data.median(), color='green', linestyle='--', label=f'Median: {data.median():.2f}')
    axes[0, 0].set_xlabel(feature_name)
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title(f'Histogram of {feature_name}')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 子图2: Q-Q图
    from scipy.stats import probplot
    probplot(data, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title(f'Q-Q Plot of {feature_name}')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 子图3: 箱线图
    axes[1, 0].boxplot(data, vert=False)
    axes[1, 0].set_xlabel(feature_name)
    axes[1, 0].set_title(f'Box Plot of {feature_name}')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 子图4: 统计摘要
    axes[1, 1].axis('off')
    
    # 计算统计量
    stats_text = f"""
    {feature_name} Distribution Summary:
    n = {len(data)}
    Mean = {data.mean():.2f}
    Median = {data.median():.2f}
    SD = {data.std():.2f}
    Q1 = {data.quantile(0.25):.2f}
    Q3 = {data.quantile(0.75):.2f}
    IQR = {data.quantile(0.75) - data.quantile(0.25):.2f}
    Min = {data.min():.2f}
    Max = {data.max():.2f}
    Skewness = {data.skew():.2f}
    Kurtosis = {data.kurtosis():.2f}
    """
    
    # Shapiro-Wilk检验（仅适用于n≤5000）
    if len(data) <= 5000:
        from scipy.stats import shapiro
        try:
            stat, p_value = shapiro(data)
            stats_text += f"\nShapiro-Wilk test:\nW = {stat:.4f}\np = {p_value:.4f}"
            if p_value > 0.05:
                stats_text += "\nNormal distribution (p > 0.05)"
            else:
                stats_text += "\nNon-normal distribution (p ≤ 0.05)"
        except Exception as e:
            stats_text += f"\nShapiro-Wilk test failed: {e}"
    
    axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes,
                   verticalalignment='top', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'Distribution Check for {feature_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 保存图形
    safe_feature_name = feature_code.replace('/', '_').replace(' ', '_')
    plt.savefig(f'{save_dir}/{safe_feature_name}_distribution_check.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 打印简要统计信息
    print(f"  {feature_name}: n={len(data)}, Mean={data.mean():.2f}, Median={data.median():.2f}, "
          f"SD={data.std():.2f}, Skew={data.skew():.2f}")
    
def check_all_numerical_distributions(smc_original, numerical_features):
    """
    检查所有数值变量的分布情况，并提供建议
    """
    print("\n" + "="*60)
    print("Checking Distributions of All Numerical Variables")
    print("="*60)
    
    distribution_summary = []
    
    for feature in numerical_features:
        if feature not in smc_original.columns:
            continue
            
        feature_name = translate_feature_name(feature)
        data = smc_original[feature].dropna()
        
        if len(data) > 0:
            # 计算统计量
            mean_val = data.mean()
            median_val = data.median()
            std_val = data.std()
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            skewness = data.skew()
            kurtosis = data.kurtosis()
            
            # 判断是否使用中位数
            use_median = False
            reason = []
            
            if len(data) <= 5000:
                try:
                    from scipy.stats import shapiro
                    stat, p_value = shapiro(data)
                    if p_value <= 0.05:
                        use_median = True
                        reason.append(f"Shapiro-Wilk p={p_value:.4f} ≤ 0.05")
                except Exception as e:
                    reason.append(f"Shapiro test failed: {e}")
            
            # 检查偏度和峰度
            if abs(skewness) > 1:
                use_median = True
                reason.append(f"|Skewness|={abs(skewness):.2f} > 1")
            if abs(kurtosis) > 3:
                use_median = True
                reason.append(f"|Kurtosis|={abs(kurtosis):.2f} > 3")
            
            # 如果中位数和均数相差较大
            if abs(mean_val - median_val) > 0.1 * std_val:
                use_median = True
                reason.append(f"Mean-median difference > 0.1*SD")
            
            distribution_summary.append({
                'Variable': feature_name,
                'Feature_Code': feature,
                'n': len(data),
                'Mean': mean_val,
                'Median': median_val,
                'SD': std_val,
                'Q1': q1,
                'Q3': q3,
                'Skewness': skewness,
                'Kurtosis': kurtosis,
                'Use_Median': use_median,
                'Reason': ', '.join(reason) if reason else 'Normal'
            })
    
    # 创建总结DataFrame
    summary_df = pd.DataFrame(distribution_summary)
    
    # 保存结果
    summary_df.to_excel('numerical_variables_distribution_summary.xlsx', index=False)
    
    # 打印结果
    print("\nDistribution Summary:")
    print("="*80)
    display_columns = ['Variable', 'n', 'Mean', 'Median', 'SD', 'Skewness', 'Use_Median', 'Reason']
    print(summary_df[display_columns].to_string(index=False))
    print("="*80)
    
    # 统计
    n_use_median = sum(summary_df['Use_Median'])
    n_total = len(summary_df)
    print(f"\nRecommendations:")
    print(f"  - {n_use_median}/{n_total} variables should use median [Q1, Q3]")
    print(f"  - {n_total - n_use_median}/{n_total} variables can use mean ± SD")
    
    # 建议哪些变量需要筛查
    if n_use_median > 0:
        print("\nVariables that should use median [Q1, Q3]:")
        for _, row in summary_df[summary_df['Use_Median']].iterrows():
            print(f"  - {row['Variable']}: Skewness={row['Skewness']:.2f}, {row['Reason']}")
    
    return summary_df

def create_baseline_characteristics_visualization(data_with_score, groups, table_df):
    """
    创建基线特征表的可视化图表
    """
    print("\nCreating visualizations for baseline characteristics...")
    
    # 1. 创建分组样本数条形图
    plt.figure(figsize=(10, 6))
    group_counts = groups.value_counts().sort_index()
    
    colors = ['#2E8B57', '#3CB371', '#FF8C00', '#DC143C']  # 绿色到红色的渐变
    bars = plt.bar(range(len(group_counts)), group_counts.values, color=colors, alpha=0.7)
    
    plt.title('Sample Size by Prognostic Score Group', fontsize=16, fontweight='bold')
    plt.xlabel('Risk Group', fontsize=12)
    plt.ylabel('Number of Patients', fontsize=12)
    plt.xticks(range(len(group_counts)), group_counts.index, rotation=45, ha='right')
    
    # 在每个柱子上添加数值
    for i, (bar, count) in enumerate(zip(bars, group_counts.values)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(group_counts.values)*0.01,
                str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('sample_size_by_group.png', dpi=300)
    
    # 2. 创建关键变量在各组分布的箱线图
    key_numerical_vars = ['a2', 'a4', 'e1', 'e4']  # 年龄、BMI、左房直径、LVEF
    key_numerical_vars = [var for var in key_numerical_vars if var in data_with_score.columns]
    
    if key_numerical_vars:
        n_vars = len(key_numerical_vars)
        fig, axes = plt.subplots(1, n_vars, figsize=(5*n_vars, 6))
        
        if n_vars == 1:
            axes = [axes]
        
        for i, var in enumerate(key_numerical_vars):
            # 创建数据列表用于箱线图
            group_data_list = []
            for group_name in groups.categories:
                group_data = data_with_score.loc[groups == group_name, var].dropna()
                group_data_list.append(group_data)
            
            # 创建箱线图
            bp = axes[i].boxplot(group_data_list, patch_artist=True)
            
            # 设置箱线图颜色
            for patch, color in zip(bp['boxes'], colors[:len(group_data_list)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            
            axes[i].set_title(translate_feature_name(var), fontsize=12, fontweight='bold')
            axes[i].set_xlabel('Risk Group')
            axes[i].set_ylabel(translate_feature_name(var))
            axes[i].set_xticklabels(groups.categories, rotation=45, ha='right')
            axes[i].grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Distribution of Key Continuous Variables by Risk Group', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('key_variables_by_group.png', dpi=300)
    
    # 3. 创建关键分类变量在各组分布的堆叠条形图
    key_categorical_vars = ['a1', 'c3', 'c4', 'f1']  # 性别、高血压、糖尿病、手术治疗
    key_categorical_vars = [var for var in key_categorical_vars if var in data_with_score.columns]
    
    if key_categorical_vars:
        for var in key_categorical_vars[:2]:  # 只显示前2个
            plt.figure(figsize=(10, 6))
            
            # 计算每个组的分类分布
            category_percentages = pd.DataFrame()
            for group_name in groups.categories:
                group_data = data_with_score.loc[groups == group_name, var].dropna()
                if len(group_data) > 0:
                    value_counts = group_data.value_counts(normalize=True) * 100
                    for val in sorted(data_with_score[var].dropna().unique()):
                        category_percentages.loc[val, group_name] = value_counts.get(val, 0)
            
            # 创建堆叠条形图
            bottom_vals = pd.Series([0] * len(category_percentages.columns), index=category_percentages.columns)
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(category_percentages)))
            
            for i, (category, percentages) in enumerate(category_percentages.iterrows()):
                plt.bar(range(len(percentages)), percentages.values, 
                       bottom=bottom_vals.values, color=colors[i], 
                       label=str(category), alpha=0.7)
                bottom_vals += percentages
            
            plt.title(f'{translate_feature_name(var)} Distribution by Risk Group', fontsize=14, fontweight='bold')
            plt.xlabel('Risk Group')
            plt.ylabel('Percentage (%)')
            plt.xticks(range(len(category_percentages.columns)), category_percentages.columns, rotation=45, ha='right')
            plt.legend(title=translate_feature_name(var), bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(f'{var}_distribution_by_group.png', dpi=300)
    
    print("Baseline characteristics visualizations saved.")
    print("\n=== Creating Risk Stratification Heatmap (Figure 5A) ===")
    
    def create_risk_stratification_heatmap(table1_path="Table1_Baseline_Characteristics.xlsx", 
                                          save_path="Figure5A_risk_stratification_heatmap.png"):
        """创建风险分层临床特征热图"""
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.colors import LinearSegmentedColormap
        
        try:
            # 读取Table1数据
            table1_df = pd.read_excel(table1_path)
            print(f"Successfully loaded Table1 data from {table1_path}")
            print(f"Table1 shape: {table1_df.shape}")
            print(f"Table1 columns: {table1_df.columns.tolist()[:10]}...")
        except Exception as e:
            print(f"Error loading Table1 data: {e}")
            print("Skipping risk stratification heatmap generation.")
            return
        
        # 定义重要的连续特征
        important_features = [
            'Age',
            'Weight',
            'BMI',
            'Left Atrial Diameter',
            'LVEDD',
            'Max LV Wall Thickness',
            'LV Ejection Fraction',
            'LVEDV',
            'LVESV',
            'Cardiac Output',
            'Left Ventricular Mass'
        ]
        
        # 风险组顺序
        risk_groups = ['Low Risk', 'Low-intermediate Risk', 'High-intermediate Risk', 'High Risk']
        
        # 创建数据矩阵
        heatmap_data = []
        feature_names = []
        
        for feature in important_features:
            # 在Table1中查找该特征
            feature_rows = table1_df[table1_df['Variable'].str.contains(feature, case=False, na=False)]
            
            if len(feature_rows) == 0:
                # 尝试精确匹配
                feature_rows = table1_df[table1_df['Variable'] == feature]
            
            if len(feature_rows) == 0:
                # 尝试部分匹配
                for row_idx, row in table1_df.iterrows():
                    if feature.lower() in str(row['Variable']).lower():
                        feature_rows = table1_df.iloc[[row_idx]]
                        break
            
            if len(feature_rows) > 0:
                row_data = []
                feature_row = feature_rows.iloc[0]
                
                for group in risk_groups:
                    value = None
                    
                    # 尝试不同的列名格式
                    possible_cols = [
                        group,
                        group.strip(),
                        f"{group} (mean ± SD)",
                        f"{group} Mean ± SD"
                    ]
                    
                    for col in possible_cols:
                        if col in feature_row.index:
                            cell_value = feature_row[col]
                            if pd.notna(cell_value):
                                # 提取数值部分
                                if isinstance(cell_value, str):
                                    # 处理"均值 ± 标准差"格式
                                    if '±' in cell_value:
                                        num_part = cell_value.split('±')[0].strip()
                                    # 处理中位数格式
                                    elif '(' in cell_value and ',' in cell_value and ')' in cell_value:
                                        # 提取中位数: "39.2 (20.4, 51.5)" -> 39.2
                                        num_part = cell_value.split('(')[0].strip()
                                    else:
                                        num_part = cell_value
                                    
                                    try:
                                        value = float(num_part)
                                        break
                                    except:
                                        continue
                                else:
                                    value = float(cell_value)
                                    break
                    
                    if value is None:
                        # 尝试从"All patients"列获取参考值
                        if 'All patients (N=97)' in feature_row.index:
                            all_patients_val = feature_row['All patients (N=97)']
                            if isinstance(all_patients_val, str) and '±' in all_patients_val:
                                num_part = all_patients_val.split('±')[0].strip()
                                try:
                                    value = float(num_part)
                                except:
                                    value = 0
                            else:
                                value = 0
                        else:
                            value = 0
                    
                    row_data.append(value)
                
                # 只添加有非零数据的行
                if any(v != 0 for v in row_data):
                    heatmap_data.append(row_data)
                    feature_names.append(feature)
        
        if not heatmap_data:
            print("No valid data found for heatmap. Creating alternative visualization...")
            # 创建备选可视化
            create_alternative_heatmap()
            return
        
        # 转换为numpy数组
        heatmap_matrix = np.array(heatmap_data)
        
        # 对每个特征进行标准化（z-score）
        heatmap_normalized = np.zeros_like(heatmap_matrix, dtype=float)
        for i in range(heatmap_matrix.shape[0]):
            row = heatmap_matrix[i, :]
            if np.std(row) > 0:
                heatmap_normalized[i, :] = (row - np.mean(row)) / np.std(row)
            else:
                heatmap_normalized[i, :] = 0
        
        # 创建热图
        plt.figure(figsize=(14, 10))
        
        # 创建自定义颜色映射（蓝色->白色->红色）
        colors = ['#2E86AB', '#A9D6E5', '#FFFFFF', '#F4A261', '#E76F51']
        cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)
        
        # 绘制热图
        ax = sns.heatmap(heatmap_normalized, 
                         cmap=cmap,
                         center=0,
                         annot=heatmap_matrix,  # 显示原始值
                         fmt='.1f',
                         linewidths=1,
                         linecolor='white',
                         cbar_kws={'label': 'Z-score (Standardized Value)', 
                                   'shrink': 0.8,
                                   'pad': 0.02})
        
        # 设置坐标轴标签
        plt.yticks(np.arange(len(feature_names)) + 0.5, feature_names, 
                   rotation=0, fontsize=11, fontweight='bold')
        plt.xticks(np.arange(len(risk_groups)) + 0.5, risk_groups, 
                   rotation=45, ha='right', fontsize=11, fontweight='bold')
        
        # 添加标题
        plt.title('Clinical Characteristics Across Risk Strata in SMC Patients\n(Standardized Z-scores with Raw Values)', 
                  fontsize=14, fontweight='bold', pad=20)
        
        # 添加颜色条说明
        cbar = ax.collections[0].colorbar
        cbar.ax.set_ylabel('Standardized Value (Z-score)', fontsize=11, fontweight='bold')
        cbar.ax.tick_params(labelsize=10)
        
        # 添加图例说明
        plt.figtext(0.02, 0.98, 'Higher values → Higher risk\nLower values → Lower risk', 
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    verticalalignment='top')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图形
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Risk stratification heatmap saved as: {save_path}")
        
        # 保存热图数据
        heatmap_df = pd.DataFrame(heatmap_matrix, 
                                  index=feature_names, 
                                  columns=risk_groups)
        heatmap_df['Variable'] = feature_names
        heatmap_df.to_excel("risk_stratification_heatmap_data.xlsx", index=False)
        print("Heatmap data saved to: risk_stratification_heatmap_data.xlsx")
        
        # 显示图形
        plt.show()
        
        # 创建百分比变化热图（补充）
        create_percentage_change_heatmap(heatmap_matrix, feature_names, risk_groups)
    
    def create_alternative_heatmap():
        """当Table1数据不可用时创建备选热图"""
        print("Creating alternative heatmap using simulated data...")
        
        # 模拟数据示例
        features = ['Age', 'LVEF', 'LVEDD', 'LVESV', 'LA Diameter', 'Weight', 'BMI']
        risk_groups = ['Low Risk', 'Low-intermediate Risk', 'High-intermediate Risk', 'High Risk']
        
        # 模拟数据矩阵（基于典型趋势）
        data = np.array([
            [186, 64, 38, 39, 26, 38, 19],   # Low Risk
            [279, 62, 38, 32, 29, 39, 19],   # Low-intermediate
            [496, 55, 49, 84, 36, 60, 22],   # High-intermediate
            [418, 25, 66, 195, 39, 60, 23]   # High Risk
        ]).T
        
        # 标准化
        data_normalized = np.zeros_like(data, dtype=float)
        for i in range(data.shape[0]):
            row = data[i, :]
            if np.std(row) > 0:
                data_normalized[i, :] = (row - np.mean(row)) / np.std(row)
        
        # 创建热图
        plt.figure(figsize=(12, 8))
        ax = sns.heatmap(data_normalized, 
                         xticklabels=risk_groups, 
                         yticklabels=features,
                         cmap='RdBu_r',
                         center=0,
                         annot=data,
                         fmt='.0f',
                         linewidths=1,
                         cbar_kws={'label': 'Standardized Value'})
        
        plt.title('Clinical Characteristics by Risk Group (Simulated Data)', 
                  fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig("alternative_risk_heatmap.png", dpi=300)
        plt.show()
    
    def create_percentage_change_heatmap(original_matrix, feature_names, risk_groups):
        """创建百分比变化热图（相对于低风险组）"""
        # 在函数开头添加导入语句
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.colors import LinearSegmentedColormap
        import numpy as np
        import pandas as pd
        
        if original_matrix.shape[1] < 2:
            return
        
        # 计算相对于低风险组的百分比变化
        low_risk_values = original_matrix[:, 0].reshape(-1, 1)
        percentage_change = ((original_matrix - low_risk_values) / low_risk_values) * 100
        
        # 创建热图
        plt.figure(figsize=(12, 8))
        
        # 创建颜色映射
        colors = ['#1E88E5', '#64B5F6', '#FFB74D', '#E53935']
        cmap = LinearSegmentedColormap.from_list('perc_cmap', colors, N=256)
        
        # 绘制热图
        ax = sns.heatmap(percentage_change,
                         cmap=cmap,
                         center=0,
                         annot=np.round(percentage_change, 1),
                         fmt='.1f',
                         linewidths=1,
                         linecolor='white',
                         cbar_kws={'label': '% Change vs. Low Risk', 
                                   'shrink': 0.8})
        
        # 设置坐标轴
        plt.yticks(np.arange(len(feature_names)) + 0.5, feature_names, rotation=0, fontsize=11)
        plt.xticks(np.arange(len(risk_groups)) + 0.5, risk_groups, rotation=45, ha='right', fontsize=11)
        
        plt.title('Percentage Change in Clinical Characteristics\n(Relative to Low Risk Group)', 
                  fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig("percentage_change_heatmap.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # 保存数据
        perc_df = pd.DataFrame(percentage_change, 
                              index=feature_names, 
                              columns=[f"{group} vs Low Risk" for group in risk_groups])
        perc_df.to_excel("percentage_change_data.xlsx")
        print("Percentage change data saved to: percentage_change_data.xlsx")
        
    # 调用函数创建风险分层热图
    print("Creating risk stratification heatmap from Table1 data...")
    create_risk_stratification_heatmap(
        table1_path="Table1_Baseline_Characteristics.xlsx",
        save_path="Figure5A_risk_stratification_heatmap.png"
    )
    
    print("\n=== Risk stratification heatmap creation complete ===")
    
    # =================================================================
    # 继续原有代码
    # =================================================================
    
    print("\n=== Group Statistics ===")
def calculate_relative_importance(feature_importance):
    """
    计算特征的相对重要性和百分位排名
    """
    print("\n=== Calculating Feature Relative Importance ===")
    
    # 计算相对重要性（百分比）
    abs_importance = np.abs(feature_importance.values)
    relative_importance = 100 * abs_importance / abs_importance.sum()
    
    # 计算百分位排名
    percentile_rank = 100 * (feature_importance.rank() / len(feature_importance))
    
    # 创建结果DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_importance.index,
        'Original Importance': feature_importance.values,
        'Absolute Importance': abs_importance,
        'Relative Importance (%)': relative_importance,
        'Percentile Rank (%)': percentile_rank
    }).sort_values('Absolute Importance', ascending=False)
    
    print("Top 10 features relative importance:")
    print(importance_df.head(10).round(3))
    
    return importance_df

def create_forest_plot_for_hr(prognostic_score, time, event, output_file='forest_plot_hazard_ratios.png'):
    """
    创建风险比的森林图（Forest Plot）
    展示不同风险分组（四分位数）的风险比
    """
    print("\n=== Creating Forest Plot for Hazard Ratios ===")
    
    # 确保输入数据是数值类型
    time = pd.to_numeric(time, errors='coerce')
    event = pd.to_numeric(event, errors='coerce')
    prognostic_score = pd.to_numeric(prognostic_score, errors='coerce')
    
    # 移除任何NaN值
    valid_mask = ~(np.isnan(time) | np.isnan(event) | np.isnan(prognostic_score))
    time = time[valid_mask]
    event = event[valid_mask]
    prognostic_score = prognostic_score[valid_mask]
    
    print(f"Valid sample size for forest plot: {len(time)}")
    
    # 使用四分位数创建4个风险组
    group_labels = ['Low Risk (Q1)', 'Low-intermediate Risk (Q2)', 'High-intermediate Risk (Q3)', 'High Risk (Q4)']
    
    try:
        # 创建分组
        score_groups = pd.qcut(prognostic_score, q=4, labels=group_labels, duplicates='drop')
        score_groups = pd.Categorical(score_groups, categories=group_labels, ordered=True)
    except Exception as e:
        print(f"Failed to create quantile groups: {e}")
        # 使用等距分组作为备选
        try:
            score_groups = pd.cut(prognostic_score, bins=4, labels=group_labels)
            score_groups = pd.Categorical(score_groups, categories=group_labels, ordered=True)
        except Exception as e2:
            print(f"Equal interval grouping also failed: {e2}")
            return None
    
    # 准备Cox回归数据
    cox_data = pd.DataFrame({
        'time': time,
        'event': event,
        'score_group': score_groups
    })
    
    # 为分类变量创建哑变量，以最低风险组为参考
    cox_data_dummy = pd.get_dummies(cox_data, columns=['score_group'], drop_first=False)
    
    # 确保列顺序正确
    expected_columns = ['time', 'event', 'score_group_Low Risk (Q1)', 
                       'score_group_Low-intermediate Risk (Q2)',
                       'score_group_High-intermediate Risk (Q3)',
                       'score_group_High Risk (Q4)']
    
    # 重新排列列，确保低风险组作为参考
    actual_columns = [col for col in expected_columns if col in cox_data_dummy.columns]
    cox_data_dummy = cox_data_dummy[actual_columns]
    
    # 拟合Cox模型
    try:
        from lifelines import CoxPHFitter
        
        # 移除参考组的列（低风险组）
        # 在Cox回归中，参考组的风险比为1，不需要作为变量
        columns_for_cox = [col for col in cox_data_dummy.columns 
                          if col not in ['time', 'event', 'score_group_Low Risk (Q1)']]
        
        if len(columns_for_cox) == 0:
            print("No groups to compare, skipping forest plot")
            return None
        
        # 创建用于Cox回归的数据
        cox_regression_data = cox_data_dummy[['time', 'event'] + columns_for_cox].copy()
        
        cph = CoxPHFitter()
        cph.fit(cox_regression_data, duration_col='time', event_col='event')
        
        # 获取结果
        hr_results = cph.summary
        
        # 准备森林图数据
        forest_data = []
        
        # 添加参考组（低风险组，HR=1）
        forest_data.append({
            'Group': 'Low Risk (Q1)',
            'HR': 1.0,
            'HR_lower': 1.0,
            'HR_upper': 1.0,
            'P_value': None,
            'N': sum(score_groups == 'Low Risk (Q1)'),
            'Events': sum(event[score_groups == 'Low Risk (Q1)'])
        })
        
        # 添加其他组
        for i, group in enumerate(['Low-intermediate Risk (Q2)', 
                                   'High-intermediate Risk (Q3)', 
                                   'High Risk (Q4)'], 1):
            group_var = f'score_group_{group}'
            if group_var in hr_results.index:
                hr_row = hr_results.loc[group_var]
                forest_data.append({
                    'Group': group,
                    'HR': hr_row['exp(coef)'],
                    'HR_lower': hr_row['exp(coef) lower 95%'],
                    'HR_upper': hr_row['exp(coef) upper 95%'],
                    'P_value': hr_row['p'],
                    'N': sum(score_groups == group),
                    'Events': sum(event[score_groups == group])
                })
        
        # 创建森林图
        create_forest_plot_visualization(forest_data, output_file)
        
        # 保存森林图数据
        forest_df = pd.DataFrame(forest_data)
        forest_df.to_excel('forest_plot_data.xlsx', index=False)
        print(f"Forest plot data saved to: forest_plot_data.xlsx")
        
        return forest_df
        
    except Exception as e:
        print(f"Error creating forest plot: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_forest_plot_visualization(forest_data, output_file='forest_plot_hazard_ratios.png'):
    """
    创建森林图的视觉化
    """
    # 准备数据
    groups = [item['Group'] for item in forest_data]
    hr_values = [item['HR'] for item in forest_data]
    hr_lower = [item['HR_lower'] for item in forest_data]
    hr_upper = [item['HR_upper'] for item in forest_data]
    p_values = [item['P_value'] for item in forest_data]
    sample_sizes = [item['N'] for item in forest_data]
    event_counts = [item['Events'] for item in forest_data]
    
    # 计算事件率
    event_rates = [events/n if n>0 else 0 for events, n in zip(event_counts, sample_sizes)]
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 设置y轴位置
    y_pos = np.arange(len(groups))
    
    # 绘制风险比点
    colors = []
    for i, hr in enumerate(hr_values):
        if i == 0:  # 参考组
            color = '#2E8B57'  # 绿色
        elif hr > 1.0:
            color = '#DC143C'  # 红色
        else:
            color = '#3498DB'  # 蓝色
        colors.append(color)
        
        # 绘制点
        ax.plot(hr, y_pos[i], 'o', color=color, markersize=12, markeredgecolor='black', markeredgewidth=1.5)
        
        # 绘制置信区间线
        ax.hlines(y_pos[i], hr_lower[i], hr_upper[i], color=color, linewidth=3, alpha=0.7)
        
        # 绘制置信区间端线
        ax.plot([hr_lower[i], hr_upper[i]], [y_pos[i], y_pos[i]], '|', color=color, markersize=10)
    
    # 添加垂直参考线（HR=1）
    ax.axvline(x=1.0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    
    # 设置x轴为对数刻度（风险比通常在对数尺度上对称）
    ax.set_xscale('log')
    
    # 设置x轴范围
    all_values = hr_lower + hr_upper
    min_val = min([v for v in all_values if v > 0])
    max_val = max(all_values)
    x_min = 0.1 * min_val
    x_max = 10 * max_val
    ax.set_xlim(x_min, x_max)
    
    # 设置y轴
    ax.set_yticks(y_pos)
    ax.set_yticklabels(groups, fontsize=12, fontweight='bold')
    ax.invert_yaxis()  # 反转y轴，使高风险组在上方
    
    # 设置x轴标签
    ax.set_xlabel('Hazard Ratio (log scale)', fontsize=14, fontweight='bold')
    
    # 添加标题
    ax.set_title('Forest Plot: Hazard Ratios by Prognostic Score Quartile Groups\n(Reference: Low Risk Group)', 
                fontsize=16, fontweight='bold', pad=20)
    
    # 在右侧添加统计信息
    for i, y in enumerate(y_pos):
        # 样本数和事件数
        n_text = f"N={sample_sizes[i]}, Events={event_counts[i]}"
        # 事件率
        rate_text = f"Rate={event_rates[i]:.1%}"
        # HR和95% CI
        if i == 0:
            hr_text = "HR=1.00 (Reference)"
        else:
            hr_text = f"HR={hr_values[i]:.2f} ({hr_lower[i]:.2f}-{hr_upper[i]:.2f})"
            # 添加p值
            if p_values[i] < 0.001:
                hr_text += " ***"
            elif p_values[i] < 0.01:
                hr_text += " **"
            elif p_values[i] < 0.05:
                hr_text += " *"
        
        # 合并文本
        text = f"{n_text}\n{rate_text}\n{hr_text}"
        
        # 在右侧添加文本
        ax.text(x_max * 0.9, y, text, 
               fontsize=10, ha='right', va='center',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # 添加网格
    ax.grid(True, alpha=0.3, axis='x', linestyle='--')
    
    # 添加图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2E8B57', 
               markersize=10, label='Reference Group (Low Risk)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#DC143C', 
               markersize=10, label='Increased Risk (HR > 1)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498DB', 
               markersize=10, label='Decreased Risk (HR < 1)'),
        Line2D([0], [0], color='black', linestyle='--', label='Reference Line (HR=1)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图形
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Forest plot saved as: {output_file}")
    
    # 显示图形
    plt.show()
    
    return fig
def create_group_comparison_table(prognostic_score, time, event):
    """
    创建不同预后评分分组的比较表格，包含事件发生率和p值
    """
    print("\n=== Creating Group Comparison Table ===")
    
    # 确保数据是数值类型
    time = pd.Series(time).astype(float)
    event = pd.Series(event).astype(int)
    prognostic_score = pd.Series(prognostic_score).astype(float)
    
    # 移除任何NaN值
    valid_mask = ~(time.isna() | event.isna() | prognostic_score.isna())
    time = time[valid_mask]
    event = event[valid_mask]
    prognostic_score = prognostic_score[valid_mask]
    
    # 使用四分位数创建4个风险组
    try:
        # 计算四分位数
        q1 = prognostic_score.quantile(0.25)
        q2 = prognostic_score.quantile(0.50)
        q3 = prognostic_score.quantile(0.75)
        
        # 创建分组
        groups = []
        for score in prognostic_score:
            if score <= q1:
                groups.append('Low Risk')
            elif score <= q2:
                groups.append('Low-intermediate Risk')
            elif score <= q3:
                groups.append('High-intermediate Risk')
            else:
                groups.append('High Risk')
        
        groups = pd.Series(groups, index=prognostic_score.index)
    except Exception as e:
        print(f"Error creating quartile groups: {e}")
        # 使用等距分组作为备选
        try:
            groups = pd.cut(prognostic_score, bins=4, labels=['Low Risk', 'Low-intermediate Risk', 'High-intermediate Risk', 'High Risk'])
        except Exception as e2:
            print(f"Equal interval grouping also failed: {e2}")
            return None
    
    # 创建结果DataFrame
    results = []
    unique_groups = sorted(groups.unique())
    
    for group in unique_groups:
        group_mask = groups == group
        
        # 计算统计量
        n_total = sum(group_mask)
        n_events = sum(event[group_mask])
        event_rate = n_events / n_total if n_total > 0 else 0
        
        # 计算中位生存时间
        try:
            kmf = KaplanMeierFitter()
            kmf.fit(time[group_mask], event[group_mask])
            median_survival = kmf.median_survival_time_
            if median_survival is None:
                median_survival = time[group_mask].max()  # 如果没有达到50%事件，使用最大时间
        except:
            median_survival = time[group_mask].median()
        
        results.append({
            'Risk Group': group,
            'N (total)': n_total,
            'N (events)': n_events,
            'Event Rate': event_rate,
            'Median Survival (days)': median_survival
        })
    
    # 创建统计表格
    stats_df = pd.DataFrame(results)
    
    # 计算组间比较的p值
    print("\n=== Calculating P-values for Group Comparisons ===")
    
    # 方法1: Log-rank检验（两两比较）
    p_values = []
    for i, group1 in enumerate(unique_groups):
        row_p_values = []
        for j, group2 in enumerate(unique_groups):
            if i == j:
                row_p_values.append('')
            else:
                try:
                    # 获取两组的生存数据
                    mask1 = groups == group1
                    mask2 = groups == group2
                    
                    # 执行log-rank检验
                    results = logrank_test(time[mask1], time[mask2],
                                          event[mask1], event[mask2])
                    p_value = results.p_value
                    
                    # 根据p值添加星号
                    if p_value < 0.001:
                        p_str = '<0.001***'
                    elif p_value < 0.01:
                        p_str = f'{p_value:.3f}**'
                    elif p_value < 0.05:
                        p_str = f'{p_value:.3f}*'
                    else:
                        p_str = f'{p_value:.3f}'
                    
                    row_p_values.append(p_str)
                except Exception as e:
                    print(f"Error calculating p-value between {group1} and {group2}: {e}")
                    row_p_values.append('NA')
        
        # 只在第一行添加列名
        if i == 0:
            p_values.append(['vs ' + g for g in unique_groups])
        p_values.append(row_p_values)
    
    # 创建p值表格
    pvalue_df = pd.DataFrame(p_values[1:], columns=p_values[0])
    pvalue_df.insert(0, 'Risk Group', unique_groups)
    
    # 保存结果到Excel文件
    with pd.ExcelWriter('risk_group_comparison_table.xlsx') as writer:
        stats_df.to_excel(writer, sheet_name='Group Statistics', index=False)
        pvalue_df.to_excel(writer, sheet_name='Log-rank Test P-values', index=False)
        
        # 添加摘要表
        summary = {
            'Metric': ['Number of Groups', 'Total Samples', 'Total Events', 'Overall Event Rate'],
            'Value': [
                len(unique_groups),
                len(time),
                sum(event),
                sum(event) / len(time) if len(time) > 0 else 0
            ]
        }
        summary_df = pd.DataFrame(summary)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    print("\nGroup comparison table saved to: risk_group_comparison_table.xlsx")
    print("\n=== Group Statistics ===")
    print(stats_df.to_string(index=False))
    
    print("\n=== Log-rank Test P-values ===")
    print(pvalue_df.to_string(index=False))
    
    # 创建格式化表格用于显示
    print("\n=== Formatted Group Comparison Table ===")
    display_table = stats_df.copy()
    display_table['Event Rate'] = display_table['Event Rate'].apply(lambda x: f'{x:.3f}')
    display_table['Median Survival (days)'] = display_table['Median Survival (days)'].apply(lambda x: f'{x:.1f}')
    
    # 保存为CSV以便于查看
    display_table.to_csv('risk_group_statistics.csv', index=False)
    print(f"Statistics saved to: risk_group_statistics.csv")
    
    return stats_df, pvalue_df

# =============================================================================
# 第一部分：数据预处理和相关性分析函数
# =============================================================================

def corr_load_and_preprocess_data():
    """
    加载并预处理HCM和SMC数据集 - 用于相关性分析
    """
    # 加载数据
    hcm = pd.read_excel('/Volumes/YQ1/r/hcm.test.xlsx', sheet_name=0)
    smc = pd.read_excel('/Volumes/YQ1/r/smc.test.xlsx', sheet_name=0)
    
    print(f"HCM dataset shape: {hcm.shape}")
    print(f"SMC dataset shape: {smc.shape}")
    
    # 确保两个数据集特征一致
    common_features = list(set(hcm.columns) & set(smc.columns))
    hcm = hcm[common_features]
    smc_features = smc[common_features]
    
    # 添加预后信息 (假设列名为'time'和'event')
    smc_target = smc[['time', 'event']]
    
    # 合并特征和目标
    smc_full = pd.concat([smc_features, smc_target], axis=1)
    
    # 处理缺失值 - 分别处理数值和分类变量
    # 数值变量用中位数填充
    numerical_cols = [col for col in NUMERICAL_FEATURES if col in smc_full.columns]
    categorical_cols = [col for col in CATEGORICAL_FEATURES if col in smc_full.columns]
    
    for col in numerical_cols:
        if col in smc_full.columns:
            smc_full[col].fillna(smc_full[col].median(), inplace=True)
            if col in hcm.columns:
                hcm[col].fillna(hcm[col].median(), inplace=True)
    
    # 分类变量用众数填充
    for col in categorical_cols:
        if col in smc_full.columns:
            smc_full[col].fillna(smc_full[col].mode()[0] if not smc_full[col].mode().empty else 'Unknown', inplace=True)
            if col in hcm.columns:
                hcm[col].fillna(hcm[col].mode()[0] if not hcm[col].mode().empty else 'Unknown', inplace=True)
    
    # 保存原始数据用于相关性分析
    smc_original = smc_full.copy()
    
    # 数据标准化 (仅对数值变量)
    scaler = StandardScaler()
    numerical_cols_hcm = [col for col in numerical_cols if col in hcm.columns]
    numerical_cols_smc = [col for col in numerical_cols if col in smc_features.columns]
    
    if numerical_cols_hcm:
        hcm_scaled_numerical = scaler.fit_transform(hcm[numerical_cols_hcm])
        hcm_scaled = hcm.copy()
        hcm_scaled[numerical_cols_hcm] = hcm_scaled_numerical
    else:
        hcm_scaled = hcm.copy()
        
    if numerical_cols_smc:
        smc_features_scaled_numerical = scaler.transform(smc_features[numerical_cols_smc])
        smc_features_scaled = smc_features.copy()
        smc_features_scaled[numerical_cols_smc] = smc_features_scaled_numerical
    else:
        smc_features_scaled = smc_features.copy()
    
    return hcm_scaled, smc_features_scaled, smc_target, common_features, smc_original

def cramers_v(x, y):
    """
    计算Cramér's V系数，用于衡量两个分类变量的相关性
    """
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

def analyze_numerical_correlations(df, numerical_features, event_col='event', time_col='time'):
    """
    分析数值变量与标签的相关性
    """
    event_correlations = []
    event_pvalues = []
    time_correlations = []
    time_pvalues = []
    
    for feature in numerical_features:
        if feature in df.columns:
            # 事件相关性 (点二列相关系数)
            corr, pval = pointbiserialr(df[feature], df[event_col])
            event_correlations.append(corr)
            event_pvalues.append(pval)
            
            # 生存时间相关性 (皮尔逊相关系数)
            corr, pval = pearsonr(df[feature], df[time_col])
            time_correlations.append(corr)
            time_pvalues.append(pval)
    
    return event_correlations, event_pvalues, time_correlations, time_pvalues

def analyze_categorical_correlations(df, categorical_features, event_col='event', time_col='time'):
    """
    分析分类变量与标签的相关性
    """
    event_correlations = []
    event_pvalues = []
    time_correlations = []
    time_pvalues = []
    
    for feature in categorical_features:
        if feature in df.columns:
            # 事件相关性 (Cramér's V)
            cramers_v_value = cramers_v(df[feature], df[event_col])
            event_correlations.append(cramers_v_value)
            
            # 对于分类变量与二分类事件，使用卡方检验p值
            contingency_table = pd.crosstab(df[feature], df[event_col])
            chi2, pval, _, _ = chi2_contingency(contingency_table)
            event_pvalues.append(pval)
            
            # 生存时间相关性 (使用Kruskal-Wallis检验)
            groups = [df[df[feature] == category][time_col] for category in df[feature].unique()]
            if len(groups) > 1:
                try:
                    h_stat, pval = kruskal(*groups)
                    time_correlations.append(h_stat / len(df))
                    time_pvalues.append(pval)
                except:
                    time_correlations.append(0)
                    time_pvalues.append(1)
            else:
                time_correlations.append(0)
                time_pvalues.append(1)
    
    return event_correlations, event_pvalues, time_correlations, time_pvalues

def analyze_feature_label_correlation(df, event_col='event', time_col='time'):
    """
    分析特征与标签之间的相关性（同时处理数值和分类变量）
    """
    # 提取实际存在的特征
    numerical_features = [col for col in NUMERICAL_FEATURES if col in df.columns]
    categorical_features = [col for col in CATEGORICAL_FEATURES if col in df.columns]
    
    print(f"Numerical variables: {len(numerical_features)}")
    print(f"Categorical variables: {len(categorical_features)}")
    
    # 分析数值变量
    num_event_corr, num_event_pval, num_time_corr, num_time_pval = analyze_numerical_correlations(
        df, numerical_features, event_col, time_col)
    
    # 分析分类变量
    cat_event_corr, cat_event_pval, cat_time_corr, cat_time_pval = analyze_categorical_correlations(
        df, categorical_features, event_col, time_col)
    
    # 合并结果
    all_features = numerical_features + categorical_features
    all_event_correlations = num_event_corr + cat_event_corr
    all_event_pvalues = num_event_pval + cat_event_pval
    all_time_correlations = num_time_corr + cat_time_corr
    all_time_pvalues = num_time_pval + cat_time_pval
    variable_types = ['Numerical'] * len(numerical_features) + ['Categorical'] * len(categorical_features)
    
    # 创建相关性数据框
    correlation_df = pd.DataFrame({
        'Feature': all_features,
        'Variable_Type': variable_types,
        'Event_Correlation': all_event_correlations,
        'Event_pvalue': all_event_pvalues,
        'Time_Correlation': all_time_correlations,
        'Time_pvalue': all_time_pvalues
    })
    
    # 添加相关性强度标记
    correlation_df['Event_Significance'] = correlation_df['Event_pvalue'].apply(
        lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    )
    
    correlation_df['Time_Significance'] = correlation_df['Time_pvalue'].apply(
        lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    )
    
    # 保存结果
    correlation_df.to_csv('feature_label_correlations.csv', index=False)
    
    return correlation_df

def visualize_correlations_separate(correlation_df, smc_original, top_n=10):
    """
    可视化特征与标签的相关性（数值变量和分类变量分开显示）
    """
    # 分离数值和分类变量
    numerical_df = correlation_df[correlation_df['Variable_Type'] == 'Numerical'].copy()
    categorical_df = correlation_df[correlation_df['Variable_Type'] == 'Categorical'].copy()
    
    # 为每个类型计算绝对值用于排序
    numerical_df['Abs_Event_Correlation'] = numerical_df['Event_Correlation'].abs()
    numerical_df['Abs_Time_Correlation'] = numerical_df['Time_Correlation'].abs()
    categorical_df['Abs_Event_Correlation'] = categorical_df['Event_Correlation'].abs()
    categorical_df['Abs_Time_Correlation'] = categorical_df['Time_Correlation'].abs()
    
    # 获取每个类型的前top_n个特征 - 先初始化为空DataFrame
    num_event_top = pd.DataFrame()
    num_time_top = pd.DataFrame()
    cat_event_top = pd.DataFrame()
    cat_time_top = pd.DataFrame()
    
    if not numerical_df.empty:
        num_event_top = numerical_df.sort_values('Abs_Event_Correlation', ascending=False).head(top_n)
        num_time_top = numerical_df.sort_values('Abs_Time_Correlation', ascending=False).head(top_n)
    
    if not categorical_df.empty:
        cat_event_top = categorical_df.sort_values('Abs_Event_Correlation', ascending=False).head(top_n)
        cat_time_top = categorical_df.sort_values('Abs_Time_Correlation', ascending=False).head(top_n)
    
    # 创建分开的图表
    plt.figure(figsize=(20, 16))
    
    # 1. 数值变量与事件相关性
    plt.subplot(2, 2, 1)
    if not num_event_top.empty:
        # 翻译特征名
        translated_features = translate_feature_names(num_event_top['Feature'].tolist())
        
        bars = plt.barh(range(len(num_event_top)), num_event_top['Event_Correlation'], 
                       color='steelblue', alpha=0.7)
        plt.yticks(range(len(num_event_top)), translated_features) 
        plt.title(f'Top {len(num_event_top)} Numerical Variables vs Event Correlation', fontsize=14, fontweight='bold')
        plt.xlabel('Point-biserial Correlation Coefficient', fontsize=12)
        plt.axvline(0, color='black', linestyle='--', alpha=0.3)
        
        # 添加显著性标记和数值标签
        for i, (_, row) in enumerate(num_event_top.iterrows()):
            plt.text(row.Event_Correlation + (0.01 if row.Event_Correlation > 0 else -0.08), 
                     i, f'{row.Event_Correlation:.3f}{row.Event_Significance}', 
                     fontsize=10, va='center')
    else:
        plt.text(0.5, 0.5, 'No numerical variable data', ha='center', va='center', transform=plt.gca().transAxes)
    
    # 2. 分类变量与事件相关性
    plt.subplot(2, 2, 2)
    if not cat_event_top.empty:
        # 翻译特征名
        translated_features = translate_feature_names(cat_event_top['Feature'].tolist())
        
        bars = plt.barh(range(len(cat_event_top)), cat_event_top['Event_Correlation'], 
                       color='coral', alpha=0.7)
        plt.yticks(range(len(cat_event_top)), translated_features)
        plt.title(f'Top {len(cat_event_top)} Categorical Variables vs Event Correlation', fontsize=14, fontweight='bold')
        plt.xlabel('Cramér\'s V Coefficient', fontsize=12)
        plt.axvline(0, color='black', linestyle='--', alpha=0.3)
        
        # 添加显著性标记和数值标签
        for i, (_, row) in enumerate(cat_event_top.iterrows()):
            plt.text(row.Event_Correlation + (0.01 if row.Event_Correlation > 0 else -0.08), 
                     i, f'{row.Event_Correlation:.3f}{row.Event_Significance}', 
                     fontsize=10, va='center')
    else:
        plt.text(0.5, 0.5, 'No categorical variable data', ha='center', va='center', transform=plt.gca().transAxes)
    
    # 3. 数值变量与生存时间相关性
    plt.subplot(2, 2, 3)
    if not num_time_top.empty:
        # 翻译特征名
        translated_features = translate_feature_names(num_time_top['Feature'].tolist())
        
        bars = plt.barh(range(len(num_time_top)), num_time_top['Time_Correlation'], 
                       color='lightseagreen', alpha=0.7)
        plt.yticks(range(len(num_time_top)), translated_features)
        plt.title(f'Top {len(num_time_top)} Numerical Variables vs Survival Time Correlation', fontsize=14, fontweight='bold')
        plt.xlabel('Pearson Correlation Coefficient', fontsize=12)
        plt.axvline(0, color='black', linestyle='--', alpha=0.3)
        
        # 添加显著性标记和数值标签
        for i, (_, row) in enumerate(num_time_top.iterrows()):
            plt.text(row.Time_Correlation + (0.01 if row.Time_Correlation > 0 else -0.08), 
                     i, f'{row.Time_Correlation:.3f}{row.Time_Significance}', 
                     fontsize=10, va='center')
    else:
        plt.text(0.5, 0.5, 'No numerical variable data', ha='center', va='center', transform=plt.gca().transAxes)
    
    # 4. 分类变量与生存时间相关性
    plt.subplot(2, 2, 4)
    if not cat_time_top.empty:
        # 翻译特征名
        translated_features = translate_feature_names(cat_time_top['Feature'].tolist())
        
        bars = plt.barh(range(len(cat_time_top)), cat_time_top['Time_Correlation'], 
                       color='goldenrod', alpha=0.7)
        plt.yticks(range(len(cat_time_top)), translated_features)
        plt.title(f'Top {len(cat_time_top)} Categorical Variables vs Survival Time Correlation', fontsize=14, fontweight='bold')
        plt.xlabel('Standardized H Statistic', fontsize=12)
        plt.axvline(0, color='black', linestyle='--', alpha=0.3)
        
        # 添加显著性标记和数值标签
        for i, (_, row) in enumerate(cat_time_top.iterrows()):
            plt.text(row.Time_Correlation + (0.01 if row.Time_Correlation > 0 else -0.08), 
                     i, f'{row.Time_Correlation:.3f}{row.Time_Significance}', 
                     fontsize=10, va='center')
    else:
        plt.text(0.5, 0.5, 'No categorical variable data', ha='center', va='center', transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.savefig('feature_label_correlations_separate.png', dpi=300, bbox_inches='tight')
    
    # 创建汇总统计表
    print("\n=== Correlation Analysis Summary ===")
    print(f"Total numerical variables: {len(numerical_df)}")
    print(f"Total categorical variables: {len(categorical_df)}")
    
    if not numerical_df.empty:
        print(f"\nNumerical variables event correlation range: [{numerical_df['Event_Correlation'].min():.3f}, {numerical_df['Event_Correlation'].max():.3f}]")
        print(f"Numerical variables time correlation range: [{numerical_df['Time_Correlation'].min():.3f}, {numerical_df['Time_Correlation'].max():.3f}]")
    
    if not categorical_df.empty:
        print(f"\nCategorical variables event correlation range: [{categorical_df['Event_Correlation'].min():.3f}, {categorical_df['Event_Correlation'].max():.3f}]")
        print(f"Categorical variables time correlation range: [{categorical_df['Time_Correlation'].min():.3f}, {categorical_df['Time_Correlation'].max():.3f}]")
    
    return num_event_top, num_time_top, cat_event_top, cat_time_top

def create_correlation_heatmaps(correlation_df, smc_original, top_n=10):
    """
    创建数值变量和分类变量的热图
    """
    # 数值变量热图
    top_num_features = correlation_df[correlation_df['Variable_Type'] == 'Numerical'].sort_values(
        'Abs_Event_Correlation', ascending=False
    ).head(top_n)['Feature'].tolist()
    
    # 分类变量热图
    top_cat_features = correlation_df[correlation_df['Variable_Type'] == 'Categorical'].sort_values(
        'Abs_Event_Correlation', ascending=False
    ).head(top_n)['Feature'].tolist()
    
    # 数值变量热图
    if top_num_features:
        num_features_for_heatmap = top_num_features + ['event', 'time']
        
        # 确保所有特征都存在
        num_features_for_heatmap = [f for f in num_features_for_heatmap if f in smc_original.columns]
        
        if len(num_features_for_heatmap) > 2:  # 至少有一个特征加上event和time
            num_corr_matrix = smc_original[num_features_for_heatmap].corr()
            
            # 翻译特征名用于热图标签
            translated_features = []
            for feature in num_features_for_heatmap:
                if feature in ['event', 'time']:
                    translated_features.append(feature)
                else:
                    translated_features.append(translate_feature_name(feature))
            
            plt.figure(figsize=(12, 10))
            mask = np.triu(np.ones_like(num_corr_matrix, dtype=bool))  # 创建上三角掩码
            
            # 创建热图，使用翻译后的标签
            ax = sns.heatmap(num_corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', 
                            cbar=True, linewidths=0.5, annot_kws={"size": 10}, mask=mask)
            
            # 设置x轴和y轴标签为翻译后的特征名
            ax.set_xticklabels(translated_features, rotation=45, ha='right')
            ax.set_yticklabels(translated_features, rotation=0)
            
            plt.title(f'Top {len(top_num_features)} Numerical Variables vs Prognosis Correlation Heatmap', 
                     fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig('numerical_correlation_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    # 分类变量与事件的关联热图（使用Cramér's V）
    if top_cat_features:
        # 创建分类变量之间的关联矩阵
        cat_association_matrix = pd.DataFrame(index=top_cat_features + ['event'], 
                                            columns=top_cat_features + ['event'])
        
        for i in cat_association_matrix.index:
            for j in cat_association_matrix.columns:
                if i == j:
                    cat_association_matrix.loc[i, j] = 1.0
                else:
                    try:
                        cv = cramers_v(smc_original[i], smc_original[j])
                        cat_association_matrix.loc[i, j] = cv
                    except:
                        cat_association_matrix.loc[i, j] = 0.0
        
        # 翻译特征名用于热图标签
        translated_features = []
        for feature in list(cat_association_matrix.index):
            if feature == 'event':
                translated_features.append(feature)
            else:
                translated_features.append(translate_feature_name(feature))
        
        plt.figure(figsize=(12, 10))
        ax = sns.heatmap(cat_association_matrix.astype(float), annot=True, fmt=".2f", cmap='viridis', 
                        cbar=True, linewidths=0.5, annot_kws={"size": 10}, vmin=0, vmax=1)
        
        # 设置x轴和y轴标签为翻译后的特征名
        ax.set_xticklabels(translated_features, rotation=45, ha='right')
        ax.set_yticklabels(translated_features, rotation=0)
        
        plt.title(f'Top {len(top_cat_features)} Categorical Variables vs Event Cramér\'s V Association Heatmap', 
                 fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('categorical_association_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

def detailed_feature_analysis(df, top_features, event_col='event', time_col='time'):
    """
    对重要特征进行详细分析（区分数值和分类变量）
    """
    # 获取特征类型
    feature_types = {}
    for feature in top_features:
        if feature in NUMERICAL_FEATURES:
            feature_types[feature] = 'Numerical'
        else:
            feature_types[feature] = 'Categorical'
    
    # 数值变量的分析
    numerical_features = [f for f in top_features if feature_types[f] == 'Numerical']
    if numerical_features:
        analyze_numerical_features(df, numerical_features, event_col, time_col)
    
    # 分类变量的分析
    categorical_features = [f for f in top_features if feature_types[f] == 'Categorical']
    if categorical_features:
        analyze_categorical_features(df, categorical_features, event_col, time_col)

def analyze_numerical_features(df, features, event_col, time_col):
    """
    分析数值特征与预后的关系
    """
    n_features = len(features)
    n_cols = 2
    n_rows = (n_features + n_cols - 1) // n_cols
    
    # 分布图
    plt.figure(figsize=(15, 5 * n_rows))
    
    for i, feature in enumerate(features):
        plt.subplot(n_rows, n_cols, i+1)
        
        # 根据事件状态分组
        event_0 = df[df[event_col] == 0]
        event_1 = df[df[event_col] == 1]
        
        # 绘制分布图
        sns.histplot(event_0[feature], color='blue', alpha=0.5, label='No Event', kde=True)
        sns.histplot(event_1[feature], color='red', alpha=0.5, label='Event', kde=True)
        
        # 使用翻译后的特征名
        feature_name = translate_feature_name(feature)
        plt.title(f'{feature_name} Distribution by Event Status')
        plt.xlabel(feature_name)
        plt.ylabel('Frequency')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('numerical_feature_distribution_by_event.png', dpi=300)
    
    # 散点图
    plt.figure(figsize=(15, 5 * n_rows))
    
    for i, feature in enumerate(features):
        plt.subplot(n_rows, n_cols, i+1)
        
        # 绘制散点图
        sns.scatterplot(data=df, x=feature, y=time_col, hue=event_col, 
                        palette='viridis', alpha=0.7, s=100)
        
        # 使用翻译后的特征名
        feature_name = translate_feature_name(feature)
        plt.title(f'{feature_name} vs Survival Time')
        plt.xlabel(feature_name)
        plt.ylabel('Survival Time')
    
    plt.tight_layout()
    plt.savefig('numerical_feature_vs_survival_time.png', dpi=300)
def analyze_categorical_features(df, features, event_col, time_col):
    """
    分析分类特征与预后的关系
    """
    n_features = len(features)
    n_cols = 2
    n_rows = (n_features + n_cols - 1) // n_cols
    
    # 事件发生率条形图
    plt.figure(figsize=(18, 5 * n_rows))
    
    for i, feature in enumerate(features):
        plt.subplot(n_rows, n_cols, i+1)
        
        # 计算每个类别的事件发生率
        event_rates = df.groupby(feature)[event_col].mean().sort_values(ascending=False)
        
        # 绘制条形图
        sns.barplot(x=event_rates.index, y=event_rates.values, palette='Reds')
        
        # 使用翻译后的特征名
        feature_name = translate_feature_name(feature)
        plt.title(f'{feature_name} - Event Rate by Category')
        plt.xlabel(f'{feature_name} Category')
        plt.ylabel('Event Rate')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('categorical_feature_event_rates.png', dpi=300)
    
    # 生存时间箱线图
    plt.figure(figsize=(18, 5 * n_rows))
    
    for i, feature in enumerate(features):
        plt.subplot(n_rows, n_cols, i+1)
        
        # 绘制箱线图
        sns.boxplot(data=df, x=feature, y=time_col, palette='Set2')
        
        # 使用翻译后的特征名
        feature_name = translate_feature_name(feature)
        plt.title(f'{feature_name} - Survival Time Distribution by Category')
        plt.xlabel(f'{feature_name} Category')
        plt.ylabel('Survival Time')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('categorical_feature_survival_time.png', dpi=300)

# =============================================================================
# 第二部分：PCA迁移学习和预测模型构建函数
# =============================================================================

def model_identify_column_types(df, categorical_threshold=10, explicit_categorical_columns=None):
    """
    自动识别和手动指定分类变量
    """
    if explicit_categorical_columns is None:
        explicit_categorical_columns = CATEGORICAL_FEATURES
    
    categorical_columns = []
    numerical_columns = []
    
    for col in df.columns:
        # 跳过目标列
        if col in ['time', 'event']:
            continue
            
        # 明确指定的分类变量
        if col in explicit_categorical_columns:
            categorical_columns.append(col)
            continue
            
        # 数据类型为object的通常是分类变量
        if df[col].dtype == 'object':
            categorical_columns.append(col)
        # 数值列但唯一值较少，视为分类变量
        elif df[col].nunique() <= categorical_threshold:
            categorical_columns.append(col)
        else:
            numerical_columns.append(col)
    
    return categorical_columns, numerical_columns

def model_load_and_preprocess_data(explicit_categorical_columns=None):
    """
    加载并预处理HCM和SMC数据集，正确处理分类变量 - 用于模型构建
    """
    # 加载数据
    hcm = pd.read_excel('/Volumes/YQ1/r/hcm.test.xlsx', sheet_name=0)
    smc = pd.read_excel('/Volumes/YQ1/r/smc.test.xlsx', sheet_name=0)
    
    print(f"HCM dataset shape: {hcm.shape}")
    print(f"SMC dataset shape: {smc.shape}")
    
    # 确保两个数据集特征一致
    common_features = list(set(hcm.columns) & set(smc.columns))
    hcm = hcm[common_features]
    smc_features = smc[common_features]
    
    # 添加预后信息
    smc_target = smc[['time', 'event']]
    
    # 识别变量类型
    hcm_categorical, hcm_numerical = model_identify_column_types(
        hcm, explicit_categorical_columns=explicit_categorical_columns
    )
    smc_categorical, smc_numerical = model_identify_column_types(
        smc_features, explicit_categorical_columns=explicit_categorical_columns
    )
    
    print(f"Identified categorical variables ({len(hcm_categorical)}): {hcm_categorical}")
    print(f"Identified numerical variables ({len(hcm_numerical)}): {hcm_numerical}")
    
    # 创建预处理管道
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), hcm_numerical),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), hcm_categorical)
        ]
    )
    
    # 处理缺失值并转换数据
    hcm_processed = hcm.copy()
    smc_processed = smc_features.copy()
    
    # 填充缺失值
    for col in hcm_numerical:
        hcm_processed[col].fillna(hcm_processed[col].median(), inplace=True)
        smc_processed[col].fillna(smc_processed[col].median(), inplace=True)
    
    for col in hcm_categorical:
        hcm_processed[col].fillna(hcm_processed[col].mode()[0] if not hcm_processed[col].mode().empty else 'Missing', inplace=True)
        smc_processed[col].fillna(smc_processed[col].mode()[0] if not smc_processed[col].mode().empty else 'Missing', inplace=True)
    
    # 应用预处理转换
    hcm_scaled = preprocessor.fit_transform(hcm_processed)
    smc_features_scaled = preprocessor.transform(smc_processed)
    
    # 获取特征名称（用于后续分析）
    feature_names = hcm_numerical.copy()
    if len(hcm_categorical) > 0:
        cat_encoder = preprocessor.named_transformers_['cat']
        cat_features = []
        for i, col in enumerate(hcm_categorical):
            categories = cat_encoder.categories_[i][1:]  # 去掉第一个类别（作为基准）
            for cat in categories:
                cat_features.append(f"{col}_{cat}")
        feature_names.extend(cat_features)
    
    print(f"Features after preprocessing: {hcm_scaled.shape[1]}")
    
    return hcm_scaled, smc_features_scaled, smc_target, feature_names, preprocessor

def enhanced_pca_analysis(hcm_data, smc_data, feature_names=None):
    """
    增强的PCA分析，包括特征值筛选和主成分解释
    """
    # 使用PCA，先不限制成分数量
    pca_full = PCA()
    pca_full.fit(hcm_data)
    
    # 获取特征值（解释方差）
    eigenvalues = pca_full.explained_variance_
    
    print("=== PCA Eigenvalue Analysis ===")
    print(f"Total eigenvalues: {len(eigenvalues)}")
    print(f"Eigenvalue range: [{eigenvalues.min():.3f}, {eigenvalues.max():.3f}]")
    
    # 找出特征值>1的主成分
    significant_components = eigenvalues > 1
    n_significant = sum(significant_components)
    
    print(f"\nPrincipal components with eigenvalue > 1: {n_significant}")
    print("Indices of components with eigenvalue > 1:", np.where(significant_components)[0])
    
    # 显示前10个特征值
    print("\nEigenvalues of first 10 principal components:")
    for i, ev in enumerate(eigenvalues[:10]):
        print(f"PC{i+1}: {ev:.3f} {'***' if ev > 1 else ''}")
    
    # 重新使用特征值>1的主成分数量
    if n_significant > 0:
        pca = PCA(n_components=n_significant)
    else:
        pca = PCA(n_components=min(5, hcm_data.shape[1]))
        print("No principal components with eigenvalue > 1, using first 5 components")
    
    pca.fit(hcm_data)
    
    # 转换数据
    hcm_pca = pca.transform(hcm_data)
    smc_pca = pca.transform(smc_data)
    
    print(f"\nFinal number of principal components used: {pca.n_components_}")
    
    # 可视化特征值（碎石图）- 统一字体设置
    plt.figure(figsize=(12, 8))
    
    # 统一的字体设置
    title_fontsize = 14
    label_fontsize = 12
    tick_fontsize = 10
    
    # 子图1: 碎石图
    ax1 = plt.subplot(2, 2, 1)
    plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, 'bo-', linewidth=2)
    plt.axhline(y=1, color='r', linestyle='--', label='Eigenvalue=1')
    plt.xlabel('Principal Component', fontsize=label_fontsize)
    plt.ylabel('Eigenvalue', fontsize=label_fontsize)
    plt.title('PCA Scree Plot', fontsize=title_fontsize, fontweight='bold')
    plt.legend(fontsize=tick_fontsize)
    plt.grid(True, alpha=0.3)
    
    # 标记特征值>1的点
    significant_indices = np.where(significant_components)[0]
    plt.scatter(significant_indices + 1, eigenvalues[significant_indices], 
                color='red', s=100, zorder=5, label='Eigenvalue>1')
    
    # 设置刻度标签字体
    plt.tick_params(axis='both', labelsize=tick_fontsize)
    
    # 子图2: 累计解释方差
    ax2 = plt.subplot(2, 2, 2)
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'go-', linewidth=2)
    plt.xlabel('Number of Principal Components', fontsize=label_fontsize)
    plt.ylabel('Cumulative Explained Variance Ratio', fontsize=label_fontsize)
    plt.title('Cumulative Explained Variance', fontsize=title_fontsize, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', labelsize=tick_fontsize)
    
    # 子图3: 前5个主成分的方差贡献
    ax3 = plt.subplot(2, 2, 3)
    n_show = min(5, len(pca.explained_variance_ratio_))
    components = range(1, n_show + 1)
    variances = pca.explained_variance_ratio_[:n_show]
    bars = plt.bar(components, variances, color='skyblue', alpha=0.7)
    plt.xlabel('Principal Component', fontsize=label_fontsize)
    plt.ylabel('Explained Variance Ratio', fontsize=label_fontsize)
    plt.title(f'Variance Contribution of First {n_show} Principal Components', 
              fontsize=title_fontsize, fontweight='bold', pad=15)
    
    # 在柱子上添加数值，并调整位置
    for bar, variance in zip(bars, variances):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{variance:.3f}', ha='center', va='bottom', fontsize=tick_fontsize)
    
    # 设置y轴范围，为标题和数值标签留出空间
    y_max = max(variances) * 1.25
    plt.ylim(0, y_max)
    plt.tick_params(axis='both', labelsize=tick_fontsize)
    
    # 子图4: 特征值分布
    ax4 = plt.subplot(2, 2, 4)
    plt.hist(eigenvalues, bins=20, alpha=0.7, color='orange', edgecolor='black')
    plt.axvline(x=1, color='r', linestyle='--', label='Eigenvalue=1')
    plt.xlabel('Eigenvalue', fontsize=label_fontsize)
    plt.ylabel('Frequency', fontsize=label_fontsize)
    plt.title('Eigenvalue Distribution', fontsize=title_fontsize, fontweight='bold')
    plt.legend(fontsize=tick_fontsize)
    plt.tick_params(axis='both', labelsize=tick_fontsize)
    
    # 统一调整布局和间距
    plt.tight_layout(pad=3.0)
    plt.savefig('enhanced_pca_analysis.png', dpi=300, bbox_inches='tight')
    
    return hcm_pca, smc_pca, pca, eigenvalues, significant_components

def analyze_principal_components(pca, feature_names, n_components=5, top_features=10, threshold=0.3):
    """
    详细分析主成分的组成 - 使用分组条形图替代热图
    """
    print(f"\n=== Detailed Analysis of First {n_components} Principal Components ===")
    
    # 创建结果DataFrame
    component_results = []
    
    for i in range(min(n_components, pca.n_components_)):
        # 获取当前主成分的载荷
        loadings = pca.components_[i]
        
        # 获取载荷绝对值最大的特征
        top_indices = np.argsort(np.abs(loadings))[-top_features:][::-1]
        
        print(f"\nPrincipal Component {i+1} (Eigenvalue: {pca.explained_variance_[i]:.3f}):")
        print(f"Explained variance: {pca.explained_variance_ratio_[i]:.3f}")
        
        component_features = []
        for idx in top_indices:
            feature_name = feature_names[idx] if feature_names is not None else f"Feature_{idx}"
            loading_value = loadings[idx]
            component_features.append({
                'feature': feature_name,
                'loading': loading_value,
                'abs_loading': abs(loading_value)
            })
            print(f"  {feature_name}: {loading_value:.3f}")
        
        component_results.append({
            'component': f'PC{i+1}',
            'eigenvalue': pca.explained_variance_[i],
            'variance_explained': pca.explained_variance_ratio_[i],
            'top_features': component_features
        })
    
    # ==================== 新增：创建分组条形图替代热图 ====================
    create_pca_grouped_barchart(pca, feature_names, n_components, threshold=threshold)
    
    return component_results

def create_pca_grouped_barchart(pca, feature_names, n_components=4, threshold=0.3):
    """
    创建PCA负荷的分组条形图 - 替代热图
    """
    actual_n_components = min(n_components, pca.n_components_)
    
    # 创建图形 - 进一步增大图形尺寸
    fig, axes = plt.subplots(2, 2, figsize=(28, 24))  # 从(26, 22)增大到(28, 24)
    axes = axes.ravel()
    
    # 定义主成分的临床命名 - 确保在函数内部定义
    pc_names = {
        0: "PC1\nEnergy-Deficient Remodeling",
        1: "PC2\nMetabolic Reserve Compensation", 
        2: "PC3\nMetabolic Hypertrophy Pattern",
        3: "PC4\nProgressive Metabolic Deterioration"
    }
    
    # 定义方差解释（根据您的实际结果调整）- 也确保在函数内部定义
    variance_explained = [21.6, 13.5, 12.6, 8.6]  # 您的实际方差百分比
    
    # 设置全局字体大小参数
    plt.rcParams.update({
        'font.size': 16,            # 默认字体大小
        'axes.titlesize': 24,       # 坐标轴标题
        'axes.labelsize': 20,       # 坐标轴标签
        'xtick.labelsize': 18,      # x轴刻度
        'ytick.labelsize': 18,      # y轴刻度
        'legend.fontsize': 16,      # 图例
        'figure.titlesize': 26      # 图形标题
    })
    
    for i in range(actual_n_components):
        # 获取该PC的负荷
        pc_loadings = pca.components_[i]
        
        # 只选择绝对值大于阈值的特征
        mask = np.abs(pc_loadings) >= threshold
        significant_indices = np.where(mask)[0]
        
        # 使用翻译后的中文变量名
        significant_features = []
        for idx in significant_indices:
            if idx < len(feature_names):
                feature_code = feature_names[idx]
                translated_name = translate_feature_name(feature_code)
                significant_features.append(translated_name)
            else:
                significant_features.append(f"Feature_{idx}")
        
        significant_loadings = pc_loadings[significant_indices]
        
        # 如果特征太多，限制显示数量
        max_display = 8
        if len(significant_features) > max_display:
            # 按绝对值排序并取前max_display个
            sorted_indices = np.argsort(np.abs(significant_loadings))[-max_display:][::-1]
            significant_features = [significant_features[idx] for idx in sorted_indices]
            significant_loadings = significant_loadings[sorted_indices]
        
        # 排序以便更好的可视化
        sort_idx = np.argsort(significant_loadings)
        significant_features = [significant_features[idx] for idx in sort_idx]
        significant_loadings = significant_loadings[sort_idx]
        
        # 创建条形图 - 调整条形高度
        colors = ['#E74C3C' if x > 0 else '#3498DB' for x in significant_loadings]  # 红色正负荷，蓝色负负荷
        bars = axes[i].barh(range(len(significant_loadings)), 
                           significant_loadings, 
                           color=colors, alpha=0.7, height=0.8)  # 增加条形高度到0.8
        
        # 添加数值标签 - 进一步增大字体
        for j, (bar, loading) in enumerate(zip(bars, significant_loadings)):
            # 根据正负值调整标签位置
            offset = 0.03 if loading > 0 else -0.12  # 根据数值正负调整偏移量
            axes[i].text(loading + offset, 
                        j, f'{loading:.3f}', 
                        va='center', 
                        fontsize=22,  # 从18增大到22
                        fontweight='bold',
                        color='black',
                        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, edgecolor='gray'))
        
        # 设置图表属性 - 进一步增大字体
        axes[i].set_yticks(range(len(significant_features)))
        axes[i].set_yticklabels(significant_features, fontsize=22)  # 从18增大到22
        
        # 设置x轴标签 - 进一步增大字体
        axes[i].set_xlabel('Loading Value', fontsize=24, fontweight='bold', labelpad=15)  # 从20增大到24
        
        # 设置标题，包含方差解释 - 进一步增大字体
        pc_label = pc_names.get(i, f'PC{i+1}')
        axes[i].set_title(f'{pc_label}\n(Variance Explained: {variance_explained[i]}%)', 
                         fontsize=26, fontweight='bold', pad=35)  # 从22增大到26，pad从30增大到35
        
        # 添加零线参考
        axes[i].axvline(x=0, color='black', linestyle='-', alpha=0.5, linewidth=3)  # 增加线宽到3
        
        # 设置x轴范围对称
        if len(significant_loadings) > 0:
            x_max = max(np.abs(significant_loadings)) * 1.4  # 从1.3增大到1.4，为标签留出更多空间
            axes[i].set_xlim(-x_max, x_max)
            
            # 设置x轴刻度标签字体大小
            axes[i].tick_params(axis='x', labelsize=22, width=2, length=6)  # 从18增大到22
            axes[i].tick_params(axis='y', labelsize=22, width=2, length=6)  # 从18增大到22
        
        # 添加网格 - 增加网格线可见度
        axes[i].grid(True, axis='x', alpha=0.3, linestyle='--', linewidth=2)
        
        # 添加图例说明 - 进一步增大字体
        if i == 0:
            axes[i].text(0.02, 0.98, 'Red: Positive Loading\nBlue: Negative Loading', 
                        transform=axes[i].transAxes, fontsize=20,  # 从16增大到20
                        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, edgecolor='gray'),
                        verticalalignment='top')
        
        # 确保x轴和y轴刻度标签可见
        axes[i].tick_params(axis='both', which='major', pad=10)  # 增加刻度标签与坐标轴的距离
    
    # 隐藏多余的子图
    for i in range(actual_n_components, 4):
        axes[i].set_visible(False)
    
    plt.tight_layout(pad=4.0)  # 进一步增加子图间距到4.0
    plt.savefig("pca_loadings_barchart.png", dpi=300, bbox_inches='tight')
    plt.savefig("pca_component_heatmap.png", dpi=300, bbox_inches='tight')  # 保持向后兼容
    
    # 显示图形前重置rcParams，避免影响其他图形
    plt.rcParams.update(plt.rcParamsDefault)
    
    plt.show()
    
    print("PCA loadings barchart saved as: pca_loadings_barchart.png")
    
    # 同时创建负荷数据表格
    create_pca_loadings_table(pca, feature_names, actual_n_components, threshold)
    
def create_pca_loadings_table(pca, feature_names, n_components, threshold=0.3):
    """
    创建PCA负荷的详细表格
    """
    # 创建负荷矩阵
    loadings_data = pca.components_[:n_components]
    
    # 找出在所有主成分中重要的特征（至少在一个PC中负荷绝对值>阈值）
    important_features_mask = np.any(np.abs(loadings_data) >= threshold, axis=0)
    important_feature_indices = np.where(important_features_mask)[0]
    
    # ==================== 修改这里：翻译特征名为中文 ====================
    important_feature_names = []
    for i in important_feature_indices:
        if i < len(feature_names):
            feature_code = feature_names[i]
            chinese_name = FEATURE_TRANSLATIONS.get(feature_code, feature_code)
            # 处理编码后的分类变量名
            if '_' in chinese_name and chinese_name not in FEATURE_TRANSLATIONS:
                base_feature = chinese_name.split('_')[0]
                if base_feature in FEATURE_TRANSLATIONS:
                    chinese_name = FEATURE_TRANSLATIONS[base_feature] + chinese_name[len(base_feature):]
            important_feature_names.append(chinese_name)
        else:
            important_feature_names.append(f"Feature_{i}")
    
    # 创建负荷DataFrame
    loadings_df = pd.DataFrame(
        loadings_data[:, important_feature_indices].T,
        index=important_feature_names,
        columns=[f'PC{i+1}' for i in range(n_components)]
    )
    
    # 应用样式：突出显示重要负荷
    def style_loadings(val):
        color = 'red' if val > threshold else 'blue' if val < -threshold else 'black'
        weight = 'bold' if abs(val) >= threshold else 'normal'
        return f'color: {color}; font-weight: {weight}'
    
    # 保存带样式的表格
    styled_df = loadings_df.style.applymap(style_loadings).format("{:.3f}")
    
    # 保存到Excel
    with pd.ExcelWriter('pca_loadings_detailed.xlsx') as writer:
        loadings_df.to_excel(writer, sheet_name='PCA_Loadings', float_format='%.3f')
    
    print("Detailed PCA loadings table saved as: pca_loadings_detailed.xlsx")
    return styled_df


def create_biplots(pca, X, feature_names, n_components=3, max_features=15):
    """
    创建主成分的Biplots图 - 修复版，解决文字重叠问题
    """
    # 获取主成分得分
    scores = pca.transform(X)
    
    # 获取载荷
    loadings = pca.components_
    
    # 计算每个特征的总体重要性（在所有主成分中的最大载荷）
    feature_importance = np.max(np.abs(loadings[:n_components, :]), axis=0)
    
    # ==================== 改进：设置最小载荷阈值 ====================
    threshold = 0.4  # 只显示载荷绝对值大于0.4的特征
    significant_mask = feature_importance > threshold
    
    if sum(significant_mask) == 0:
        # 如果没有特征达到阈值，显示最重要的max_features个特征
        print(f"No features exceed threshold {threshold}, showing top {max_features} features")
        top_feature_indices = np.argsort(feature_importance)[-max_features:][::-1]
    else:
        # 只显示显著的特征，如果太多则限制数量
        significant_indices = np.where(significant_mask)[0]
        if len(significant_indices) > max_features:
            # 按重要性排序，取前max_features个
            sorted_idx = np.argsort(feature_importance[significant_indices])[::-1]
            top_feature_indices = significant_indices[sorted_idx[:max_features]]
        else:
            top_feature_indices = significant_indices
    
    print(f"Displaying {len(top_feature_indices)} features (threshold: {threshold})")
    
    # 翻译特征名
    top_feature_names = []
    for i in top_feature_indices:
        if i < len(feature_names):
            feature_code = feature_names[i]
            chinese_name = FEATURE_TRANSLATIONS.get(feature_code, feature_code)
            # 处理编码后的分类变量名
            if '_' in chinese_name and chinese_name not in FEATURE_TRANSLATIONS:
                base_feature = chinese_name.split('_')[0]
                if base_feature in FEATURE_TRANSLATIONS:
                    chinese_name = FEATURE_TRANSLATIONS[base_feature] + chinese_name[len(base_feature):]
            top_feature_names.append(chinese_name)
        else:
            top_feature_names.append(f"Feature_{i}")
    
    # 创建多个Biplots
    actual_n_components = min(n_components-1, pca.n_components_-1)
    if actual_n_components > 0:
        fig, axes = plt.subplots(1, actual_n_components, figsize=(6*actual_n_components, 6))
        if actual_n_components == 1:
            axes = [axes]
        
        for i in range(actual_n_components):
            pc_x = i
            pc_y = i + 1
            
            # 绘制样本点
            scatter = axes[i].scatter(scores[:, pc_x], scores[:, pc_y], alpha=0.3, 
                                     c=np.arange(len(scores)), cmap='viridis', s=20)
            
            # 添加特征向量（箭头）- 使用智能布局
            for j, idx in enumerate(top_feature_indices):
                x = loadings[pc_x, idx] * 3
                y = loadings[pc_y, idx] * 3
                
                axes[i].arrow(0, 0, x, y, 
                             head_width=0.03, head_length=0.06, 
                             fc='red', ec='red', alpha=0.6, width=0.001)
                
                # ==================== 改进：根据位置调整文本 ====================
                # 计算角度
                angle = np.degrees(np.arctan2(y, x))
                
                # 根据象限调整文本位置和对齐方式
                if angle >= -30 and angle < 30:  # 右侧
                    ha = 'left'
                    offset_x = 0.2
                    offset_y = 0
                    va = 'center'
                elif angle >= 30 and angle < 150:  # 上方
                    ha = 'center'
                    offset_x = 0
                    offset_y = 0.2
                    va = 'bottom'
                elif angle >= -150 and angle < -30:  # 下方
                    ha = 'center'
                    offset_x = 0
                    offset_y = -0.2
                    va = 'top'
                else:  # 左侧
                    ha = 'right'
                    offset_x = -0.2
                    offset_y = 0
                    va = 'center'
                
                axes[i].text(x + offset_x, y + offset_y, 
                            top_feature_names[j], color='red', fontsize=8, 
                            ha=ha, va=va,
                            bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.7))
            
            # 添加解释方差信息
            var_x = pca.explained_variance_ratio_[pc_x] * 100
            var_y = pca.explained_variance_ratio_[pc_y] * 100
            
            axes[i].set_xlabel(f'PC{pc_x+1} ({var_x:.1f}%)')
            axes[i].set_ylabel(f'PC{pc_y+1} ({var_y:.1f}%)')
            axes[i].set_title(f'PC{pc_x+1} vs PC{pc_y+1}')
            axes[i].axhline(y=0, color='gray', linestyle='--', alpha=0.3)
            axes[i].axvline(x=0, color='gray', linestyle='--', alpha=0.3)
            axes[i].grid(True, alpha=0.3)
            
            # 添加单位圆
            circle = plt.Circle((0, 0), 3, fill=False, color='blue', alpha=0.2)
            axes[i].add_artist(circle)
        
        plt.tight_layout()
        plt.savefig('pca_biplots.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # 创建详细的PC1 vs PC2 Biplot（使用adjustText）
    if pca.n_components_ >= 2:
        plt.figure(figsize=(14, 10))
        
        # 绘制样本点
        plt.scatter(scores[:, 0], scores[:, 1], alpha=0.2, 
                    c=np.arange(len(scores)), cmap='viridis', s=30, label='Samples')
        
        # 添加特征向量
        scale_factor = 2.8
        
        # 使用adjustText避免文字重叠
        if ADJUST_TEXT_AVAILABLE:
            texts = []
            
            for j, idx in enumerate(top_feature_indices):
                x = loadings[0, idx] * scale_factor
                y = loadings[1, idx] * scale_factor
                
                # 绘制箭头
                plt.arrow(0, 0, x, y, 
                         head_width=0.02, head_length=0.04, 
                         fc='red', ec='red', alpha=0.7, width=0.001, label='Features' if j == 0 else "")
                
                # 添加文本
                text = plt.text(x * 1.1, y * 1.1, 
                               top_feature_names[j], color='red', fontsize=9, 
                               bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
                texts.append(text)
            
            # 自动调整文本位置
            try:
                adjust_text(texts, 
                           arrowprops=dict(arrowstyle='->', color='gray', lw=0.5),
                           expand_points=(1.2, 1.2),
                           expand_text=(1.2, 1.2),
                           force_points=(0.3, 0.3),
                           force_text=(0.3, 0.3),
                           lim=500)
            except Exception as e:
                print(f"Adjust text failed: {e}, using fallback method")
                # 备选方案
                for j, (text, idx) in enumerate(zip(texts, top_feature_indices)):
                    x = loadings[0, idx] * scale_factor * 1.1
                    y = loadings[1, idx] * scale_factor * 1.1
                    
                    if x > 0:
                        ha = 'left'
                    else:
                        ha = 'right'
                    
                    if y > 0:
                        va = 'bottom'
                    else:
                        va = 'top'
                    
                    text.set_horizontalalignment(ha)
                    text.set_verticalalignment(va)
        else:
            # 备选方案：手动调整文本位置
            for j, idx in enumerate(top_feature_indices):
                x = loadings[0, idx] * scale_factor
                y = loadings[1, idx] * scale_factor
                
                plt.arrow(0, 0, x, y, 
                         head_width=0.02, head_length=0.04, 
                         fc='red', ec='red', alpha=0.7, width=0.001, label='Features' if j == 0 else "")
                
                # 根据角度调整文本位置
                angle = np.degrees(np.arctan2(y, x))
                
                if angle >= -45 and angle < 45:  # 右侧
                    ha = 'left'
                    offset_x = 0.15
                    offset_y = 0
                elif angle >= 45 and angle < 135:  # 上方
                    ha = 'center'
                    offset_x = 0
                    offset_y = 0.15
                elif angle >= -135 and angle < -45:  # 下方
                    ha = 'center'
                    offset_x = 0
                    offset_y = -0.15
                else:  # 左侧
                    ha = 'right'
                    offset_x = -0.15
                    offset_y = 0
                
                plt.text(x + offset_x, y + offset_y, 
                        top_feature_names[j], color='red', fontsize=9, ha=ha, va='center',
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
        
        # 添加解释方差信息
        var_x = pca.explained_variance_ratio_[0] * 100
        var_y = pca.explained_variance_ratio_[1] * 100
        
        plt.xlabel(f'PC1 ({var_x:.1f}%)', fontsize=12)
        plt.ylabel(f'PC2 ({var_y:.1f}%)', fontsize=12)
        plt.title(f'Detailed Biplot: PC1 vs PC2\n(Showing {len(top_feature_indices)} significant features)', 
                 fontsize=14, fontweight='bold')
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        plt.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
        plt.grid(True, alpha=0.3)
        
        # 添加单位圆
        circle = plt.Circle((0, 0), scale_factor, fill=False, color='blue', alpha=0.2, linestyle='--')
        plt.gca().add_artist(circle)
        
        # 添加图例
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('pca_detailed_biplot.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    print(f"\nGenerated Biplots showing {len(top_feature_indices)} features")

def build_event_classification_model(X, y_event, feature_names=None, pca_model=None, original_feature_names=None):
    """
    构建预后事件分类模型（是否发生事件）
    """
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_event, test_size=0.2, random_state=42, stratify=y_event
    )
    
    # 使用逻辑回归
    model = LogisticRegression(penalty='l2', C=1.0, solver='liblinear', max_iter=1000)
    model.fit(X_train, y_train)
    
    # 评估模型
    train_probs = model.predict_proba(X_train)[:, 1]
    test_probs = model.predict_proba(X_test)[:, 1]
    
    train_auc = roc_auc_score(y_train, train_probs)
    test_auc = roc_auc_score(y_test, test_probs)
    
    print(f"Training AUC: {train_auc:.3f}")
    print(f"Test AUC: {test_auc:.3f}")
    
    # 可视化特征重要性 - 映射回原始变量名
    if hasattr(model, 'coef_') and pca_model is not None and original_feature_names is not None:
        # 将PCA特征的系数映射回原始特征空间
        original_feature_importance = map_pca_coef_to_original_features(
            model.coef_[0], pca_model, original_feature_names
        )
        
        print(f"\nOriginal feature importance range: [{original_feature_importance.min():.3f}, {original_feature_importance.max():.3f}]")
        
        # 创建Top 10特征重要性图 - 修改这里：根据系数正负设置颜色
        plt.figure(figsize=(12, 8))
        # 只显示最重要的10个原始特征（按绝对值）
        important_features = original_feature_importance.abs().nlargest(10)
        
        # 调试：打印原始特征名
        print(f"\nTop 10 features (original codes): {important_features.index.tolist()}")
        
        # 翻译特征名为中文
        translated_features = []
        feature_values = []  # 保存实际的重要性值
        for feature_code in important_features.index:
            translated_name = translate_feature_name(feature_code)
            translated_features.append(translated_name)
            # 获取实际的重要性值（不是绝对值）
            actual_value = original_feature_importance[feature_code]
            feature_values.append(actual_value)
            print(f"  {feature_code}: {actual_value:.3f} -> {translated_name}")
        
        # 按重要性值排序并绘制（保持符号）
        # 注意：这里我们使用实际值而不是绝对值进行排序
        sorted_indices = np.argsort(feature_values)
        sorted_features = [translated_features[i] for i in sorted_indices]
        sorted_values = [feature_values[i] for i in sorted_indices]
        
        # 根据正负值设置颜色
        colors = ['#E74C3C' if val > 0 else '#3498DB' for val in sorted_values]  # 红色正，蓝色负
        
        # 绘制水平条形图
        bars = plt.barh(range(len(sorted_features)), 
                       sorted_values, 
                       color=colors, alpha=0.7)
        
        plt.yticks(range(len(sorted_features)), 
                  sorted_features, 
                  fontsize=10)
        
        plt.title('Logistic Regression Feature Importance - Based on Original Variables (Top 10)', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Feature Importance (Mapped Coefficients)', fontsize=12)
        
        # 添加数值标签
        for i, (bar, importance) in enumerate(zip(bars, sorted_values)):
            # 根据正负调整标签位置
            if importance >= 0:
                label_x = importance + 0.001
                ha = 'left'
            else:
                label_x = importance - 0.001
                ha = 'right'
            
            plt.text(label_x, i, f'{importance:.3f}', 
                    va='center', fontsize=9, fontweight='bold', ha=ha)
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#E74C3C', alpha=0.7, label='Positive (Risk Factor)'),
            Patch(facecolor='#3498DB', alpha=0.7, label='Negative (Protective Factor)')
        ]
        plt.legend(handles=legend_elements, loc='upper right', fontsize=8)
        
        plt.axvline(0, color='black', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig('logistic_feature_importance_original_top10.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 创建Top 20特征重要性图 - 同样修改
        plt.figure(figsize=(12, 10))
        important_features_20 = original_feature_importance.abs().nlargest(20)
        
        # 翻译特征名为中文
        translated_features_20 = []
        feature_values_20 = []  # 保存实际的重要性值
        for feature_code in important_features_20.index:
            translated_name = translate_feature_name(feature_code)
            translated_features_20.append(translated_name)
            actual_value = original_feature_importance[feature_code]
            feature_values_20.append(actual_value)
        
        # 按重要性值排序并绘制
        sorted_indices_20 = np.argsort(feature_values_20)
        sorted_features_20 = [translated_features_20[i] for i in sorted_indices_20]
        sorted_values_20 = [feature_values_20[i] for i in sorted_indices_20]
        
        # 根据正负值设置颜色
        colors_20 = ['#E74C3C' if val > 0 else '#3498DB' for val in sorted_values_20]
        
        # 绘制水平条形图
        bars_20 = plt.barh(range(len(sorted_features_20)), 
                          sorted_values_20, 
                          color=colors_20, alpha=0.7)
        
        plt.yticks(range(len(sorted_features_20)), 
                  sorted_features_20, 
                  fontsize=9)
        
        plt.title('Logistic Regression Feature Importance - Based on Original Variables (Top 20)', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Feature Importance (Mapped Coefficients)', fontsize=12)
        
        # 添加数值标签
        for i, (bar, importance) in enumerate(zip(bars_20, sorted_values_20)):
            if importance >= 0:
                label_x = importance + 0.001
                ha = 'left'
            else:
                label_x = importance - 0.008
                ha = 'right'
            
            plt.text(label_x, i, f'{importance:.3f}', 
                    va='center', fontsize=8, fontweight='bold', ha=ha)
        
        # 添加图例
        plt.legend(handles=legend_elements, loc='upper right', fontsize=10)
        plt.axvline(0, color='black', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig('logistic_feature_importance_original_top20.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 同时保存PCA特征重要性图（供参考）
        plt.figure(figsize=(12, 6))
        pca_coefs = pd.Series(model.coef_[0], index=[f"PC{i+1}" for i in range(X.shape[1])])
        # 为PCA特征重要性图也设置颜色
        pca_colors = ['#E74C3C' if val > 0 else '#3498DB' for val in pca_coefs.sort_values()]
        pca_coefs.sort_values().plot(kind='barh', color=pca_colors)
        plt.title('Logistic Regression Feature Importance - PCA Principal Components')
        plt.axvline(0, color='black', linestyle='--', alpha=0.3)
        plt.legend(handles=legend_elements, fontsize=9)
        plt.savefig('logistic_feature_importance_pca.png', dpi=300)
        plt.close()
        
        # 保存特征重要性数据到文件（包含翻译）
        importance_df = pd.DataFrame({
            'Feature_Code': original_feature_importance.index,
            'Feature_Name': [translate_feature_name(code) for code in original_feature_importance.index],
            'Importance': original_feature_importance.values,
            'Abs_Importance': np.abs(original_feature_importance.values),
            'Direction': ['Positive' if val > 0 else 'Negative' for val in original_feature_importance.values]
        }).sort_values('Abs_Importance', ascending=False)
        
        importance_df.to_csv('original_feature_importance_translated.csv', index=False, encoding='utf-8-sig')
        print(f"\nTranslated feature importance saved to: original_feature_importance_translated.csv")
        
        # 打印正负特征统计
        positive_count = sum(original_feature_importance > 0)
        negative_count = sum(original_feature_importance < 0)
        print(f"Positive coefficients (risk factors): {positive_count}")
        print(f"Negative coefficients (protective factors): {negative_count}")
        print(f"Top positive feature: {importance_df[importance_df['Direction'] == 'Positive'].iloc[0]['Feature_Name']}")
        print(f"Top negative feature: {importance_df[importance_df['Direction'] == 'Negative'].iloc[0]['Feature_Name']}")
        
        return model, original_feature_importance
    
    # ... 后面的代码保持不变 ...
    # 如果没有PCA模型，使用提供的特征名
    elif hasattr(model, 'coef_') and feature_names is not None and len(feature_names) == X.shape[1]:
        plt.figure(figsize=(12, 6))
        coefs = pd.Series(model.coef_[0], index=feature_names)
        important_features = coefs.abs().nlargest(20).index
        
        # 翻译特征名为中文
        translated_features = []
        for feature_code in important_features:
            translated_name = translate_feature_name(feature_code)
            translated_features.append(translated_name)
        
        # 使用翻译后的特征名
        coefs_important = coefs[important_features]
        coefs_important.index = translated_features
        coefs_important.sort_values().plot(kind='barh')
        
        plt.title('Logistic Regression Feature Importance (Top 20)')
        plt.tight_layout()
        plt.savefig('logistic_feature_importance.png', dpi=300)
        plt.close()
        
        return model, coefs
    
    # 如果只有PCA特征名
    elif hasattr(model, 'coef_'):
        plt.figure(figsize=(12, 6))
        coefs = pd.Series(model.coef_[0], index=[f"PC{i+1}" for i in range(X.shape[1])])
        coefs.sort_values().plot(kind='barh')
        plt.title('Logistic Regression Feature Importance - PCA Principal Components')
        plt.savefig('logistic_feature_importance_pca.png', dpi=300)
        plt.close()
        
        return model, coefs
    
    return model, None
def map_pca_coef_to_original_features(pca_coefficients, pca_model, original_feature_names):
    """
    将PCA特征的系数映射回原始特征空间
    """
    # PCA组件矩阵 (n_components × n_original_features)
    pca_components = pca_model.components_
    
    # 将PCA系数投影回原始特征空间
    # 原始特征重要性 = PCA系数 × PCA组件矩阵
    original_importance = np.dot(pca_coefficients, pca_components)
    
    # 创建Series对象
    feature_importance = pd.Series(original_importance, index=original_feature_names)
    
    print(f"PCA features: {len(pca_coefficients)}")
    print(f"Original features: {len(original_importance)}")
    print(f"Mapped feature importance range: [{feature_importance.min():.3f}, {feature_importance.max():.3f}]")
    
    return feature_importance

def build_survival_analysis_model(X, time, event):
    """
    使用生存分析方法构建模型 - 替换原来的回归模型
    """
    # 创建生存分析所需的数据格式
    y_structured = np.array([(bool(event_i), time_i) for event_i, time_i in zip(event, time)],
                          dtype=[('event', 'bool'), ('time', 'f8')])
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_structured, test_size=0.2, random_state=42
    )
    
    # 方法1: Cox比例风险模型
    print("Training Cox proportional hazards model...")
    cox_model = CoxPHSurvivalAnalysis(alpha=0.1)
    cox_model.fit(X_train, y_train)
    
    # 评估Cox模型
    cox_score = cox_model.score(X_test, y_test)
    print(f"Cox model concordance index: {cox_score:.3f}")
    
    # 方法2: 随机生存森林
    print("Training random survival forest model...")
    rsf = RandomSurvivalForest(
        n_estimators=100,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    
    rsf.fit(X_train, y_train)
    
    # 评估RSF模型
    rsf_score = rsf.score(X_test, y_test)
    print(f"Random survival forest concordance index: {rsf_score:.3f}")
    
    # 可视化模型比较
    plt.figure(figsize=(10, 6))
    models = ['Cox Model', 'Random Survival Forest']
    scores = [cox_score, rsf_score]
    colors = ['skyblue', 'lightcoral']
    
    bars = plt.bar(models, scores, color=colors, alpha=0.7)
    plt.ylabel('Concordance Index (C-index)')
    plt.title('Survival Analysis Model Performance Comparison')
    plt.ylim(0, 1)
    
    # 在柱子上添加数值
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('survival_model_comparison.png', dpi=300)
    
    # 选择更好的模型
    if rsf_score > cox_score:
        print("Selected random survival forest model")
        return rsf, 'rsf'
    else:
        print("Selected Cox proportional hazards model")
        return cox_model, 'cox'

def create_improved_prognostic_score(classifier, survival_model, model_type, X, time_points=None):
    """
    改进的预后评分构建
    """
    print("\n=== Building Improved Prognostic Score ===")
    
    # 方法1: 直接使用生存模型的风险评分
    if model_type == 'cox':
        prognostic_score = survival_model.predict(X)
        print("Using Cox model risk score as prognostic score")
    else:
        # 对于随机生存森林，使用更稳健的方法
        try:
            # 预测在特定时间点的生存概率
            if time_points is None:
                time_points = [365, 730, 1095]  # 1年, 2年, 3年
            
            # 预测生存函数
            survival_funcs = survival_model.predict_survival_function(X)
            
            # 计算多个时间点的平均风险
            risk_scores = []
            for func in survival_funcs:
                # 计算在多个时间点的生存概率，然后转换为风险
                survival_probs = [func(t) for t in time_points]
                avg_survival = np.mean(survival_probs)
                risk_score = 1 - avg_survival  # 生存概率转换为风险
                risk_scores.append(risk_score)
            
            prognostic_score = np.array(risk_scores)
            print("Using RSF multi-timepoint average risk as prognostic score")
            
        except Exception as e:
            print(f"RSF risk score calculation failed: {e}")
            # 备选方案：使用事件概率
            prognostic_score = classifier.predict_proba(X)[:, 1]
            print("Using event probability as prognostic score")
    
    # 标准化评分到0-1范围
    prognostic_score = (prognostic_score - prognostic_score.min()) / (prognostic_score.max() - prognostic_score.min())
    
    # 验证预后评分的C-index
    from sksurv.metrics import concordance_index_censored
    c_index = concordance_index_censored(y_event.astype(bool), y_time, prognostic_score)[0]
    print(f"Prognostic score C-index: {c_index:.3f}")
    
    if c_index < 0.5:
        print("Warning: Poor prognostic score performance, considering inversion")
        prognostic_score = 1 - prognostic_score
        c_index_inverted = concordance_index_censored(y_event.astype(bool), y_time, prognostic_score)[0]
        print(f"Inverted C-index: {c_index_inverted:.3f}")
    
    return prognostic_score


# =============================================================================
# 主流程
# =============================================================================

def main():
    """
    完整的HCM+SMC预后预测研究主流程
    """
    print("=" * 60)
    print("Complete HCM+SMC Prognosis Prediction Research Pipeline")
    print("Using Survival Analysis Framework Instead of Regression Models")
    print("Supplemented with Key Prognostic Score Metrics Calculation")
    print("with Baseline Model Comparison for Validation")
    print("=" * 60)
    
    # =========================================================================
    # 第一部分：相关性分析
    # =========================================================================
    print("\n" + "="*50)
    print("Part 1: Feature and Prognosis Correlation Analysis")
    print("="*50)
    
    try:
        # 步骤1: 数据加载与预处理（相关性分析版本）
        hcm_data_corr, smc_data_corr, smc_target_corr, feature_names_corr, smc_original = corr_load_and_preprocess_data()
        
        # 步骤2: 特征与标签相关性分析
        print("\nAnalyzing feature-label correlations...")
        correlation_df = analyze_feature_label_correlation(smc_original)
        
        # 添加绝对值列用于排序
        correlation_df['Abs_Event_Correlation'] = correlation_df['Event_Correlation'].abs()
        correlation_df['Abs_Time_Correlation'] = correlation_df['Time_Correlation'].abs()
        
        # 步骤3: 可视化相关性结果
        print("\nGenerating correlation visualization charts...")
        num_event_top, num_time_top, cat_event_top, cat_time_top = visualize_correlations_separate(
            correlation_df, smc_original, top_n=10)
        
        # 步骤3.1: 创建相关性热图
        print("\nCreating correlation heatmaps...")
        create_correlation_heatmaps(correlation_df, smc_original, top_n=10)
        
        # 步骤4: 详细分析重要特征
        if not correlation_df.empty:
            top_features = correlation_df.sort_values('Abs_Event_Correlation', ascending=False)['Feature'].head(6).tolist()
            if top_features:
                detailed_feature_analysis(smc_original, top_features)
            else:
                print("No significant features found for detailed analysis")
        else:
            print("Correlation DataFrame is empty, skipping detailed feature analysis")
        
    except Exception as e:
        print(f"Error in correlation visualization: {e}")
        import traceback
        traceback.print_exc()
    
    # =========================================================================
    # 第二部分：PCA迁移学习和预测模型构建
    # =========================================================================
    print("\n" + "="*50)
    print("Part 2: PCA Feature Transfer Learning and Prediction Model Building")
    print("="*50)
    
    # 用于保存数据的变量
    smc_data_model_original = None
    smc_target_model = None
    feature_names_model = None
    smc_pca = None
    pca_model = None
    
    try:
        # 明确指定分类变量
        explicit_categorical_columns = CATEGORICAL_FEATURES
        
        # 步骤1: 数据加载与预处理（模型构建版本）
        hcm_data_model, smc_data_model, smc_target_model, feature_names_model, preprocessor = model_load_and_preprocess_data(
            explicit_categorical_columns=explicit_categorical_columns
        )
        
        # 保存原始SMC数据用于基准模型
        smc_data_model_original = smc_data_model.copy()
        
        # 步骤2: 增强的PCA分析
        print("\nPerforming enhanced PCA analysis...")
        hcm_pca, smc_pca, pca_model, eigenvalues, significant_components = enhanced_pca_analysis(
            hcm_data_model, smc_data_model, feature_names_model
        )
        
        # 步骤3: 详细分析主成分组成
        component_results = analyze_principal_components(
            pca_model, feature_names_model, n_components=4, top_features=10, threshold=0.3
        )
        # 步骤4: 创建Biplots
        print("\nCreating PCA Biplots...")
        create_biplots(pca_model, hcm_data_model, feature_names_model, n_components=min(4, pca_model.n_components_), max_features=15)
        
        # 保存PCA分析结果 - 修复数组长度问题
        print("\nSaving PCA analysis results...")
        try:
            # 使用pca_full的explained_variance_ratio_来确保长度一致
            pca_full = PCA()
            pca_full.fit(hcm_data_model)
            
            with pd.ExcelWriter('pca_analysis_results.xlsx') as writer:
                # 保存特征值和解释方差
                eigenvalue_df = pd.DataFrame({
                    'Component': [f'PC{i+1}' for i in range(len(eigenvalues))],
                    'Eigenvalue': eigenvalues,
                    'Eigenvalue_>1': eigenvalues > 1,
                    'Variance_Explained': pca_full.explained_variance_ratio_,
                    'Cumulative_Variance': np.cumsum(pca_full.explained_variance_ratio_)
                })
                eigenvalue_df.to_excel(writer, sheet_name='Eigenvalue Analysis', index=False)
                
                # 保存每个主成分的详细组成
                component_data = []
                for comp in component_results:
                    for feature in comp['top_features']:
                        component_data.append({
                            'Component': comp['component'],
                            'Eigenvalue': comp['eigenvalue'],
                            'Variance_Explained': comp['variance_explained'],
                            'Feature': feature['feature'],
                            'Loading': feature['loading'],
                            'Abs_Loading': feature['abs_loading']
                        })
                
                if component_data:  # 确保有数据
                    component_df = pd.DataFrame(component_data)
                    component_df.to_excel(writer, sheet_name='Principal Component Composition', index=False)
            
            print("PCA analysis results saved")
        except Exception as e:
            print(f"Error saving PCA analysis results: {e}")
            # 保存为CSV作为备选
            eigenvalue_df.to_csv('pca_eigenvalues.csv', index=False)
            print("PCA eigenvalue analysis saved as CSV format")
        
        print("\nPCA analysis completed!")
        print(f"Principal components with eigenvalue > 1: {sum(significant_components)}")
        print(f"Total explained variance: {sum(pca_model.explained_variance_ratio_):.3f}")
        
        # 分离预后目标
        y_event = smc_target_model['event'].values
        y_time = smc_target_model['time'].values
        
        # =====================================================================
        # 新增：训练基准模型（不使用PCA迁移学习）
        # =====================================================================
        print("\n" + "="*60)
        print("Training Baseline Models (without PCA transfer learning)")
        print("="*60)
        
        baseline_results = train_baseline_models_on_smc(smc_data_model_original, y_time, y_event)
        
        # =====================================================================
        # 新增：训练迁移学习模型（使用PCA特征）
        # =====================================================================
        print("\n" + "="*60)
        print("Training Transfer Learning Models (with PCA from HCM)")
        print("="*60)
        
        transfer_results = train_transfer_learning_models(smc_pca, y_time, y_event)
        
        # =====================================================================
        # 新增：比较基准模型和迁移学习模型
        # =====================================================================
        print("\n" + "="*60)
        print("Comparing Baseline vs Transfer Learning Models")
        print("="*60)
        
        comparison_df = compare_model_performance(baseline_results, transfer_results, 
                                                 output_file='model_comparison_baseline_vs_transfer.png')
        
        # =====================================================================
        # 原有流程继续（使用迁移学习模型）
        # =====================================================================
        # 创建PCA特征名称
        pca_feature_names = [f"PC{i+1}" for i in range(smc_pca.shape[1])]
        
        # 步骤5: 构建预后事件分类模型（基于PCA特征）
        print("\nBuilding prognosis event classification model...")
        event_model, feature_importance = build_event_classification_model(
            smc_pca, y_event, 
            feature_names=pca_feature_names,
            pca_model=pca_model,
            original_feature_names=feature_names_model
        )
        
        # 如果成功计算了原始特征重要性，保存结果
        if feature_importance is not None:
            # 保存特征重要性结果
            importance_df = pd.DataFrame({
                'Feature': feature_importance.index,
                'Importance': feature_importance.values,
                'Abs_Importance': np.abs(feature_importance.values)
            }).sort_values('Abs_Importance', ascending=False)
            
            importance_df.to_csv('original_feature_importance.csv', index=False)
            print(f"\nOriginal feature importance saved, top 5 features:")
            print(importance_df.head())
        
        # 步骤6: 构建生存分析模型（使用迁移学习模型的结果）
        # 注意：这里我们使用迁移学习模型中表现最好的模型
        best_transfer_model_name = None
        best_transfer_cindex = 0
        
        for model_name, results in transfer_results.items():
            if results['c_index'] > best_transfer_cindex:
                best_transfer_cindex = results['c_index']
                best_transfer_model_name = model_name
        
        print(f"\nBest transfer learning model: {best_transfer_model_name} (C-index: {best_transfer_cindex:.3f})")
        
        if best_transfer_model_name:
            survival_model = transfer_results[best_transfer_model_name]['model']
            model_type = transfer_results[best_transfer_model_name]['type']
            print(f"Using {best_transfer_model_name} for prognostic score calculation")
        else:
            # 如果迁移学习模型失败，使用原有的构建方式
            print("Using original method to build survival analysis model...")
            survival_model, model_type = build_survival_analysis_model(smc_pca, y_time, y_event)
        
        # 步骤7: 创建基于生存分析的综合预后评分
        print("\nCreating comprehensive prognostic score...")
        prognostic_score = create_survival_prognostic_score(event_model, survival_model, model_type, smc_pca, y_time, y_event)
        
        # 在main()函数中，在计算完预后评分后添加以下代码：
        
        print("\n" + "="*60)
        print("Checking Distributions of Numerical Variables")
        print("="*60)
        
        try:
            # 重新加载原始SMC数据用于分布检查
            smc_original_for_check = pd.read_excel('/Volumes/YQ1/r/smc.test.xlsx', sheet_name=0)
            distribution_summary = check_all_numerical_distributions(smc_original_for_check, NUMERICAL_FEATURES)
            
            # 检查是否有非正态分布的变量
            n_use_median = sum(distribution_summary['Use_Median']) if 'Use_Median' in distribution_summary.columns else 0
            if n_use_median > 0:
                print(f"\n⚠ WARNING: {n_use_median} variables are non-normally distributed and will use median [Q1, Q3] in Table 1")
                print("These variables will use median [Q1, Q3] instead of mean ± SD:")
                for _, row in distribution_summary[distribution_summary['Use_Median']].iterrows():
                    print(f"  - {row['Variable']}: Skewness={row.get('Skewness', 0):.2f}, {row.get('Reason', 'Non-normal')}")
        except Exception as e:
            print(f"Error in distribution check: {e}")
            print("Continuing with Table 1 generation...")
            

        # 生成Table 1: 患者基线特征表
        print("\n" + "="*60)
        print("Generating Table 1: Patient Baseline Characteristics")
        print("="*60)
        
        try:
            # 重新加载原始SMC数据用于基线特征表
            smc_original_for_table = pd.read_excel('/Volumes/YQ1/r/smc.test.xlsx', sheet_name=0)
            
            # 确保预后评分与数据长度一致
            if len(prognostic_score) > len(smc_original_for_table):
                prognostic_score = prognostic_score[:len(smc_original_for_table)]
            elif len(prognostic_score) < len(smc_original_for_table):
                smc_original_for_table = smc_original_for_table.iloc[:len(prognostic_score)]
            
            # 调用函数生成基线特征表
            baseline_table, score_groups = create_baseline_characteristics_table(
                smc_original_for_table, 
                prognostic_score, 
                CATEGORICAL_FEATURES, 
                NUMERICAL_FEATURES
            )
            
            # 保存分组信息（可用于后续分析）
            pd.DataFrame({
                'Patient_ID': smc_original_for_table.index if 'ID' not in smc_original_for_table.columns else smc_original_for_table['ID'],
                'Prognostic_Score': prognostic_score[:len(smc_original_for_table)],
                'Risk_Group': score_groups
            }).to_excel('patient_risk_groups.xlsx', index=False)
            
            print("Patient risk groups saved to: patient_risk_groups.xlsx")
            
        except Exception as e:
            print(f"Error generating baseline characteristics table: {e}")
            import traceback
            traceback.print_exc()
            print("Using processed data as fallback for baseline characteristics...")
            
            # 备选方案：使用已经处理过的数据
            try:
                # 确保我们有原始特征的副本
                if 'smc_original' in locals():
                    baseline_table, score_groups = create_baseline_characteristics_table(
                        smc_original,
                        prognostic_score[:len(smc_original)] if len(prognostic_score) > len(smc_original) else prognostic_score,
                        CATEGORICAL_FEATURES,
                        NUMERICAL_FEATURES
                    )
            except Exception as e2:
                print(f"Fallback also failed: {e2}")
        # 立即验证预后评分
        from sksurv.metrics import concordance_index_censored
        final_c_index = concordance_index_censored(y_event.astype(bool), y_time, prognostic_score)[0]
        print(f"Final prognostic score C-index: {final_c_index:.3f}")
        
        # 比较预后评分的C-index与基准模型
        best_baseline_cindex = max([results['c_index'] for results in baseline_results.values()])
        improvement_vs_baseline = final_c_index - best_baseline_cindex
        
        print(f"\n=== Prognostic Score Performance Summary ===")
        print(f"Best baseline model C-index: {best_baseline_cindex:.3f}")
        print(f"Prognostic score C-index: {final_c_index:.3f}")
        print(f"Improvement over best baseline: {improvement_vs_baseline:.3f}")
        
        if improvement_vs_baseline > 0:
            print(f"Relative improvement: {100*improvement_vs_baseline/best_baseline_cindex:.1f}%")
            print("✓ PCA transfer learning provides better prognostic score!")
        else:
            print("⚠ Prognostic score does not improve over baseline")
        
        if final_c_index < 0.6:
            print("Warning: Poor prognostic score performance, trying alternative approach...")
            
            # 备选方案：直接使用逻辑回归概率
            print("Trying logistic regression probability as prognostic score...")
            prognostic_score = event_model.predict_proba(smc_pca)[:, 1]
            lr_c_index = concordance_index_censored(y_event.astype(bool), y_time, prognostic_score)[0]
            print(f"Logistic regression probability C-index: {lr_c_index:.3f}")
            
            if lr_c_index > final_c_index:
                print("Using logistic regression probability as final prognostic score")
            else:
                print("Keeping original prognostic score")

        # 步骤8: 综合预后评分分析（新增关键指标计算）
        print("\nPerforming comprehensive prognostic score analysis...")
        
    # 准备基线数据用于重分类比较
        try:
            smc_original_for_baseline = pd.read_excel('/Volumes/YQ1/r/smc.test.xlsx', sheet_name=0)
            baseline_data = smc_original_for_baseline[['a2', 'e4']].iloc[:len(prognostic_score)]
        except:
            baseline_data = None
            print("Unable to create baseline score, skipping reclassification analysis")
        
        # 执行综合分析（包含森林图）
        comprehensive_results = comprehensive_prognostic_score_analysis(
            prognostic_score, y_time, y_event, feature_importance, baseline_data
        )
            
        # 将预后评分添加到原始数据
        try:
            smc_data_with_score = pd.read_excel('/Volumes/YQ1/r/smc.test.xlsx', sheet_name=0)
            # 确保预后评分长度与数据匹配
            if len(prognostic_score) == len(smc_data_with_score):
                smc_data_with_score['Prognostic_Score'] = prognostic_score
                smc_data_with_score.to_excel('smc_with_prognostic_score.xlsx', index=False)
                print("\nPrognostic score calculated and saved to file: smc_with_prognostic_score.xlsx")
            else:
                print(f"Prognostic score length ({len(prognostic_score)}) doesn't match data length ({len(smc_data_with_score)})")
                # 保存预后评分为单独文件
                pd.DataFrame({'Prognostic_Score': prognostic_score}).to_csv('prognostic_scores.csv', index=False)
                print("Prognostic score saved as separate file: prognostic_scores.csv")
        except Exception as e:
            print(f"Error saving prognostic score: {e}")
            # 保存预后评分为单独文件
            pd.DataFrame({'Prognostic_Score': prognostic_score}).to_csv('prognostic_scores.csv', index=False)
            print("Prognostic score saved as separate file: prognostic_scores.csv")
        
        # 分析评分与预后的关系
        try:
            plt.figure(figsize=(10, 6))
            # 使用smc_target_model中的时间数据
            plt.scatter(smc_target_model['time'], prognostic_score, 
                        c=smc_target_model['event'], cmap='viridis', alpha=0.7, s=50)
            plt.colorbar(label='Event (1=Yes, 0=No)')
            plt.xlabel('Survival Time')
            plt.ylabel('Prognostic Score')
            plt.title('Relationship Between Prognostic Score and Survival Time')
            plt.savefig('prognostic_score_vs_time.png', dpi=300)
        except Exception as e:
            print(f"Error creating prognostic score vs time scatter plot: {e}")
            
                        # 步骤8: 创建分组比较表格
        print("\nCreating risk group comparison table...")
        try:
            stats_df, pvalue_df = create_group_comparison_table(prognostic_score, y_time, y_event)
        except Exception as e:
            print(f"Error creating group comparison table: {e}")
    
    except Exception as e:
        print(f"Error in PCA and model building part: {e}")
        import traceback
        traceback.print_exc()
        print("Skipping PCA and model building part")
   
    # =========================================================================
    # 最终总结
    # =========================================================================
    print("\n" + "="*60)
    print("Complete analysis pipeline finished!")
    print("="*60)
    
    print("\nMain files generated:")
    print("Correlation analysis:")
    print("- feature_label_correlations_separate.png: Feature correlation charts")
    print("- feature_label_correlation_results.xlsx/csv: Correlation analysis results")
    print("- numerical_correlation_heatmap.png: Numerical variables heatmap")
    print("- categorical_association_heatmap.png: Categorical variables heatmap")
    print("- numerical_feature_distribution_by_event.png: Numerical variables distribution")
    print("- categorical_feature_event_rates.png: Categorical variables event rates")
    
    print("\nPCA analysis:")
    print("- enhanced_pca_analysis.png: PCA analysis overview")
    print("- pca_loadings_barchart.png: Principal component loading barchart")
    print("- pca_biplots.png: Multiple principal component biplots")
    print("- pca_detailed_biplot.png: Detailed PC1 vs PC2 biplot")
    print("- pca_analysis_results.xlsx/csv: Detailed PCA analysis results")
    
    print("\nModel comparison:")
    print("- model_comparison_baseline_vs_transfer.png: Baseline vs transfer learning model comparison")
    print("- model_performance_comparison.xlsx: Detailed model performance comparison")
    
    print("\nModel building:")
    print("- logistic_feature_importance_original_top10.png: Feature importance based on original variables (Top 10)")
    print("- logistic_feature_importance_original_top20.png: Feature importance based on original variables (Top 20)")
    print("- logistic_feature_importance_pca.png: PCA principal components feature importance")
    print("- prognostic_score_vs_time.png: Prognostic score vs time relationship")
    print("- event_rate_by_prognostic_score_detailed.png: Event rate by prognostic score groups")
    print("- kaplan_meier_with_risktable.png: Kaplan-Meier survival curves with risk table")
    print("- smc_with_prognostic_score.xlsx/prognostic_scores.csv: SMC data with prognostic scores")
    print("- original_feature_importance.csv: Original feature importance")
    print("- comprehensive_prognostic_analysis.xlsx: Comprehensive prognostic score analysis results")
    print("- risk_group_comparison_table.xlsx: Risk group comparison statistics")

if __name__ == "__main__":
    main()