import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from scipy.interpolate import BSpline, splrep, splev
import warnings
warnings.filterwarnings('ignore')

# Assuming you have combined_df from previous analysis
# combined_df['hallucination_present'] = combined_df['hallucination_present'].astype(int)

def create_alternative_features(df):
    """Create alternative representations of question length"""
    
    df_alt = df.copy()
    
    # 1. Polynomial terms
    df_alt['question_length_squared'] = df_alt['question_length'] ** 2
    df_alt['question_length_cubed'] = df_alt['question_length'] ** 3
    
    # 2. Categorical length (Short/Medium/Long)
    df_alt['length_category'] = pd.qcut(df_alt['question_length'], 
                                       q=3, labels=['Short', 'Medium', 'Long'])
    
    # 3. More granular categorical (5 groups)
    df_alt['length_quintile'] = pd.qcut(df_alt['question_length'], 
                                       q=5, labels=['Very_Short', 'Short', 'Medium', 'Long', 'Very_Long'])
    
    # 4. Log transformation (common for skewed data)
    df_alt['log_question_length'] = np.log(df_alt['question_length'] + 1)  # +1 to handle any zeros
    
    # 5. Standardized length for interactions
    df_alt['question_length_std'] = (df_alt['question_length'] - df_alt['question_length'].mean()) / df_alt['question_length'].std()
    
    print("Alternative Features Created:")
    print(f"Original length range: {df_alt['question_length'].min():.1f} - {df_alt['question_length'].max():.1f}")
    print(f"Length categories: {df_alt['length_category'].value_counts()}")
    print(f"Length quintiles: {df_alt['length_quintile'].value_counts()}")
    
    return df_alt

def test_polynomial_models(df):
    """Test polynomial terms: question_length + question_length²"""
    
    print("\n" + "="*60)
    print("1. POLYNOMIAL MODELS ANALYSIS")
    print("="*60)
    
    df_clean = df.dropna(subset=['question_length', 'hallucination_present', 'domain']).copy()
    df_clean['hallucination_present'] = df_clean['hallucination_present'].astype(int)
    
    # Model specifications with polynomial terms
    polynomial_models = {
        'Linear': 'hallucination_present ~ question_length + C(domain)',
        'Quadratic': 'hallucination_present ~ question_length + question_length_squared + C(domain)',
        'Cubic': 'hallucination_present ~ question_length + question_length_squared + question_length_cubed + C(domain)',
        'Log': 'hallucination_present ~ log_question_length + C(domain)'
    }
    
    results = {}
    
    for name, formula in polynomial_models.items():
        print(f"\n{'-'*30}")
        print(f"Running {name} Model")
        print(f"{'-'*30}")
        
        try:
            model = smf.logit(formula, data=df_clean).fit(disp=0)
            results[name] = model
            
            print(f"AIC: {model.aic:.2f}")
            print(f"BIC: {model.bic:.2f}")
            print(f"Pseudo R²: {model.prsquared:.4f}")
            
            # Length-related coefficients
            length_params = [p for p in model.params.index if 'question_length' in p or 'log_question' in p]
            print(f"\nLength-related coefficients:")
            for param in length_params:
                coef = model.params[param]
                pval = model.pvalues[param]
                print(f"  {param}: {coef:.6f} (p = {pval:.4f})")
        
        except Exception as e:
            print(f"Error fitting {name}: {e}")
    
    # Model comparison
    if results:
        print(f"\n{'-'*30}")
        print("Polynomial Model Comparison")
        print(f"{'-'*30}")
        
        comparison_df = pd.DataFrame({
            'Model': list(results.keys()),
            'AIC': [model.aic for model in results.values()],
            'BIC': [model.bic for model in results.values()],
            'Pseudo_R2': [model.prsquared for model in results.values()],
        })
        
        comparison_df['AIC_Rank'] = comparison_df['AIC'].rank()
        comparison_df['BIC_Rank'] = comparison_df['BIC'].rank()
        
        print(comparison_df.round(4))
        
        best_aic = comparison_df.loc[comparison_df['AIC'].idxmin(), 'Model']
        best_bic = comparison_df.loc[comparison_df['BIC'].idxmin(), 'Model']
        
        print(f"\nBest polynomial model by AIC: {best_aic}")
        print(f"Best polynomial model by BIC: {best_bic}")
    
    return results

def test_categorical_models(df):
    """Test categorical length: Short/Medium/Long groups"""
    
    print("\n" + "="*60)
    print("2. CATEGORICAL LENGTH MODELS")
    print("="*60)
    
    df_clean = df.dropna(subset=['length_category', 'hallucination_present', 'domain']).copy()
    df_clean['hallucination_present'] = df_clean['hallucination_present'].astype(int)
    
    # Descriptive analysis first
    print("Hallucination rates by length category:")
    category_stats = df_clean.groupby('length_category')['hallucination_present'].agg(['count', 'sum', 'mean']).round(4)
    category_stats.columns = ['Count', 'Hallucinations', 'Rate']
    print(category_stats)
    
    # Chi-square test
    contingency = pd.crosstab(df_clean['length_category'], df_clean['hallucination_present'])
    chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
    
    print(f"\nChi-square test for length category independence:")
    print(f"Chi-square: {chi2:.4f}, p-value: {p_val:.4f}")
    print(f"Significant: {'Yes' if p_val < 0.05 else 'No'}")
    
    # Logistic regression models
    categorical_models = {
        '3_Categories': 'hallucination_present ~ C(length_category) + C(domain)',
        '5_Categories': 'hallucination_present ~ C(length_quintile) + C(domain)',
        '3_Cat_with_Model': 'hallucination_present ~ C(length_category) + C(domain) + C(model)',
    }
    
    results = {}
    
    for name, formula in categorical_models.items():
        print(f"\n{'-'*30}")
        print(f"Running {name}")
        print(f"{'-'*30}")
        
        try:
            model = smf.logit(formula, data=df_clean).fit(disp=0)
            results[name] = model
            
            print(f"AIC: {model.aic:.2f}")
            print(f"BIC: {model.bic:.2f}")
            print(f"Pseudo R²: {model.prsquared:.4f}")
            
            # Length category coefficients
            length_params = [p for p in model.params.index if 'length_category' in p or 'length_quintile' in p]
            if length_params:
                print(f"\nLength category coefficients:")
                for param in length_params:
                    coef = model.params[param]
                    pval = model.pvalues[param]
                    odds_ratio = np.exp(coef)
                    print(f"  {param}: OR = {odds_ratio:.3f} (p = {pval:.4f})")
        
        except Exception as e:
            print(f"Error fitting {name}: {e}")
    
    # Visualization
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    category_stats['Rate'].plot(kind='bar')
    plt.title('Hallucination Rate by Length Category')
    plt.ylabel('Hallucination Rate')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    sns.heatmap(contingency, annot=True, fmt='d', cmap='Reds')
    plt.title('Length Category vs Hallucination')
    
    # Quintile analysis
    quintile_stats = df_clean.groupby('length_quintile')['hallucination_present'].mean()
    plt.subplot(2, 2, 3)
    quintile_stats.plot(kind='bar')
    plt.title('Hallucination Rate by Length Quintile')
    plt.ylabel('Hallucination Rate')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Length distribution by category
    plt.subplot(2, 2, 4)
    for cat in df_clean['length_category'].unique():
        cat_data = df_clean[df_clean['length_category'] == cat]['question_length']
        plt.hist(cat_data, alpha=0.6, label=cat, bins=20)
    plt.xlabel('Question Length')
    plt.ylabel('Frequency')
    plt.title('Length Distribution by Category')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return results, category_stats

def test_interaction_models(df):
    """Test interaction terms: Length × Domain"""
    
    print("\n" + "="*60)
    print("3. INTERACTION MODELS ANALYSIS")
    print("="*60)
    
    df_clean = df.dropna(subset=['question_length', 'hallucination_present', 'domain', 'model']).copy()
    df_clean['hallucination_present'] = df_clean['hallucination_present'].astype(int)
    
    # Interaction models
    interaction_models = {
        'Length_x_Domain': 'hallucination_present ~ question_length * C(domain)',
        'Length_x_Model': 'hallucination_present ~ question_length * C(model)',
        'Category_x_Domain': 'hallucination_present ~ C(length_category) * C(domain)',
        'Full_Interaction': 'hallucination_present ~ question_length * C(domain) + question_length * C(model)',
    }
    
    results = {}
    
    for name, formula in interaction_models.items():
        print(f"\n{'-'*30}")
        print(f"Running {name}")
        print(f"{'-'*30}")
        
        try:
            model = smf.logit(formula, data=df_clean).fit(disp=0)
            results[name] = model
            
            print(f"AIC: {model.aic:.2f}")
            print(f"BIC: {model.bic:.2f}")
            print(f"Pseudo R²: {model.prsquared:.4f}")
            
            # Main length effect
            if 'question_length' in model.params.index:
                coef = model.params['question_length']
                pval = model.pvalues['question_length']
                print(f"Main length effect: {coef:.6f} (p = {pval:.4f})")
            
            # Significant interaction terms
            interaction_params = [p for p in model.params.index if ':' in p and ('question_length' in p or 'length_category' in p)]
            if interaction_params:
                sig_interactions = []
                for param in interaction_params:
                    pval = model.pvalues[param]
                    if pval < 0.05:
                        coef = model.params[param]
                        sig_interactions.append(f"{param}: {coef:.6f} (p = {pval:.4f})")
                
                if sig_interactions:
                    print(f"Significant interactions:")
                    for interaction in sig_interactions:
                        print(f"  {interaction}")
                else:
                    print("No significant interaction terms")
        
        except Exception as e:
            print(f"Error fitting {name}: {e}")
    
    # Visualize interactions
    plt.figure(figsize=(16, 10))
    
    # Length effect by domain
    plt.subplot(2, 3, 1)
    top_domains = df_clean['domain'].value_counts().head(4).index
    for domain in top_domains:
        domain_data = df_clean[df_clean['domain'] == domain]
        if len(domain_data) > 20:
            # Bin lengths and calculate rates
            domain_data['length_bin'] = pd.qcut(domain_data['question_length'], q=3, duplicates='drop')
            binned_rates = domain_data.groupby('length_bin')['hallucination_present'].mean()
            bin_centers = [interval.mid for interval in binned_rates.index]
            plt.plot(bin_centers, binned_rates.values, marker='o', label=domain)
    plt.xlabel('Question Length (binned)')
    plt.ylabel('Hallucination Rate')
    plt.title('Length Effect by Domain')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Length effect by model
    plt.subplot(2, 3, 2)
    for model_name in df_clean['model'].unique():
        model_data = df_clean[df_clean['model'] == model_name]
        model_data['length_bin'] = pd.qcut(model_data['question_length'], q=3, duplicates='drop')
        binned_rates = model_data.groupby('length_bin')['hallucination_present'].mean()
        bin_centers = [interval.mid for interval in binned_rates.index]
        plt.plot(bin_centers, binned_rates.values, marker='o', label=model_name)
    plt.xlabel('Question Length (binned)')
    plt.ylabel('Hallucination Rate')
    plt.title('Length Effect by Model')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Category by domain heatmap
    plt.subplot(2, 3, 3)
    pivot_data = df_clean[df_clean['domain'].isin(top_domains)].pivot_table(
        values='hallucination_present', 
        index='domain', 
        columns='length_category', 
        aggfunc='mean'
    )
    sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='Reds')
    plt.title('Hallucination Rate: Domain × Length Category')
    
    # Category by model heatmap
    plt.subplot(2, 3, 4)
    pivot_model = df_clean.pivot_table(
        values='hallucination_present', 
        index='model', 
        columns='length_category', 
        aggfunc='mean'
    )
    sns.heatmap(pivot_model, annot=True, fmt='.3f', cmap='Blues')
    plt.title('Hallucination Rate: Model × Length Category')
    
    # Domain-specific length distributions
    plt.subplot(2, 3, 5)
    for domain in top_domains:
        domain_lengths = df_clean[df_clean['domain'] == domain]['question_length']
        plt.hist(domain_lengths, alpha=0.6, label=domain, bins=15)
    plt.xlabel('Question Length')
    plt.ylabel('Frequency')
    plt.title('Length Distribution by Domain')
    plt.legend()
    
    # Model-specific length distributions
    plt.subplot(2, 3, 6)
    for model_name in df_clean['model'].unique():
        model_lengths = df_clean[df_clean['model'] == model_name]['question_length']
        plt.hist(model_lengths, alpha=0.6, label=model_name, bins=15)
    plt.xlabel('Question Length')
    plt.ylabel('Frequency')
    plt.title('Length Distribution by Model')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return results

def spline_analysis(df):
    """Semi-parametric approach using splines (advanced)"""
    
    print("\n" + "="*60)
    print("4. SPLINE ANALYSIS (SEMI-PARAMETRIC)")
    print("="*60)
    
    df_clean = df.dropna(subset=['question_length', 'hallucination_present']).copy()
    df_clean['hallucination_present'] = df_clean['hallucination_present'].astype(int)
    
    # Simple spline visualization
    lengths = df_clean['question_length'].values
    halluc_rates = df_clean['hallucination_present'].values
    
    # Bin the data for smoothing
    length_bins = np.linspace(lengths.min(), lengths.max(), 20)
    bin_centers = []
    bin_rates = []
    
    for i in range(len(length_bins)-1):
        mask = (lengths >= length_bins[i]) & (lengths < length_bins[i+1])
        if mask.sum() > 5:  # Only if enough data points
            bin_centers.append((length_bins[i] + length_bins[i+1]) / 2)
            bin_rates.append(halluc_rates[mask].mean())
    
    # Plot smooth trend
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(lengths, halluc_rates, alpha=0.3, s=10)
    if len(bin_centers) > 3:
        # Smooth curve through bin centers
        from scipy.interpolate import UnivariateSpline
        spline = UnivariateSpline(bin_centers, bin_rates, s=0.1)
        x_smooth = np.linspace(min(bin_centers), max(bin_centers), 100)
        y_smooth = spline(x_smooth)
        plt.plot(x_smooth, y_smooth, 'r-', linewidth=2, label='Spline Smooth')
    
    plt.scatter(bin_centers, bin_rates, color='red', s=50, label='Binned Rates')
    plt.xlabel('Question Length')
    plt.ylabel('Hallucination Rate')
    plt.title('Spline Smoothing of Length-Hallucination Relationship')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Residual plot
    plt.subplot(1, 2, 2)
    if len(bin_centers) > 3:
        predicted = spline(bin_centers)
        residuals = np.array(bin_rates) - predicted
        plt.scatter(bin_centers, residuals)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Question Length')
        plt.ylabel('Residuals')
        plt.title('Spline Model Residuals')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return bin_centers, bin_rates

def comprehensive_model_comparison(all_results):
    """Compare all alternative approaches"""
    
    print("\n" + "="*60)
    print("5. COMPREHENSIVE MODEL COMPARISON")
    print("="*60)
    
    # Flatten all results
    all_models = {}
    for category, models in all_results.items():
        if isinstance(models, dict):
            for name, model in models.items():
                all_models[f"{category}_{name}"] = model
    
    if not all_models:
        print("No models to compare")
        return
    
    # Create comparison dataframe
    comparison_data = []
    for name, model in all_models.items():
        if hasattr(model, 'aic'):  # Check if it's a fitted model
            comparison_data.append({
                'Model': name,
                'AIC': model.aic,
                'BIC': model.bic,
                'Pseudo_R2': model.prsquared,
                'Log_Likelihood': model.llf
            })
    
    if not comparison_data:
        print("No valid models to compare")
        return
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Add rankings
    comparison_df['AIC_Rank'] = comparison_df['AIC'].rank()
    comparison_df['BIC_Rank'] = comparison_df['BIC'].rank()
    comparison_df['R2_Rank'] = comparison_df['Pseudo_R2'].rank(ascending=False)
    
    # Sort by AIC
    comparison_df = comparison_df.sort_values('AIC')
    
    print("Complete Model Comparison:")
    print(comparison_df.round(4))
    
    # Best models
    best_aic = comparison_df.iloc[0]['Model']
    best_bic = comparison_df.loc[comparison_df['BIC'].idxmin(), 'Model']
    best_r2 = comparison_df.loc[comparison_df['Pseudo_R2'].idxmax(), 'Model']
    
    print(f"\n🏆 BEST MODELS:")
    print(f"Best AIC: {best_aic}")
    print(f"Best BIC: {best_bic}")
    print(f"Best Pseudo R²: {best_r2}")
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.barh(range(len(comparison_df)), comparison_df['AIC'])
    plt.yticks(range(len(comparison_df)), comparison_df['Model'], rotation=0, fontsize=8)
    plt.xlabel('AIC (lower is better)')
    plt.title('Model Comparison by AIC')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.barh(range(len(comparison_df)), comparison_df['BIC'])
    plt.yticks(range(len(comparison_df)), comparison_df['Model'], rotation=0, fontsize=8)
    plt.xlabel('BIC (lower is better)')
    plt.title('Model Comparison by BIC')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.barh(range(len(comparison_df)), comparison_df['Pseudo_R2'])
    plt.yticks(range(len(comparison_df)), comparison_df['Model'], rotation=0, fontsize=8)
    plt.xlabel('Pseudo R² (higher is better)')
    plt.title('Model Comparison by Pseudo R²')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return comparison_df

def main_alternative_analysis(df):
    """Run complete alternative analysis"""
    
    print("ALTERNATIVE APPROACHES FOR RQ3 ANALYSIS")
    print("=" * 70)
    print("Testing: Polynomial, Categorical, Interactions, and Splines")
    print("=" * 70)
    
    # Create alternative features
    df_alt = create_alternative_features(df)
    
    # Run all analyses
    results = {}
    
    # 1. Polynomial models
    results['Polynomial'] = test_polynomial_models(df_alt)
    
    # 2. Categorical models
    results['Categorical'], category_stats = test_categorical_models(df_alt)
    
    # 3. Interaction models
    results['Interaction'] = test_interaction_models(df_alt)
    
    # 4. Spline analysis
    spline_results = spline_analysis(df_alt)
    
    # 5. Comprehensive comparison
    comparison = comprehensive_model_comparison(results)
    
    # Final summary
    print("\n" + "=" * 70)
    print("ALTERNATIVE APPROACHES SUMMARY")
    print("=" * 70)
    
    print("Key Findings:")
    print("1. Polynomial vs Linear: [See model comparison above]")
    print("2. Categorical Analysis:", category_stats['Rate'].to_dict() if 'category_stats' in locals() else "Check output above")
    print("3. Interaction Effects: [Check significant interactions above]")
    print("4. Non-parametric Trends: [See spline plots]")
    
    print(f"\n📊 IMPLICATIONS FOR RQ3:")
    print(f"These alternative approaches test whether non-linear relationships")
    print(f"exist that simple logistic regression might miss.")
    
    return results, comparison

# Run the alternative analysis
# Assuming you have combined_df with the features created
# results, comparison = main_alternative_analysis(combined_df)