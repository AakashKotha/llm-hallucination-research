import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class Phase1EDA:
    def __init__(self):
        self.datasets = {}
        self.model_names = ['Claude_3.5_Sonnet', 'GPT_3.5', 'GPT_4o']
        self.file_paths = [
            '../Data/qna_dataset_Claude3.5Sonnet_final.csv',
            '../Data/qna_dataset_GPT3.5_final.csv', 
            '../Data/qna_dataset_GPT4o_final.csv'
        ]
        
    def load_and_examine_datasets(self):
        """Step 1: Load datasets and perform initial examination"""
        print("="*80)
        print("PHASE 1 EDA: DATA STRUCTURE & QUALITY ASSESSMENT")
        print("="*80)
        
        print("\n🔍 STEP 1: DATASET OVERVIEW")
        print("-"*50)
        
        # Load each dataset
        for model_name, file_path in zip(self.model_names, self.file_paths):
            try:
                df = pd.read_csv(file_path)
                self.datasets[model_name] = df
                print(f"✓ {model_name}: {df.shape[0]} rows × {df.shape[1]} columns")
            except Exception as e:
                print(f"❌ Error loading {model_name}: {e}")
                
        # Dataset dimensions comparison
        print(f"\n📊 Dataset Dimensions Summary:")
        for model_name, df in self.datasets.items():
            print(f"  {model_name:15}: {df.shape}")
            
        # Column consistency check
        print(f"\n📋 Column Consistency Check:")
        if self.datasets:
            reference_cols = list(self.datasets[self.model_names[0]].columns)
            print(f"  Reference columns ({self.model_names[0]}): {len(reference_cols)}")
            
            for model_name, df in self.datasets.items():
                current_cols = list(df.columns)
                if current_cols == reference_cols:
                    print(f"  ✓ {model_name:15}: Columns match reference")
                else:
                    print(f"  ❌ {model_name:15}: Column mismatch detected")
                    missing = set(reference_cols) - set(current_cols)
                    extra = set(current_cols) - set(reference_cols)
                    if missing:
                        print(f"    Missing: {missing}")
                    if extra:
                        print(f"    Extra: {extra}")
        
        return reference_cols
    
    def check_data_quality(self):
        """Step 2: Comprehensive data quality assessment"""
        print("\n🔍 STEP 2: DATA QUALITY ASSESSMENT")
        print("-"*50)
        
        # Missing values analysis
        print("\n📉 Missing Values Analysis:")
        for model_name, df in self.datasets.items():
            print(f"\n  {model_name}:")
            missing_summary = df.isnull().sum()
            missing_pct = (df.isnull().sum() / len(df)) * 100
            
            missing_df = pd.DataFrame({
                'Missing_Count': missing_summary,
                'Missing_Percentage': missing_pct
            })
            missing_df = missing_df[missing_df['Missing_Count'] > 0]
            
            if len(missing_df) == 0:
                print("    ✓ No missing values detected")
            else:
                print("    ❌ Missing values found:")
                print(missing_df.to_string())
        
        # Data type consistency
        print(f"\n📊 Data Types Analysis:")
        for model_name, df in self.datasets.items():
            print(f"\n  {model_name}:")
            print(df.dtypes.to_string())
        
        # Key column value ranges
        print(f"\n📈 Key Numeric Columns Summary:")
        numeric_cols = ['question_length', 'response_length', 'confidence_markers_count', 
                       'uncertainty_markers_count']
        
        for col in numeric_cols:
            print(f"\n  {col}:")
            for model_name, df in self.datasets.items():
                if col in df.columns:
                    stats = df[col].describe()
                    print(f"    {model_name:15}: min={stats['min']:6.1f}, max={stats['max']:8.1f}, "
                          f"mean={stats['mean']:6.1f}, std={stats['std']:6.1f}")
    
    def verify_sampling_balance(self):
        """Step 3: Sampling balance verification"""
        print("\n🔍 STEP 3: SAMPLING BALANCE VERIFICATION")
        print("-"*50)
        
        # Domain distribution check
        print("\n🎯 Domain Distribution Check:")
        domain_summary = []
        
        for model_name, df in self.datasets.items():
            domain_counts = df['domain'].value_counts().sort_index()
            domain_summary.append({
                'Model': model_name,
                **domain_counts.to_dict()
            })
            print(f"\n  {model_name}:")
            print(domain_counts.to_string())
        
        # Create domain balance summary table
        domain_df = pd.DataFrame(domain_summary).set_index('Model')
        print(f"\n📊 Domain Balance Summary:")
        print(domain_df.to_string())
        
        # Check if perfectly balanced
        if domain_df.nunique().nunique() == 1 and domain_df.nunique().iloc[0] == 1:
            print("✓ Perfect domain balance across all models")
        else:
            print("❌ Domain imbalance detected")
        
        # Question type distributions
        categorical_cols = ['question_type', 'question_nature', 'question_style']
        
        for col in categorical_cols:
            print(f"\n📊 {col.replace('_', ' ').title()} Distribution:")
            for model_name, df in self.datasets.items():
                if col in df.columns:
                    counts = df[col].value_counts()
                    pct = (counts / len(df)) * 100
                    print(f"\n  {model_name}:")
                    for category, count in counts.items():
                        print(f"    {category:15}: {count:3d} ({pct[category]:5.1f}%)")
        
        # Identify potential sampling biases
        print(f"\n⚠️  Potential Sampling Biases Identified:")
        
        for model_name, df in self.datasets.items():
            print(f"\n  {model_name}:")
            
            # Check question_type balance
            if 'question_type' in df.columns:
                type_dist = df['question_type'].value_counts(normalize=True) * 100
                if type_dist.max() > 90:
                    dominant_type = type_dist.idxmax()
                    print(f"    ❌ Heavy bias toward {dominant_type} questions ({type_dist.max():.1f}%)")
                
            # Check question_style balance  
            if 'question_style' in df.columns:
                style_dist = df['question_style'].value_counts(normalize=True) * 100
                if style_dist.max() > 90:
                    dominant_style = style_dist.idxmax()
                    print(f"    ❌ Heavy bias toward {dominant_style} questions ({style_dist.max():.1f}%)")
    
    def validate_question_consistency(self):
        """Step 4: Validate that all models answered the same questions"""
        print("\n🔍 STEP 4: QUESTION CONSISTENCY VALIDATION")
        print("-"*50)
        
        # Extract question IDs from each dataset
        question_ids = {}
        for model_name, df in self.datasets.items():
            question_ids[model_name] = set(df['question_id'].unique())
            print(f"  {model_name:15}: {len(question_ids[model_name])} unique questions")
        
        # Check for perfect overlap
        if len(self.datasets) >= 2:
            all_ids = list(question_ids.values())
            intersection = set.intersection(*all_ids)
            union = set.union(*all_ids)
            
            print(f"\n📊 Question ID Overlap Analysis:")
            print(f"  Questions in common: {len(intersection)}")
            print(f"  Total unique questions: {len(union)}")
            print(f"  Overlap percentage: {(len(intersection)/len(union))*100:.1f}%")
            
            if len(intersection) == len(union):
                print("  ✓ Perfect question consistency - all models answered identical questions")
                self.identical_questions = True
            else:
                print("  ❌ Question inconsistency detected")
                self.identical_questions = False
                
                # Show which questions are missing from which models
                for model_name, ids in question_ids.items():
                    missing = union - ids
                    if missing:
                        print(f"  {model_name} missing: {sorted(list(missing))[:10]}...")
        
        # Verify question text consistency (sample check)
        if hasattr(self, 'identical_questions') and self.identical_questions:
            print(f"\n🔍 Question Text Consistency Check (Sample):")
            
            # Take first 5 question IDs and compare question text
            sample_ids = sorted(list(intersection))[:5]
            
            for qid in sample_ids:
                texts = []
                for model_name, df in self.datasets.items():
                    question_row = df[df['question_id'] == qid]
                    if not question_row.empty:
                        texts.append(question_row['question_text'].iloc[0])
                
                if len(set(texts)) == 1:
                    print(f"  ✓ Question {qid}: Identical across all models")
                else:
                    print(f"  ❌ Question {qid}: Text variations detected")
    
    def create_data_quality_visualizations(self):
        """Create visualizations for data quality assessment"""
        print("\n🔍 STEP 5: DATA QUALITY VISUALIZATIONS")
        print("-"*50)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Phase 1 EDA: Data Quality Assessment', fontsize=16, fontweight='bold')
        
        # 1. Dataset sizes comparison
        model_names = list(self.datasets.keys())
        dataset_sizes = [len(df) for df in self.datasets.values()]
        
        axes[0,0].bar(model_names, dataset_sizes, color=['skyblue', 'lightcoral', 'lightgreen'])
        axes[0,0].set_title('Dataset Sizes Comparison')
        axes[0,0].set_ylabel('Number of Responses')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for i, v in enumerate(dataset_sizes):
            axes[0,0].text(i, v + 5, str(v), ha='center', va='bottom', fontweight='bold')
        
        # 2. Domain distribution heatmap
        domain_data = []
        domains = None
        
        for model_name, df in self.datasets.items():
            domain_counts = df['domain'].value_counts().sort_index()
            domain_data.append(domain_counts.values)
            if domains is None:
                domains = domain_counts.index.tolist()
        
        domain_matrix = np.array(domain_data)
        sns.heatmap(domain_matrix, 
                   xticklabels=domains,
                   yticklabels=model_names,
                   annot=True, 
                   fmt='d',
                   cmap='Blues',
                   ax=axes[0,1])
        axes[0,1].set_title('Domain Distribution Across Models')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. Question length distribution
        for model_name, df in self.datasets.items():
            axes[1,0].hist(df['question_length'], alpha=0.6, label=model_name, bins=20)
        
        axes[1,0].set_title('Question Length Distribution')
        axes[1,0].set_xlabel('Question Length (characters)')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].legend()
        
        # 4. Response length distribution
        for model_name, df in self.datasets.items():
            axes[1,1].hist(df['response_length'], alpha=0.6, label=model_name, bins=20)
        
        axes[1,1].set_title('Response Length Distribution')
        axes[1,1].set_xlabel('Response Length (characters)')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].legend()
        
        plt.tight_layout()
        plt.show()
    
    def generate_data_quality_report(self):
        """Generate comprehensive data quality report"""
        print("\n" + "="*80)
        print("PHASE 1 DATA QUALITY REPORT")
        print("="*80)
        
        # Overall assessment
        total_responses = sum(len(df) for df in self.datasets.values())
        avg_responses = total_responses / len(self.datasets) if self.datasets else 0
        
        print(f"\n📊 OVERALL DATASET SUMMARY:")
        print(f"  Total responses across all models: {total_responses:,}")
        print(f"  Average responses per model: {avg_responses:.0f}")
        print(f"  Number of models: {len(self.datasets)}")
        
        # Data quality scores
        print(f"\n🎯 DATA QUALITY ASSESSMENT:")
        
        # Check for perfect balance
        domain_balanced = True
        for df in self.datasets.values():
            domain_counts = df['domain'].value_counts()
            if domain_counts.std() > 0:
                domain_balanced = False
                break
        
        print(f"  Domain Balance: {'✓ Perfect' if domain_balanced else '❌ Imbalanced'}")
        print(f"  Question Consistency: {'✓ Identical' if hasattr(self, 'identical_questions') and self.identical_questions else '❌ Inconsistent'}")
        
        # Missing data assessment
        missing_data = False
        for df in self.datasets.values():
            if df.isnull().sum().sum() > 0:
                missing_data = True
                break
        
        print(f"  Missing Data: {'❌ Present' if missing_data else '✓ None detected'}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS FOR NEXT STEPS:")
        
        if domain_balanced and hasattr(self, 'identical_questions') and self.identical_questions:
            print("  ✓ Excellent data quality - proceed with confidence to Phase 2")
            print("  ✓ Direct model comparisons are valid due to identical question sets")
        
        if not domain_balanced:
            print("  ⚠️  Consider domain weighting in statistical analyses")
            
        # Check for sampling biases
        for model_name, df in self.datasets.items():
            if 'question_type' in df.columns:
                type_dist = df['question_type'].value_counts(normalize=True) * 100
                if type_dist.max() > 90:
                    print(f"  ⚠️  {model_name}: Heavy bias toward one question type - discuss in limitations")
        
        print(f"\n📋 READY FOR PHASE 2: Core Outcome Variables Analysis")
        
    def run_complete_phase1_analysis(self):
        """Run the complete Phase 1 EDA analysis"""
        # Load and examine datasets
        columns = self.load_and_examine_datasets()
        
        # Check data quality
        self.check_data_quality()
        
        # Verify sampling balance
        self.verify_sampling_balance()
        
        # Validate question consistency
        self.validate_question_consistency()
        
        # Create visualizations
        self.create_data_quality_visualizations()
        
        # Generate final report
        self.generate_data_quality_report()
        
        return self.datasets

# Usage example:
if __name__ == "__main__":
    # Initialize and run Phase 1 analysis
    phase1_analyzer = Phase1EDA()
    datasets = phase1_analyzer.run_complete_phase1_analysis()
    
    print("\n" + "="*80)
    print("Phase 1 EDA Complete! Datasets loaded and ready for Phase 2.")
    print("="*80)