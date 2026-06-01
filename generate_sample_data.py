import pandas as pd
import numpy as np

def generate():
    np.random.seed(42)
    n_samples = 500  # Generate 500 records
    
    df = pd.DataFrame()
    
    # Continuous fields
    df['F3799'] = np.random.uniform(1000, 500000, n_samples) # Total in
    df['F3800'] = np.random.uniform(1000, 500000, n_samples) # Total out
    df['F3796'] = np.random.randint(1, 50, n_samples)        # Credit count
    df['F3797'] = np.random.randint(1, 50, n_samples)        # Debit count
    
    # Categorical & Flags
    df['F3920'] = np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
    df['F3923'] = np.random.choice([0, 1], n_samples, p=[0.99, 0.01])
    df['F3919'] = np.random.randint(1, 3, n_samples)
    df['F3922'] = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
    
    df['F3891'] = np.random.choice(['salaried', 'housewife', 'selfemployed', 'others', 'student'], n_samples)
    df['F3886'] = np.random.choice(['Savings', 'Current'], n_samples)
    df['F3889'] = np.random.choice(['G365D', 'L180D', 'L90D', 'L30D', 'L14D', 'L7D'], n_samples)
    df['F3890'] = np.random.choice(['R', 'SU', 'U', 'M'], n_samples)
    df['F3893'] = np.random.choice(['RETAIL', 'CORPORATE'], n_samples)
    
    df['F3894'] = np.random.randint(18, 65, n_samples) # Age
    df['F3895'] = np.random.randint(400, 800, n_samples) # Credit Score
    df['F3887'] = np.random.randint(1, 240, n_samples) # Tenure
    
    for f in ['F3900', 'F3901', 'F3902', 'F3905', 'F3912', 'F3913', 'F3915', 'F3916']:
        df[f] = np.random.choice([0, 1], n_samples, p=[0.98, 0.02])
        
    for f in ['F13', 'F14', 'F15', 'F16', 'F19', 'F25']:
        df[f] = np.random.uniform(0.1, 0.9, n_samples)
        
    for f in ['F3856', 'F3859']:
        df[f] = np.random.uniform(0.5, 1.5, n_samples)
        
    for f in ['F3882', 'F3883', 'F2796', 'F3877']:
        df[f] = np.random.uniform(-1.0, 1.0, n_samples)
        
    # Create target (10% mules for testing)
    df['F3924'] = np.random.choice([0, 1], n_samples, p=[0.90, 0.10])
    mule_idx = df['F3924'] == 1
    
    # Inject mule patterns (high velocity, high volume, young/student, new SIM)
    n_mules = mule_idx.sum()
    df.loc[mule_idx, 'F3799'] = np.random.uniform(5000000, 20000000, n_mules)
    df.loc[mule_idx, 'F3800'] = np.random.uniform(5000000, 20000000, n_mules)
    df.loc[mule_idx, 'F3796'] = np.random.randint(300, 800, n_mules)
    df.loc[mule_idx, 'F3891'] = 'student'
    df.loc[mule_idx, 'F3889'] = 'L7D'
    df.loc[mule_idx, 'F3894'] = np.random.randint(18, 22, n_mules)
    df.loc[mule_idx, 'F3895'] = np.random.randint(300, 500, n_mules) # low credit score
    
    df.to_csv('Sample_DataSet.csv', index=False)
    print(f"Successfully generated Sample_DataSet.csv with {n_samples} records and {n_mules} mules.")

if __name__ == "__main__":
    generate()
