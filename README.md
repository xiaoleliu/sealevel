# sealevel

import pandas as pd
import numpy as np

# Load datasets
fl_income = pd.read_csv('/content/FL_income.csv')
#housing
fl_housing = pd.read_csv('/content/FL_housing.csv')
#financial
fl_fin = pd.read_csv('/content/FL_fin.csv')
#poverty
fl_pov = pd.read_csv('/content/FL_pov.csv')
#risk
fl_risk = pd.read_csv('/content/NRI_Table_CensusTracts_FL_short.csv')


# Create a column for census tract identifier in "Income"
fl_income['TRACTFIPS'] = fl_income['geography'].apply(lambda x: x.split('US')[-1])


#convert risk['TRACTFIPS'] to string
fl_risk['TRACTFIPS']=fl_risk['TRACTFIPS'].astype(str)




fl_data = pd.merge(fl_income, fl_risk, on='TRACTFIPS')





# Clean the 'families_median_income' from non-numeric characters and convert to numeric
fl_data = fl_data[fl_data['families_median_income'] != '-']
fl_data = fl_data[~fl_data['families_median_income'].str.contains(r'[@#&$%+-/*]')]
fl_data['families_median_income'] = pd.to_numeric(fl_data['families_median_income'], errors='coerce')
fl_data.dropna(subset=['families_median_income'], inplace=True)  # Drop any rows that still have NaN


# Drop rows with NaN values in 'CFLD_AFREQ'
fl_data['CFLD_AFREQ'] = pd.to_numeric(fl_data['CFLD_AFREQ'], errors='coerce')
fl_data.dropna(subset=['CFLD_AFREQ'], inplace=True)  # Drop any rows that still have NaN


# Define low income as income below a threshold (e.g., 25th percentile)
income_threshold = fl_data['families_median_income'].quantile(0.10)
fl_data['LowIncome'] = fl_data['families_median_income'] < income_threshold


# Focus on flooding risk; assume 'FloodRisk' is a column in risk data
fl_data['HighFloodRisk'] = fl_data['CFLD_AFREQ'] > fl_data['CFLD_AFREQ'].quantile(0.90)


# Intersection between low income and high risk
low_income_and_high_risk_fl = fl_data[(fl_data['LowIncome']) & (fl_data['HighFloodRisk'])]


low_income_and_high_risk_fl.head()  # Display first few rows of the relevant data


low_income_and_high_risk_fl.shape[0]  # Output the number of rows meeting criteria


#rehome solution
#merge all data
#merge all census data
fl = pd.merge(fl_income, fl_housing, on=['geography','geographic_area_name'])
fl = pd.merge(fl, fl_fin, on=['geography','geographic_area_name'])
fl = pd.merge(fl, fl_pov, on=['geography','geographic_area_name'])


fl['TRACTFIPS'] = fl['geography'].apply(lambda x: x.split('US')[-1])


fl = pd.merge(fl, fl_risk, on='TRACTFIPS')




# Drop rows where 'RESL_SCORE' is NaN
fl.dropna(subset=['RESL_SCORE'], inplace=True)


# Calculate the median of 'CFLD_AFREQ' and create a new column based on this threshold
cfl_afreq_median = fl['CFLD_AFREQ'].quantile(0.5)
fl['LowFloodFre_fl'] = fl['CFLD_AFREQ'] < cfl_afreq_median


# Assuming 'RESL_SCORE' needs to be considered as high resilience if greater than its 50th percentile
resilience_threshold = fl['RESL_SCORE'].quantile(0.5)
fl['High_Resilience_fl'] = fl['RESL_SCORE'] > resilience_threshold


#NRI
risk_cap = fl['RISK_SCORE'].quantile(0.25)
fl['LowRisk_fl'] = fl['RISK_SCORE'] < risk_cap


# Filter DataFrame based on low risk and high resilience
low_risk_and_high_resilience_fl = fl[(fl['LowFloodFre_fl']) & (fl['High_Resilience_fl'])&(fl['LowRisk_fl'])]


# Sort the DataFrame by 'RISK_SCORE' in ascending order and 'RESL_SCORE' in descending order
sorted_fl = low_risk_and_high_resilience_fl.sort_values(by=['RISK_SCORE', 'RESL_SCORE'], ascending=[True, False])


# Display the top 10% of sorted entries based on 'TRACTFIPS', 'RISK_SCORE', 'RESL_SCORE'
top_10_percent_fl = sorted_fl.head(int(len(sorted_fl) * 0.1))
top_10_percent_fl = top_10_percent_fl[~top_10_percent_fl['families_median_income'].str.contains('-')]
top_10_percent_fl['families_median_income'] = pd.to_numeric(top_10_percent_fl['families_median_income'], errors='coerce')
top_10_percent_fl.dropna(subset=['families_median_income'], inplace=True)  # Drop any rows that still have NaN
top_10_percent_fl[['TRACTFIPS', 'RISK_SCORE', 'RESL_SCORE']].head()
print(top_10_percent_fl[['TRACTFIPS', 'RISK_SCORE', 'RESL_SCORE']].shape[0])


def find_closest_tracknumber(income):
    # Calculate the absolute difference with every 'income_median' in df2
    differences = np.abs(top_10_percent_fl['families_median_income'] - income)
    # Find the index of the minimum difference
    index_min = differences.idxmin()
    # Return the 'tracknumber' at this index
    return top_10_percent_fl.at[index_min, 'TRACTFIPS']



low_income_and_high_risk_fl['destination'] = low_income_and_high_risk_fl['families_median_income'].apply(find_closest_tracknumber)


final_df_fl = pd.merge(low_income_and_high_risk_fl, top_10_percent_fl, left_on='destination', right_on='TRACTFIPS', suffixes=('_low_income_and_high_risk_fl', '_top_10_percent_fl'))


print(final_df_fl.columns)



def find_closest_tracknumber(income, assigned):
    differences = np.abs(top_10_percent_fl['families_median_income'] - income)
    differences[assigned] = np.inf  # Set differences for already assigned TRACTFIPS to infinity
    index_min = differences.idxmin()
    return index_min


# List to keep track of already assigned TRACTFIPS
assigned_tractfips = []



results = []
for income in low_income_and_high_risk_fl['families_median_income']:
    index_min = find_closest_tracknumber(income, assigned_tractfips)
    if index_min is not None:
        results.append(top_10_percent_fl.at[index_min, 'TRACTFIPS'])
        assigned_tractfips.append(index_min)
    else:
        results.append(None)  # Or handle the case where no unique match is found


low_income_and_high_risk_fl['destination'] = results


# Merge the two DataFrames based on the 'destination'
final_df_fl = pd.merge(low_income_and_high_risk_fl, top_10_percent_fl, left_on='destination', right_on='TRACTFIPS', suffixes=('_low_income_and_high_risk_fl', '_top_10_percent_fl'))


print(final_df_fl.columns)


final_df_fl[['TRACTFIPS_low_income_and_high_risk_fl','families_median_income_low_income_and_high_risk_fl','destination','families_median_income_top_10_percent_fl']].head(25)

fl_fin['TRACTFIPS'] = fl_fin['geography'].apply(lambda x: x.split('US')[-1])
final_df_fl = pd.merge(final_df_fl, fl_fin, left_on='TRACTFIPS_low_income_and_high_risk_fl', right_on='TRACTFIPS')
final_df_fl = final_df_fl.rename(columns = {'median_monthly_costs_occupied_units_x' :'start'})
final_df_fl = pd.merge(final_df_fl, fl_fin, left_on='destination', right_on='TRACTFIPS', suffixes = ('_2', '_des'))
final_df_fl = final_df_fl.rename(columns = {'median_monthly_costs_occupied_units_y' : 'end'})
final_df_fl.head()


final_df_fl = final_df_fl[~final_df_fl['start'].str.contains(r'[@#&$%+-/*]')]
final_df_fl['start'] = pd.to_numeric(final_df_fl['start'], errors='coerce')
final_df_fl.dropna(subset=['start'], inplace=True)  # Drop any rows that still have NaN


final_df_fl = final_df_fl[~final_df_fl['end'].str.contains(r'[@#&$%+-/*]')]
final_df_fl['end'] = pd.to_numeric(final_df_fl['end'], errors='coerce')
final_df_fl.dropna(subset=['end'], inplace=True)  # Drop any rows that still have NaN


final_df_fl['saving'] =  final_df_fl['start'] - final_df_fl['end']
print(final_df_fl['saving'])

# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns


sns.set(style="whitegrid")


fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

# Scatter plot for Low Income and High Risk
sns.scatterplot(data=low_income_and_high_risk_fl, x='families_median_income', y='CFLD_AFREQ', ax=axes[0], color='blue')
axes[0].set_title('Low Income vs. High Flood Risk_FL')
axes[0].set_xlabel('Median Family Income_FL')
axes[0].set_ylabel('Annual Flood Frequency_FL')

# Scatter plot for Top 10 Percent by Resilience and Income
sns.scatterplot(data=top_10_percent_fl, x='families_median_income', y='RESL_SCORE', ax=axes[1], color='green')
axes[1].set_title('Top 10% Resilient Locations by Income_FL')
axes[1].set_xlabel('Median Family Income_FL')
axes[1].set_ylabel('Resilience Score_FL')

# Scatter plot for Savings on Housing Costs
sns.scatterplot(data=final_df_fl, x='start', y='end', ax=axes[2], color='red')
axes[2].set_title('Housing Cost Savings_FL')
axes[2].set_xlabel('Monthly Costs Before_FL')
axes[2].set_ylabel('Monthly Costs After_FL')

plt.tight_layout()


plt.show()

# Load datasets
ga_income = pd.read_csv('/content/GA_income.csv')
#housing
ga_housing = pd.read_csv('/content/GA_housing.csv')
#financial
ga_fin = pd.read_csv('/content/GA_fin.csv')
#poverty
ga_pov = pd.read_csv('/content/GA_pov.csv')
#risk
ga_risk = pd.read_csv('/content/NRI_Table_CensusTracts_GA_short.csv')


# Create a column for census tract identifier in "Income"
ga_income['TRACTFIPS'] = ga_income['geography'].apply(lambda x: x.split('US')[-1])


#convert risk['TRACTFIPS'] to string
ga_risk['TRACTFIPS']=ga_risk['TRACTFIPS'].astype(str)




ga_data = pd.merge(ga_income, ga_risk, on='TRACTFIPS')













# Clean the 'families_median_income' from non-numeric characters and convert to numeric
ga_data = ga_data[ga_data['families_median_income'] != '-']
ga_data = ga_data[~ga_data['families_median_income'].str.contains(r'[@#&$%+-/*]')]
ga_data['families_median_income'] = pd.to_numeric(ga_data['families_median_income'], errors='coerce')
ga_data.dropna(subset=['families_median_income'], inplace=True)  # Drop any rows that still have NaN


# Drop rows with NaN values in 'CFLD_AFREQ'
ga_data['CFLD_AFREQ'] = pd.to_numeric(ga_data['CFLD_AFREQ'], errors='coerce')
ga_data.dropna(subset=['CFLD_AFREQ'], inplace=True)  # Drop any rows that still have NaN


# Define low income as income below a threshold (e.g., 10th percentile)
income_threshold = ga_data['families_median_income'].quantile(0.1)
ga_data['LowIncome'] = ga_data['families_median_income'] < income_threshold


# Focus on flooding risk; assume 'FloodRisk' is a column in risk data
ga_data['HighFloodRisk'] = ga_data['CFLD_AFREQ'] > ga_data['CFLD_AFREQ'].quantile(0.9)


# Intersection between low income and high risk
low_income_and_high_risk_ga = ga_data[(ga_data['LowIncome']) & (ga_data['HighFloodRisk'])]


low_income_and_high_risk_ga.head()  # Display first few rows of the relevant data


low_income_and_high_risk_ga.shape[0]  # Output the number of rows meeting criteria


#rehome solution
#merge all data
#merge all census data
ga = pd.merge(ga_income, ga_housing, on=['geography','geographic_area_name'])
ga = pd.merge(ga, ga_fin, on=['geography','geographic_area_name'])
ga = pd.merge(ga, ga_pov, on=['geography','geographic_area_name'])


ga['TRACTFIPS'] = ga['geography'].apply(lambda x: x.split('US')[-1])


ga = pd.merge(ga, ga_risk, on='TRACTFIPS')



# Drop rows where 'RESL_SCORE' is NaN
ga.dropna(subset=['RESL_SCORE'], inplace=True)


# Calculate the median of 'CFLD_AFREQ' and create a new column based on this threshold
cga_afreq_median = ga['CFLD_AFREQ'].quantile(0.75)
ga['LowFloodFre_ga'] = ga['CFLD_AFREQ'] < cga_afreq_median


# Assuming 'RESL_SCORE' needs to be considered as high resilience if greater than its 50th percentile
resilience_threshold = ga['RESL_SCORE'].quantile(0.5)
ga['High_Resilience_ga'] = ga['RESL_SCORE'] > resilience_threshold


#NRI
risk_cap = ga['RISK_SCORE'].quantile(0.5)
ga['LowRisk_ga'] = ga['RISK_SCORE'] < risk_cap


# Filter DataFrame based on low risk and high resilience
low_risk_and_high_resilience_ga = ga[(ga['LowFloodFre_ga']) & (ga['High_Resilience_ga'])&(ga['LowRisk_ga'])]


# Sort the DataFrame by 'RISK_SCORE' in ascending order and 'RESL_SCORE' in descending order
sorted_ga = low_risk_and_high_resilience_ga.sort_values(by=['RISK_SCORE', 'RESL_SCORE'], ascending=[True, False])


# Display the top 10% of sorted entries based on 'TRACTFIPS', 'RISK_SCORE', 'RESL_SCORE', take all available if there's less than 30 entries
if sorted_ga.shape[0] <= 30:
  perc = 1
else: 
  perc = 0.1
top_10_percent_ga = sorted_ga.head(int(len(sorted_ga) * perc))
top_10_percent_ga = top_10_percent_ga[~top_10_percent_ga['families_median_income'].str.contains('-')]
top_10_percent_ga['families_median_income'] = pd.to_numeric(top_10_percent_ga['families_median_income'], errors='coerce')
top_10_percent_ga.dropna(subset=['families_median_income'], inplace=True)  # Drop any rows that still have NaN
top_10_percent_ga[['TRACTFIPS', 'RISK_SCORE', 'RESL_SCORE']].head()
print(top_10_percent_ga[['TRACTFIPS', 'RISK_SCORE', 'RESL_SCORE']].shape[0])

def find_closest_tracknumber(income):
    # Calculate the absolute difference with every 'income_median' in df2
    differences = np.abs(top_10_percent_ga['families_median_income'] - income)
    # Find the index of the minimum difference
    index_min = differences.idxmin()
    # Return the 'tracknumber' at this index
    return top_10_percent_ga.at[index_min, 'TRACTFIPS']


low_income_and_high_risk_ga['destination'] = low_income_and_high_risk_ga['families_median_income'].apply(find_closest_tracknumber)

final_df_ga = pd.merge(low_income_and_high_risk_ga, top_10_percent_ga, left_on='destination', right_on='TRACTFIPS', suffixes=('_low_income_and_high_risk_ga', '_top_10_percent_ga'))


print(final_df_ga.columns)



def find_closest_tracknumber(income, assigned):
    differences = np.abs(top_10_percent_ga['families_median_income'] - income)
    differences[assigned] = np.inf  # Set differences for already assigned TRACTFIPS to infinity
    index_min = differences.idxmin()
    return index_min



assigned_tractfips = []



results = []
for income in low_income_and_high_risk_ga['families_median_income']:
    index_min = find_closest_tracknumber(income, assigned_tractfips)
    if index_min is not None:
        results.append(top_10_percent_ga.at[index_min, 'TRACTFIPS'])
        assigned_tractfips.append(index_min)
    else:
        results.append(None)  # Or handle the case where no unique match is found


low_income_and_high_risk_ga['destination'] = results


# Merge the two DataFrames based on the 'destination'
final_df_ga = pd.merge(low_income_and_high_risk_ga, top_10_percent_ga, left_on='destination', right_on='TRACTFIPS', suffixes=('_low_income_and_high_risk_ga', '_top_10_percent_ga'))


print(final_df_ga.columns)


final_df_ga[['TRACTFIPS_low_income_and_high_risk_ga','families_median_income_low_income_and_high_risk_ga','destination','families_median_income_top_10_percent_ga']].head(25)

ga_fin['TRACTFIPS'] = ga_fin['geography'].apply(lambda x: x.split('US')[-1])
final_df_ga = pd.merge(final_df_ga, ga_fin, left_on='TRACTFIPS_low_income_and_high_risk_ga', right_on='TRACTFIPS')
final_df_ga = final_df_ga.rename(columns = {'median_monthly_costs_occupied_units_x' :'start'})
final_df_ga = pd.merge(final_df_ga, ga_fin, left_on='destination', right_on='TRACTFIPS', suffixes = ('_2', '_des'))
final_df_ga = final_df_ga.rename(columns = {'median_monthly_costs_occupied_units_y' : 'end'})
final_df_ga.head()


final_df_ga = final_df_ga[~final_df_ga['start'].str.contains(r'[@#&$%+-/*]')]
final_df_ga['start'] = pd.to_numeric(final_df_ga['start'], errors='coerce')
final_df_ga.dropna(subset=['start'], inplace=True)  # Drop any rows that still have NaN


final_df_ga = final_df_ga[~final_df_ga['end'].str.contains(r'[@#&$%+-/*]')]
final_df_ga['end'] = pd.to_numeric(final_df_ga['end'], errors='coerce')
final_df_ga.dropna(subset=['end'], inplace=True)  # Drop any rows that still have NaN


final_df_ga['saving'] =  final_df_ga['start'] - final_df_ga['end']
print(final_df_ga['saving'])

# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns


sns.set(style="whitegrid")


fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

# Scatter plot for Low Income and High Risk
sns.scatterplot(data=low_income_and_high_risk_ga, x='families_median_income', y='CFLD_AFREQ', ax=axes[0], color='blue')
axes[0].set_title('Low Income vs. High Flood Risk_GA')
axes[0].set_xlabel('Median Family Income_GA')
axes[0].set_ylabel('Annual Flood Frequency_GA')

# Scatter plot for Top 10 Percent by Resilience and Income
sns.scatterplot(data=top_10_percent_ga, x='families_median_income', y='RESL_SCORE', ax=axes[1], color='green')
axes[1].set_title('Top 10% Resilient Locations by Income_GA')
axes[1].set_xlabel('Median Family Income_GA')
axes[1].set_ylabel('Resilience Score_GA')

# Scatter plot for Savings on Housing Costs
sns.scatterplot(data=final_df_ga, x='start', y='end', ax=axes[2], color='red')
axes[2].set_title('Housing Cost Savings_GA')
axes[2].set_xlabel('Monthly Costs Before_GA')
axes[2].set_ylabel('Monthly Costs After_GA')


plt.tight_layout()


plt.show()



# Load datasets
tx_income = pd.read_csv('/content/TX_income.csv')
#housing
tx_housing = pd.read_csv('/content/TX_housing.csv')
#financial
tx_fin = pd.read_csv('/content/TX_fin.csv')
#poverty
tx_pov = pd.read_csv('/content/TX_pov.csv')
#risk
tx_risk = pd.read_csv('/content/NRI_Table_CensusTracts_TX_short.csv')


# Create a column for census tract identifier in "Income"
tx_income['TRACTFIPS'] = tx_income['geography'].apply(lambda x: x.split('US')[-1])


#convert risk['TRACTFIPS'] to string
tx_risk['TRACTFIPS']=tx_risk['TRACTFIPS'].astype(str)




tx_data = pd.merge(tx_income, tx_risk, on='TRACTFIPS')











# Clean the 'families_median_income' from non-numeric characters and convert to numeric
tx_data = tx_data[tx_data['families_median_income'] != '-']
tx_data = tx_data[~tx_data['families_median_income'].str.contains(r'[@#&$%+-/*]')]
tx_data['families_median_income'] = pd.to_numeric(tx_data['families_median_income'], errors='coerce')
tx_data.dropna(subset=['families_median_income'], inplace=True)  # Drop any rows that still have NaN


# Drop rows with NaN values in 'CFLD_AFREQ'
tx_data['CFLD_AFREQ'] = pd.to_numeric(tx_data['CFLD_AFREQ'], errors='coerce')
tx_data.dropna(subset=['CFLD_AFREQ'], inplace=True)  # Drop any rows that still have NaN


# Define low income as income below a threshold (e.g., 10th percentile)
income_threshold = tx_data['families_median_income'].quantile(0.10)
tx_data['LowIncome'] = tx_data['families_median_income'] < income_threshold


# Focus on txooding risk; assume 'FloodRisk' is a column in risk data
tx_data['HighFloodRisk'] = tx_data['CFLD_AFREQ'] > tx_data['CFLD_AFREQ'].quantile(0.90)


# Intersection between low income and high risk
low_income_and_high_risk_tx = tx_data[(tx_data['LowIncome']) & (tx_data['HighFloodRisk'])]


low_income_and_high_risk_tx.head()  # Display first few rows of the relevant data


low_income_and_high_risk_tx.shape[0]  # Output the number of rows meeting criteria


#rehome solution

#merge all census data
tx = pd.merge(tx_income, tx_housing, on=['geography','geographic_area_name'])
tx = pd.merge(tx, tx_fin, on=['geography','geographic_area_name'])
tx = pd.merge(tx, tx_pov, on=['geography','geographic_area_name'])


tx['TRACTFIPS'] = tx['geography'].apply(lambda x: x.split('US')[-1])


tx = pd.merge(tx, tx_risk, on='TRACTFIPS')



# Drop rows where 'RESL_SCORE' is NaN
tx.dropna(subset=['RESL_SCORE'], inplace=True)


# Calculate the median of 'CtxD_AFREQ' and create a new column based on this threshold
ctx_afreq_median = tx['CFLD_AFREQ'].quantile(0.75)
tx['LowFloodFre_tx'] = tx['CFLD_AFREQ'] < ctx_afreq_median


# Assuming 'RESL_SCORE' needs to be considered as high resilience if greater than its 75th percentile
resilience_threshold = tx['RESL_SCORE'].quantile(0.5)
tx['High_Resilience_tx'] = tx['RESL_SCORE'] > resilience_threshold


#NRI
risk_cap = tx['RISK_SCORE'].quantile(0.25)
tx['LowRisk_tx'] = tx['RISK_SCORE'] < risk_cap


# Filter DataFrame based on low risk and high resilience
low_risk_and_high_resilience_tx = tx[(tx['LowFloodFre_tx']) & (tx['High_Resilience_tx'])&(tx['LowRisk_tx'])]


# Sort the DataFrame by 'RISK_SCORE' in ascending order and 'RESL_SCORE' in descending order
sorted_tx = low_risk_and_high_resilience_tx.sort_values(by=['RISK_SCORE', 'RESL_SCORE'], ascending=[True, False])


# Display the top 10% of sorted entries based on 'TRACTFIPS', 'RISK_SCORE', 'RESL_SCORE', if there're less than 30 available, then analyze all of them
if sorted_tx.shape[0] <= 30:
  perc = 1
else: 
  perc = 0.1
top_10_percent_tx = sorted_tx.head(int(len(sorted_tx) * perc))
top_10_percent_tx = top_10_percent_tx[~top_10_percent_tx['families_median_income'].str.contains('-')]
top_10_percent_tx['families_median_income'] = pd.to_numeric(top_10_percent_tx['families_median_income'], errors='coerce')
top_10_percent_tx.dropna(subset=['families_median_income'], inplace=True)  # Drop any rows that still have NaN
top_10_percent_tx[['TRACTFIPS', 'RISK_SCORE', 'RESL_SCORE']].head()
print(top_10_percent_tx[['TRACTFIPS', 'RISK_SCORE', 'RESL_SCORE']].shape[0])


def find_closest_tracknumber(income):
    # Calculate the absolute difference with every 'income_median' in df2
    differences = np.abs(top_10_percent_tx['families_median_income'] - income)
    # Find the index of the minimum difference
    index_min = differences.idxmin()
    # Return the 'tracknumber' at this index
    return top_10_percent_tx.at[index_min, 'TRACTFIPS']



low_income_and_high_risk_tx['destination'] = low_income_and_high_risk_tx['families_median_income'].apply(find_closest_tracknumber)

final_df_tx = pd.merge(low_income_and_high_risk_tx, top_10_percent_tx, left_on='destination', right_on='TRACTFIPS', suffixes=('_low_income_and_high_risk_tx', '_top_10_percent_tx'))


print(final_df_tx.columns)



def find_closest_tracknumber(income, assigned):
    differences = np.abs(top_10_percent_tx['families_median_income'] - income)
    differences[assigned] = np.inf  # Set differences for already assigned TRACTFIPS to infinity
    index_min = differences.idxmin()
    return index_min


assigned_tractfips = []


results = []
for income in low_income_and_high_risk_tx['families_median_income']:
    index_min = find_closest_tracknumber(income, assigned_tractfips)
    if index_min is not None:
        results.append(top_10_percent_tx.at[index_min, 'TRACTFIPS'])
        assigned_tractfips.append(index_min)
    else:
        results.append(None)  # Or handle the case where no unique match is found


low_income_and_high_risk_tx['destination'] = results


# Merge the two DataFrames based on the 'destination'
final_df_tx = pd.merge(low_income_and_high_risk_tx, top_10_percent_tx, left_on='destination', right_on='TRACTFIPS', suffixes=('_low_income_and_high_risk_tx', '_top_10_percent_tx'))


print(final_df_tx.columns)


final_df_tx[['TRACTFIPS_low_income_and_high_risk_tx','families_median_income_low_income_and_high_risk_tx','destination','families_median_income_top_10_percent_tx']].head(25)

tx_fin['TRACTFIPS'] = tx_fin['geography'].apply(lambda x: x.split('US')[-1])
final_df_tx = pd.merge(final_df_tx, tx_fin, left_on='TRACTFIPS_low_income_and_high_risk_tx', right_on='TRACTFIPS')
final_df_tx = final_df_tx.rename(columns = {'median_monthly_costs_occupied_units_x' :'start'})
final_df_tx = pd.merge(final_df_tx, tx_fin, left_on='destination', right_on='TRACTFIPS', suffixes = ('_2', '_des'))
final_df_tx = final_df_tx.rename(columns = {'median_monthly_costs_occupied_units_y' : 'end'})
final_df_tx.head()


final_df_tx = final_df_tx[~final_df_tx['start'].str.contains(r'[@#&$%+-/*]')]
final_df_tx['start'] = pd.to_numeric(final_df_tx['start'], errors='coerce')
final_df_tx.dropna(subset=['start'], inplace=True)  # Drop any rows that still have NaN


final_df_tx = final_df_tx[~final_df_tx['end'].str.contains(r'[@#&$%+-/*]')]
final_df_tx['end'] = pd.to_numeric(final_df_tx['end'], errors='coerce')
final_df_tx.dropna(subset=['end'], inplace=True)  # Drop any rows that still have NaN


final_df_tx['saving'] =  final_df_tx['start'] - final_df_tx['end']
print(final_df_tx['saving'])

# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns


sns.set(style="whitegrid")

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

# Scatter plot for Low Income and High Risk
sns.scatterplot(data=low_income_and_high_risk_tx, x='families_median_income', y='CFLD_AFREQ', ax=axes[0], color='blue')
axes[0].set_title('Low Income vs. High Flood Risk_TX')
axes[0].set_xlabel('Median Family Income_TX')
axes[0].set_ylabel('Annual Flood Frequency_TX')

# Scatter plot for Top 10 Percent by Resilience and Income
sns.scatterplot(data=top_10_percent_tx, x='families_median_income', y='RESL_SCORE', ax=axes[1], color='green')
axes[1].set_title('Top 10% Resilient Locations by Income_TX')
axes[1].set_xlabel('Median Family Income_TX')
axes[1].set_ylabel('Resilience Score_TX')

# Scatter plot for Savings on Housing Costs
sns.scatterplot(data=final_df_tx, x='start', y='end', ax=axes[2], color='red')
axes[2].set_title('Housing Cost Savings_TX')
axes[2].set_xlabel('Monthly Costs Before_TX')
axes[2].set_ylabel('Monthly Costs After_TX')

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plots
plt.show()
