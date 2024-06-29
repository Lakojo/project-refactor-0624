df_shape=df.shape
print('Shape of the',dfname,'DataFrame :', df_shape)

df.drop_duplicates(inplace=True)
drop_duplicates=df_shape[0]-df.shape[0]

df_columns=df.columns

## Create a dataframe with continuous columns 
df_cont = df.select_dtypes(include = ['int64','float64'])
col_cont=df_cont.columns
## Create a dataframe with categorical columns 
df_cat = df.select_dtypes(include =['object'])
col_cat=df_cat.columns
print('Continuous columns are:', col_cont.values )
print('Cartegorical columns are:', col_cat.values)


# %%
df[col_to_numeric]= pd.to_numeric(df[col_to_numeric], errors='coerce')

# %%
## Create a dataframe with continuous columns 
df_cont = df.select_dtypes(include = ['int64','float64'])
col_cont=df_cont.columns
## Create a dataframe with categorical columns 
df_cat = df.select_dtypes(include =['object'])
col_cat=df_cat.columns
print('Continuous columns are:', col_cont.values )
print('Cartegorical columns are:', col_cat.values)

# %%
bin_col= df.columns[df.nunique().values==2]
for col in bin_col: #['Partner','Dependents', 'PhoneService','PaperlessBilling' ,'Churn']:
    df[col]=df[col].replace(to_replace=['no', 'yes'], value=[0, 1])

# %%
# How much unique values per column?
print('Numbers of unique values in each column by', df.shape[0],'numbers of rows: \n', df.nunique())

# %%
# Percentage of missing values in categorical data along with visualization 
if sum(df.dtypes==object)==0:
    print('df_cat ist empty')
else:
    total = df_cat.isnull().sum().sort_values(ascending=False)
    percent = df_cat.isnull().sum()/df_cat.isnull().count().sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    f, ax = plt.subplots(figsize=(15, 6))
    plt.xticks(rotation=90)
    sns.barplot(x=missing_data.index, y=missing_data['Percent'])
    plt.xlabel('df_cat', fontsize=15)
    plt.ylabel('Percent of missing values', fontsize=15)
    plt.title('Percent missing data by feature', fontsize=15)
    print(missing_data,'\n')
    missing_data

# %%
# Percentage of missing values in continuous columns along with visualization
total = df_cont.isnull().sum().sort_values(ascending=False)
percent = df_cont.isnull().sum()/df_cont.isnull().count().sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
f, ax = plt.subplots(figsize=(15, 6))
plt.xticks(rotation=90)
sns.barplot(x=missing_data.index, y=missing_data['Percent'])
plt.xlabel('df_cont', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)
print(missing_data,'\n')
miss5_col=missing_data.index[missing_data['Percent']>0.05]
print('Columns with more the 5% missing values:\n', miss5_col)

# %%
# Drop Features e.g. to prevent data leagake
act=input('Are the any Features which may cause data lekage? Do you want to drop columns?(y/n)')
if act== 'y':
    cols=input('Enter the name(s) of the column(s) as list')    
    df.drop(columns=cols, inplace=True)
    
    # Update continuous and categorical columns
    ## Create a dataframe with continuous columns 
    df_cont = df.select_dtypes(include = ['int64','float64'])
    col_cont=df_cont.columns
    ## Create a dataframe with categorical columns 
    df_cat = df.select_dtypes(include =['object'])
    col_cat=df_cat.columns
    df.head(2)

# %%
# Correlation heatmap
sns.heatmap(df_cont.corr(),annot=True)

# %%
# Boxplot
plt.figure(figsize=(15,8))
sns.boxplot(data=df_cont,orient="h")


# %%
for col in col_cont:
    plt.figure(figsize=(10,4))
    sns.histplot(x=df_cont[col],hue=df[target_name],multiple="dodge")


for col in col_cont:
    plt.figure(figsize=(10,4))
    sns.barplot(y=df_cont[col],hue=df[target_name])
