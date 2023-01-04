#!/usr/bin/env python
# coding: utf-8

# In[88]:


#====== Importing the required Library===============
import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from  sklearn.preprocessing import minmax_scale
import random
#====================================================
# Import the data set into workspace:
usedcar = pd.read_csv("D:\Learn\CarDekho_SPPrediction\car_price.csv")    


# In[89]:


# Data preparation and Cleaning:
print(usedcar.info(verbose=True) )


# In[90]:


null_percentage = usedcar.isnull().sum()/usedcar.shape[0]*100
null_percentage


# In[91]:


usedcar = usedcar.drop(columns='Unnamed: 0')
usedcar.head()


# In[92]:


for i in usedcar.index:
    if usedcar['car_prices_in_rupee'].iloc[i].__contains__(' Lakh'):
        usedcar['car_prices_in_rupee'].iloc[i] = usedcar['car_prices_in_rupee'].iloc[i].replace(' Lakh','')
        temp_float = float(usedcar['car_prices_in_rupee'].iloc[i])
        covert_to_Lakhs = temp_float*100000
        usedcar['car_prices_in_rupee'].iloc[i] = float(usedcar['car_prices_in_rupee'].iloc[i])
        usedcar['car_prices_in_rupee'].iloc[i] = covert_to_Lakhs
        
    elif usedcar['car_prices_in_rupee'].iloc[i].__contains__(' Crore'):
        usedcar['car_prices_in_rupee'].iloc[i] = usedcar['car_prices_in_rupee'].iloc[i].replace(' Crore','')
        temp_float = float(usedcar['car_prices_in_rupee'].iloc[i])
        covert_to_crore = temp_float*10000000
        usedcar['car_prices_in_rupee'].iloc[i] = float(usedcar['car_prices_in_rupee'].iloc[i])
        usedcar['car_prices_in_rupee'].iloc[i] = covert_to_crore
        
    elif usedcar['car_prices_in_rupee'].iloc[i].__contains__(','):
        usedcar['car_prices_in_rupee'].iloc[i] = usedcar['car_prices_in_rupee'].iloc[i].replace(',','')
        usedcar['car_prices_in_rupee'].iloc[i] = float(usedcar['car_prices_in_rupee'].iloc[i])     


# In[93]:


usedcar['car_prices_in_rupee'] = usedcar['car_prices_in_rupee'].astype(int)


# In[94]:


usedcar['kms_driven'] = usedcar['kms_driven'].replace({',':''},regex=True)
usedcar['kms_driven'] = usedcar['kms_driven'].replace({' kms':''},regex=True)
usedcar['kms_driven'] = usedcar['kms_driven'].astype(int)
usedcar.head()


# In[95]:


usedcar.rename(columns = {'engine':'engine_in_cc'}, inplace = True)
usedcar['engine_in_cc'] = usedcar['engine_in_cc'].replace({' cc':''},regex=True)
usedcar.head()


# In[96]:


usedcar[['Make','Model']] = usedcar.car_name.apply(lambda x : pd.Series(str(x).split(" ",1)))
usedcar.head()


# In[97]:


usedcar_df = usedcar.copy()
usedcar_df.to_csv('used_car_price_cleaned.csv')


# In[98]:


#Store a list of columns in variable:
colName = list(usedcar_df.columns)    
print(colName)    
usedcar_df['Make'].value_counts().plot(kind='bar') 
tempMakedf= usedcar_df[colName[7]].value_counts().rename_axis('Make').to_frame('Counts').reset_index()


# In[99]:


# Plot for Selling Price:
sns.histplot(data=usedcar_df,x='car_prices_in_rupee')
plt.title("Spread of Selling Price",size=15)
plt.show()


# # This shows that it is Right/positive skewed as tail is on the right side. Hence we apply log scale transformation
# 

# In[100]:


plot_sp =sns.histplot(data=usedcar_df,x='car_prices_in_rupee',kde=True,element="poly",log_scale=True)
plt.title("Log Transformation for Output variable : Selling Price",size=15)
plt.show()


# In[101]:


plt.figure(figsize=(12, 8))

sns.scatterplot(x='manufacture',y='car_prices_in_rupee',data=usedcar_df)
# Setting the label for x-axis
plt.xlabel("manufacture", size=15)

# Setting the label for y-axis
plt.ylabel("Selling car_prices_in_rupee", size=15)

# Setting the title for the graph
plt.title("Scatter Plot between year and Selling Price", size=15)
# Fianlly showing the plot
plt.tight_layout()


# In[102]:


plt.figure(figsize=(15, 8))

ax=sns.countplot(x='manufacture',data=usedcar_df)
for p in ax.patches:
   ax.annotate(p.get_height(), (p.get_x()+0.2, p.get_height()+0.25))


# In[103]:


plt.figure(figsize=(12, 8))

sns.scatterplot(x='kms_driven',y='car_prices_in_rupee',data=usedcar_df)
# Setting the label for x-axis
plt.xlabel("Km Driven", size=15)

# Setting the label for y-axis
plt.ylabel("Selling Price", size=15)

# Setting the title for the graph
plt.title("Scatter Plot between Km Driven and Selling Price", size=15)
# Fianlly showing the plot
plt.tight_layout()


# In[104]:


"""Function for creating bar graph between 2 variables and adding labels on top of bars"""
def labels_barPlot(dataFrame_Name,xAxisColName,yAxisColName,xlabelName,ylabelName,
                   grphTitle,extrahtforlabel =5,sizelabel=13,
                  figureht=8,figurewidt=12,xlabelNameSize=18,ylabelNamesize=18,grphTitlesize=18):
    
    plt.figure(figsize=(figurewidt, figureht))
    plots= sns.barplot(x=xAxisColName,y=yAxisColName,data=dataFrame_Name)
    """
  # Using Matplotlib's annotate function and
  # passing the coordinates where the annotation shall be done
  # x-coordinate: bar.get_x() + bar.get_width() / 2
  # y-coordinate: bar.get_height()
  # sfree space to be left to make graph pleasing: (0, 8)
  # ha and va stand for the horizontal and vertical alignment
    """
    for bar in plots.patches:
        plots.annotate(format(bar.get_height(), '.0f'),(bar.get_x() + bar.get_width() / 2, 
                                                        bar.get_height()+extrahtforlabel), ha='center', 
                       va='center', size=sizelabel, xytext=(0, 8),textcoords='offset points')   
        
    
    # Setting the label for x-axis
    plt.xlabel(xlabelName, size=xlabelNameSize)
      
    # Setting the label for y-axis
    plt.ylabel(ylabelName, size=ylabelNamesize)
      
    # Setting the title for the graph
    plt.title(grphTitle, size=grphTitlesize)
    # Fianlly showing the plot
    plt.tight_layout()
   # plt.show()   
#------------------------------------------------------------------------------------------    


# In[105]:


#--- How many kms was driven for different type of transmission car
trans_km =usedcar_df.groupby(['transmission']).mean()['kms_driven'].rename_axis('Trans').to_frame('TotalKM').reset_index()

labels_barPlot(trans_km,'Trans','TotalKM',
               'Type of Transmission',
               'Avg KM driven','Transmission wise km driven')


# In[ ]:





# In[106]:


usedcar_df['fuel_type'].unique()


# In[107]:


fuelType_kmDriven_carMake = usedcar_df.groupby(['fuel_type','Make'])['kms_driven'].sum().reset_index()
fuelType_kmDriven_carMake


# In[108]:


fuelType_kmDriven_carMake['kms_driven']=fuelType_kmDriven_carMake['kms_driven'].astype(float)
for i in fuelType_kmDriven_carMake.index:
    if fuelType_kmDriven_carMake['kms_driven'].iloc[i]>1000:
        temp_float = fuelType_kmDriven_carMake['kms_driven'].iloc[i]/1000
        fuelType_kmDriven_carMake['kms_driven'].iloc[i]=round(temp_float,2)
    


# In[109]:


fuelType_kmDriven_carMake


# In[110]:


plt.figure(figsize=(34, 48))

ax=sns.barplot(data=fuelType_kmDriven_carMake,y='Make',x='kms_driven',hue='fuel_type')
for bar in ax.patches:
    ax.annotate(format(bar.get_height(), '.0f'),(bar.get_x() + bar.get_width() / 2, 
                                                        bar.get_height()+0.25), ha='center', 
                       va='center', size=2, xytext=(0, 8),textcoords='offset points')
# Setting the label for x-axis
plt.xlabel("Make", size=15)
plt.xticks(size=20)
# Setting the label for y-axis
plt.ylabel("Km Driven", size=25)
plt.yticks(size=20)

# Setting the title for the graph
plt.title("Scatter Plot between Km Driven and Selling Price", size=15)
# Fianlly showing the plot
plt.tight_layout()


# In[111]:


fuelType_kmDriven_carMake_count = usedcar_df.groupby(['fuel_type','Make'])['kms_driven'].count().reset_index()
fuelType_kmDriven_carMake_count


# In[112]:


fuelType_kmDriven_carMake_count.rename(columns = {'kms_driven':'No_of_Cars_forSale'}, inplace = True)
fuelType_kmDriven_carMake_count


# In[113]:


with sns.plotting_context(font_scale=15.5):
    
    
    graph = sns.FacetGrid(data=fuelType_kmDriven_carMake_count[fuelType_kmDriven_carMake_count['No_of_Cars_forSale']>0]
                                      , col='fuel_type',col_wrap=5)
    graph.map(sns.barplot, "No_of_Cars_forSale",  "Make", edgecolor ="w").add_legend()

    graph.set_axis_labels("Number of Cars for Sale", "Make")
    font_dict = {'fontsize': 5, 'fontweight': 'heavy', 'verticalalignment':'top'}
    graph.set_xticklabels(fontdict=font_dict, rotation=90)
    graph.set_yticklabels(fontdict= {'fontsize': 3, 'fontweight': 'heavy', 'horizontalalignment':'right'}, rotation=0)
    graph.set_titles(col_template="{col_name}", fontweight='bold', size=9)


# In[114]:


ownership_price_mean = usedcar_df.groupby(['ownership'])['car_prices_in_rupee'].mean().reset_index()

ownership_price_mean = ownership_price_mean.sort_values('car_prices_in_rupee', ascending=False)
ownership_price_mean


# In[115]:


ownership_price_mean1 = usedcar.groupby(['ownership'])['car_prices_in_rupee'].mean().reset_index()
ownership_price_mean1 = ownership_price_mean1.sort_values('car_prices_in_rupee', ascending=False)
labels_barPlot(ownership_price_mean1,'ownership','car_prices_in_rupee',
               'Type of Ownership',
               'Average Selling Price','Ownership wise price distribution')


# In[116]:


transmission_price_mean1 = usedcar.groupby(['transmission'])['car_prices_in_rupee'].mean().reset_index()
transmission_price_mean1 = transmission_price_mean1.sort_values('car_prices_in_rupee', ascending=False)
labels_barPlot(transmission_price_mean1,'transmission','car_prices_in_rupee',
               'Type of Transmission',
               'Average Selling Price','Transmission wise price distribution')


# In[117]:


make_km_mean1 = usedcar.groupby(['Make'])['kms_driven'].mean().reset_index()
make_km_mean1 = make_km_mean1.sort_values('kms_driven', ascending=False)

plt.figure(figsize=(12,8))
sns.barplot(data=make_km_mean1,x='kms_driven',y='Make')
# Setting the label for x-axis
plt.xlabel("Average Kms driven", size=15)

# Setting the label for y-axis
plt.ylabel("Make of the car", size=15)

# Setting the title for the graph
plt.title("Make vs Avg km driven distribution", size=15)
# Fianlly showing the plot
plt.tight_layout()
# plt.show()   


# In[118]:


# Agrupamento do car_name com a média de kms_driven
mean_km_carname = usedcar_df.groupby(['Make']).agg({'kms_driven':'mean'}).sort_values('kms_driven', ascending=False).head(5)

# Plot evolução da média de km rodado por marca
mean_km_carname.plot.bar()


# In[119]:


df_plot = usedcar_df[usedcar_df['Make'].isin(mean_km_carname.index.values)].groupby(['Make', 'manufacture']).agg({'kms_driven':'mean'})
df_plot


# In[ ]:





# In[120]:


plt.figure(figsize=(15,5))
sns.lineplot(data=df_plot, x='manufacture', y='kms_driven', hue = 'Make')
plt.grid('on')


# In[121]:


#xyz = pd.crosstab(usedcar_df['Make'], usedcar_df['manufacture'],me)
carname_year_kmdriven_price_mean = usedcar_df.pivot_table(index=['Make','manufacture'], 
                                                          values=["kms_driven"],
                                                          aggfunc=np.mean, fill_value=0)
carname_year_kmdriven_price_mean = carname_year_kmdriven_price_mean.sort_values('kms_driven', ascending=False)
carname_year_kmdriven_price_mean


# In[122]:


sns.lineplot(data=carname_year_kmdriven_price_mean.head(7), x='manufacture', y='kms_driven', hue = 'Make')


# In[ ]:





# In[123]:


#graph = sns.FacetGrid(data=fuelType_kmDriven_carMake_count[fuelType_kmDriven_carMake_count['No_of_Cars_forSale']>0]
#                                  , col='Make', height=10, aspect=15,col_wrap=5)
#graph.map(sns.barplot, "No_of_Cars_forSale",  "fuel_type", edgecolor ="w").add_legend()

#graph.set_axis_labels("Number of Cars for Sale", "Make")
#graph.set_yticklabels(np.arange(0,210,50), rotation=90)


# In[124]:


#usedcar_df = usedcar_df.drop(columns='car_name')
usedcar_df['engine_in_cc'] = usedcar_df['engine_in_cc'].astype(int)
usedcar_df['manufacture'] = usedcar_df['manufacture'].astype(int)
usedcar_df


# In[125]:


sns.heatmap(usedcar_df.corr(), annot=True)


# In[126]:


#H₀: The two categorical variables have no relationship
#H₁: There is a relationship between two categorical variables
#The number of degrees of freedom of the χ2 independence test statistics:
#d.f. = (# rows -1) *(#columns-1)


# In[127]:


# Contingency Table
# To run the Chi-Square Test, the easiest way is to convert the data into 
# a contingency table with frequencies. We will use the crosstab command from pandas.
# fuel_type vs transmission
contigency= pd.crosstab(usedcar_df['fuel_type'], usedcar_df['transmission'])
contigency


# In[128]:


contigency_pct = pd.crosstab(usedcar_df['fuel_type'], usedcar_df['transmission'], normalize='index')
contigency_pct


# In[129]:


plt.figure(figsize=(12,8))
sns.heatmap(contigency, annot=True, cmap="YlGnBu")


# In[130]:


# Chi-square test of independence.
from scipy.stats import chi2_contingency
c, p, dof, expected = chi2_contingency(contigency)

#H0= transmission and fuel type are correalted 
#h1 = not correlated

if p>0.05:
    print('Null Hypothesis not rejected - transmission and fuel type are correalted ')
elif p<0.05:
    print('Null Hypothesis rejected - transmission and fuel type are NOT correalted ')

    
print(f'{p:.33f}')


# In[131]:


usedcar_df


# In[132]:


usedcar_df['Make'].unique()


# In[133]:


usedcar_df['fuel_type'] = usedcar_df['fuel_type'].replace({'Diesel':4, 'Petrol':1, 'Cng':2, 'Electric':0, 'Lpg':3})
usedcar_df['transmission'] = usedcar_df['transmission'].replace({'Manual':0, 'Automatic':1})
usedcar_df['ownership'] = usedcar_df['ownership'].replace({'1st Owner':1, '2nd Owner':2, '3rd Owner':3, '4th Owner':4, 
                                                               '5th Owner':5,'0th Owner':0})
usedcar_df['Seats'] = usedcar_df['Seats'].replace({'5 Seats':5, '6 Seats':6, '7 Seats':7, '4 Seats':4, 
                                                               '8 Seats':8,'2 Seats':2})
usedcar_df['Make'] = usedcar_df['Make'].replace({'Jeep':0, 'Renault':1, 'Toyota':2, 'Honda':3, 'Volkswagen':4, 'Maruti':5,
                                                   'Mahindra':6, 'Hyundai':7, 'Nissan':8, 'MG':9, 'Tata':10, 'BMW':11,
                                                   'Mercedes-Benz':12, 'Datsun':13, 'Volvo':14, 'Audi':15, 'Porsche':16, 'Ford':17,
                                                   'Chevrolet':18, 'Skoda':19, 'Lexus':20, 'Land':21, 'Mini':22, 'Jaguar':23,
                                                   'Mitsubishi':24, 'Force':25, 'Premier':26, 'Fiat':27, 'Maserati':28, 'Bentley':29,
                                                   'Isuzu':30, 'Kia':31})


# In[134]:


usedcar_df


# In[135]:


from scipy.stats import chisquare

#usedcar_df['p_value'] = chisquare(usedcar_df[['fuel_type', 'transmission', 'ownership','Seats', 'Make']], axis=1)[1]
#usedcar_df['same_diff'] = np.where(usedcar_df['p_value'] > 0.05, 'same', 'different')


# In[136]:


usedcar_df


# In[137]:


def corelation_categorical(categorical_column_list, df):
    from scipy.stats import chi2_contingency
    import itertools
    for i in range(len(categorical_column_list)):
        for j in range(i + 1, len(categorical_column_list)):
            #print(col_list[i], col_list[j])
            contigency= pd.crosstab(df[col_list[i]], df[col_list[j]])
           # print(contigency)
            c, p, dof, expected = chi2_contingency(contigency)
            if p>0.05:
                print('Null Hypothesis not rejected ==>',col_list[i], ' and ',col_list[j], ' are co-related')
            elif p<0.05:
                print('Null Hypothesis rejected ==> ',col_list[i], ' and ',col_list[j], ' are NOT co-related')


# In[138]:


col_list = ['fuel_type','transmission', 'ownership', 'Seats','Make']
corelation_categorical(col_list, usedcar_df)


# In[139]:


# ANOVA is used when the categorical variable has at least 3 groups (i.e three different unique values).
usedcar_df['transmission'].unique()
# Hence ANOVA cannot be used on transmission column


# In[ ]:





# In[140]:


usedcar_df = usedcar_df.drop(columns='car_name')
usedcar_df


# In[141]:


from scipy.stats import f_oneway
CategoryGroupLists1=usedcar_df.groupby('fuel_type')['car_prices_in_rupee'].apply(list)
CategoryGroupLists2=usedcar_df.groupby('ownership')['car_prices_in_rupee'].apply(list)
CategoryGroupLists3=usedcar_df.groupby('Seats')['car_prices_in_rupee'].apply(list)
CategoryGroupLists4=usedcar_df.groupby('Make')['car_prices_in_rupee'].apply(list)


AnovaResults = f_oneway(*CategoryGroupLists1)
print('P-Value for Anova fuel_type vs  car_prices_in_rupee is: ', AnovaResults[1])
print('F-Value for Anova fuel_type vs  car_prices_in_rupee is: ', AnovaResults[0])


# In[142]:


AnovaResults = f_oneway(*CategoryGroupLists4)
print('P-Value for Anova Make vs  car_prices_in_rupee is: ', AnovaResults[1])
print('F-Value for Anova Make  vs  car_prices_in_rupee is: ', AnovaResults[0])


# In[143]:


AnovaResults = f_oneway(*CategoryGroupLists2)
print('P-Value for Anova ownership vs  car_prices_in_rupee is: ', AnovaResults[1])
print('F-Value for Anova ownership  vs  car_prices_in_rupee is: ', AnovaResults[0])


# In[144]:


AnovaResults = f_oneway(*CategoryGroupLists3)
print('P-Value for Anova Seats vs  car_prices_in_rupee is: ', AnovaResults[1])
print('F-Value for Anova Seats  vs  car_prices_in_rupee is: ', AnovaResults[0])


# In[145]:


from scipy.stats import ttest_ind
#define samples -Independent Two Sample t-Test
group1 = usedcar_df[usedcar_df['transmission']==0]
group2 = usedcar_df[usedcar_df['transmission']==1]
# 'Manual':0, 'Automatic':1
ttest_ind(group1['car_prices_in_rupee'], group2['car_prices_in_rupee'])


# In[146]:


from scipy.stats import ttest_ind
#Welch’s t-Test in Pandas
#Welch’s t-test is similar to the independent two sample t-test,
#except it does not assume that the two populations that the samples came from have equal variance.
ttest_ind(group1['car_prices_in_rupee'], group2['car_prices_in_rupee'], equal_var=False)


# # Linear Regression

# In[147]:


usedcar_df


# In[148]:


df1 = usedcar_df[list(usedcar_df.columns)[1:9]]
df1


# In[149]:


# Normalizando kms_driven: 
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
df1['kms_driven'] = MinMaxScaler().fit_transform(df1['kms_driven'].values.reshape(len(df1), 1))
df1


# In[150]:


from sklearn.linear_model import LinearRegression  #Import Linear regression model
from sklearn.model_selection import train_test_split  #To split the dataset into Train and test randomly
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score
train_X,test_X,train_y,test_y=train_test_split(df1,usedcar_df['car_prices_in_rupee'],test_size=0.3,random_state=0)


# In[151]:


model = LinearRegression()
model.fit(train_X,train_y)


# In[152]:


train_predict_lr = model.predict(train_X)
test_predict_lr = model.predict(test_X)


# In[153]:


print("R2 SCORE")
print("Train : ",r2_score(train_y,train_predict_lr))
print("Test  : ",r2_score(test_y,test_predict_lr))  
print("====================================")


# In[154]:


# In[44]:

plt.figure(figsize=(10,10))
plt.scatter(test_y,test_predict_lr, c='crimson')
#plt.yscale('log')
#plt.xscale('log')

p1 = max(max(test_predict_lr), max(test_y))
p2 = min(min(test_predict_lr), min(test_y))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')
plt.show()


# In[155]:


# Knn Regression
from sklearn.neighbors import KNeighborsRegressor
model_KNNReg = KNeighborsRegressor()
model_KNNReg.fit(train_X,train_y)


# In[156]:


train_predict_knn = model_KNNReg.predict(train_X)
test_predict_knn = model_KNNReg.predict(test_X)


# In[157]:


predicted_df = pd.DataFrame({"Actual":test_y,"Predicted":test_predict_knn})
predicted_df.head()


# In[158]:


print("Train : ",mean_absolute_error(train_y,train_predict_knn))
print("Test  : ",mean_absolute_error(test_y,test_predict_knn))
print("====================================")

print("MSE")
print("Train : ",mean_squared_error(train_y,train_predict_knn))
print("Test  : ",mean_squared_error(test_y,test_predict_knn))
print("====================================")

print("RMSE")
print("Train : ",np.sqrt(mean_squared_error(train_y,train_predict_knn)))
print("Test  : ",np.sqrt(mean_squared_error(test_y,test_predict_knn)))
print("====================================")

print("R2 SCORE")
print("Train : ",r2_score(train_y,train_predict_knn))
print("Test  : ",r2_score(test_y,test_predict_knn))  
print("====================================")


# In[159]:


from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold


# In[160]:


lasso_model = Lasso(alpha=1.0)
lasso_model.fit(train_X,train_y)


# In[161]:


print("Intercept : ", lasso_model.intercept_)
print("Slope : ", lasso_model.coef_)


# In[162]:


from sklearn.model_selection import GridSearchCV
from numpy import mean
from numpy import std
from numpy import absolute
from numpy import arange
# define model evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=1)
# evaluate model
scores = cross_val_score(lasso_model, train_X, train_y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# force scores to be positive
scores = absolute(scores)
print('Mean MAE: %.3f (%.3f)' % (mean(scores), std(scores)))
print("====================================")


# In[163]:


#Predicting TEST & TRAIN DATA
train_predict = lasso_model.predict(train_X)
test_predict = lasso_model.predict(test_X)


# In[164]:


print("====================================")
print("MAE")
print("Train : ",mean_absolute_error(train_y,train_predict))
print("Test  : ",mean_absolute_error(test_y,test_predict))
print("====================================")

print("MSE")
print("Train : ",mean_squared_error(train_y,train_predict))
print("Test  : ",mean_squared_error(test_y,test_predict))
print("====================================")

print("RMSE")
print("Train : ",np.sqrt(mean_squared_error(train_y,train_predict)))
print("Test  : ",np.sqrt(mean_squared_error(test_y,test_predict)))
print("====================================")

print("R2 SCORE")
print("Train : ",r2_score(train_y,train_predict))
print("Test  : ",r2_score(test_y,test_predict))  
print("====================================")


# In[44]:


plt.figure(figsize=(10,10))
plt.scatter(test_y,test_predict, c='crimson')
#plt.yscale('log')
#plt.xscale('log')

p1 = max(max(test_predict), max(test_y))
p2 = min(min(test_predict), min(test_y))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')
plt.show()


# In[165]:


# Random Forest
from sklearn.ensemble import RandomForestRegressor
train_X,test_X,train_y,test_y=train_test_split(df1,usedcar_df['car_prices_in_rupee'],test_size=0.3,random_state=42)

model_RFReg = RandomForestRegressor(random_state=42)
model_RFReg.fit(train_X,train_y)

train_predict_rf = model_RFReg.predict(train_X)
test_predict_rf = model_RFReg.predict(test_X)


# In[166]:



predicted_df = pd.DataFrame({"Actual":test_y,"Predicted":test_predict_rf})
predicted_df.head()


# In[167]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score
model_RFReg.score


# In[168]:


plt.figure(figsize=(10,10))
plt.scatter(test_y,test_predict, c='crimson')
#plt.yscale('log')
#plt.xscale('log')

p1 = max(max(test_predict_rf), max(test_y))
p2 = min(min(test_predict_rf), min(test_y))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')
plt.show()


# In[169]:


# Create pickle files -KNN Regressor
import pickle
pickle_out = open("knn_regressor.pkl","wb")
pickle.dump(model_KNNReg, pickle_out)
pickle_out.close()


# In[175]:


# Create pickle files -Linear Regressor
import pickle
pickle_out = open("lr_regressor.pkl","wb")
pickle.dump(model, pickle_out)
pickle_out.close()


# In[171]:


# Create pickle files -Lasso Regressor
import pickle
pickle_out = open("lasso_regressor.pkl","wb")
pickle.dump(lasso_model, pickle_out)
pickle_out.close()


# In[172]:


# Create pickle files -RF Regressor
import pickle
pickle_out = open("rf_regressor.pkl","wb")
pickle.dump(model_RFReg, pickle_out)
pickle_out.close()


# In[ ]:





# In[173]:


import bisect


# In[ ]:





# In[ ]:




