import streamlit as st
import pandas as pd
import io
st.title("Economic indicators and Stock Index Price Analysis")
file=st.file_uploader("C:/economic_data.csv")
data=pd.read_csv(file)
st.header("Dataset Information")
st.write("Kaggle data set with 6 columns & 1200 entries \n\n columns : year, month, unemployment rate, interest rate, stock index price")
data
buffer=io.StringIO()
data.info(buf=buffer)
info_str=buffer.getvalue()
st.text("DataFrame Info:")
st.text(info_str)
st.write("No null entries")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
st.header("Quantitative Variable Analysis: Correlation & Regression")
qunati_data=data.drop(columns=['year','month'])
qunati_data.describe()
for col in qunati_data:
    Q1 = qunati_data[col].quantile(0.25)
    Q3 = qunati_data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data_new = qunati_data[(qunati_data[col] >= lower_bound) & (qunati_data[col] <= upper_bound)]
buffer=io.StringIO()
data_new.info(buf=buffer)
info_str=buffer.getvalue()
st.text("DataFrame without Outliers Info:")
st.text(info_str)
st.subheader("Histogram of each quantitative variable")
fig, ax = plt.subplots(1, 3, figsize=(9, 4))
for i, col in enumerate(data_new): 
    sns.histplot(data=data_new, x=col, ax=ax[i])  
    ax[i].set_ylabel('')
plt.tight_layout()
st.pyplot(fig)
st.write("Not extremely skewed")
st.subheader("Boxplot of each quantitative variable")
fig, ax = plt.subplots(1, 3, figsize=(9, 4))
for i, col in enumerate(data_new):
    sns.boxplot(y=data_new[col], ax=ax[i])
    ax[i].set_title(f'Boxplot of {col}')
plt.tight_layout()
st.pyplot(fig)
st.write("No extreme outliers")
st.subheader("Correlation Analysis with stock index price")
correlation_matrix = data_new.corr()
correlation_matrix
st.subheader("Heatmap of the correlation matrix")
fig, ax=plt.subplots(figsize=(6, 4))
sns.heatmap(correlation_matrix, annot=True, cmap='Blues', fmt='.2f', cbar=True)
plt.title('Correlation Matrix of Economic Indicators and Stock Index Price')
st.pyplot(fig)
st.write("Analysis:")
st.write(" Index price & interest rate seems to have strong positive linear relationship (0.89) \n\n Unemployment rate & interest rate seems to have weak negative linear relationship (-0.43)")
st.subheader("Pairplot of quantitative variables")
fig=sns.pairplot(data_new[['interest_rate', 'unemployment_rate', 'index_price']])
st.pyplot(fig)
from sklearn.linear_model import LinearRegression
x = data_new[['interest_rate', 'unemployment_rate']]
y = data_new['index_price']
model=LinearRegression()
model.fit(x,y)
y_pred=model.predict(x)
st.subheader("Linear Regression Analysis")
st.write(f'Coefficients: {model.coef_}')
st.write(f'Intercept: {model.intercept_}')
st.subheader("Regression Plot of Interest Rate and Stock Index Price")
x=data_new['interest_rate']
y=data_new['index_price']
st.write("Analysis : ")
st.write("When 1% of interest rate increase, 310.02 of stock index price increase")
st.write("When 1% of unemployment increase, 106.78 of stock index price decrease")
fig, ax=plt.subplots(figsize=(8,6))
sns.regplot(x=x,y=y,scatter_kws={'color':'blue'},line_kws={'color':'red'},ax=ax)
ax.set_title('Regression Plot: Interest Rate and Stock Index Price')
ax.set_xlabel('Interest Rate')
ax.set_ylabel('Stock Index Price')
st.pyplot(fig)
st.subheader("Regression Plot of Unemployment Rate and Stock Index Price")
x=data_new['unemployment_rate']
y=data_new['index_price']
fig, ax=plt.subplots(figsize=(8,6))
sns.regplot(x=x,y=y,scatter_kws={'color':'blue'},line_kws={'color':'red'},ci=None,ax=ax)
ax.set_title('Regression Plot: Unemployment Rate and Stock Index Price (order=1)')
ax.set_xlabel('Unemployment Rate')
ax.set_ylabel('Stock Index Price')
st.pyplot(fig)
x=data_new['unemployment_rate']
y=data_new['index_price']
fig, ax=plt.subplots(figsize=(8,6))
sns.regplot(x=x,y=y,scatter_kws={'color':'blue'},line_kws={'color':'red'},order=2,ci=None,ax=ax)
ax.set_title('Regression Plot: Unemployment Rate and Stock Index Price (order=2)')
ax.set_xlabel('Unemployment Rate')
ax.set_ylabel('Stock Index Price')
st.pyplot(fig)
st.header("Qualitative Variable Analysis: 2019~2020 Stock Index Price")
st.write("As time pass, various events effect the Stock Index Price")
st.write("Hypothesis: Covid-19 might have highly affected the stock index price")
st.write("Test: Use lineplot to check 2019~2020 stock index price")
quali_data=data.loc[(data['year']==2019)|(data['year']==2020)]
quali_data['date']=pd.to_datetime(quali_data['year'].astype('str')+'-'+quali_data['month'].astype('str'))
quali_data.set_index('date',inplace=True)
st.write("Data with date index")
quali_data
st.subheader("Lineplot of Stock Index Price 2019~2020")
fig, ax = plt.subplots(figsize=(10,5))
sns.lineplot(data=quali_data, x='date', y='index_price',marker='o',linestyle='--', ax=ax);
ax.set_title("Lineplot of Stock Index Price 2019~2020", fontsize=16);
ax.set_xlabel("Date");
ax.set_ylabel("Stock Index Price")
st.pyplot(fig)
st.write("Analysis: ")
st.write("1) Volatility: 2020 was marked by extreme volatility, reflecting uncertainty during the pandemic and subsequent recovery.")
st.write("2) Policy Influence: Market trends in both 2019 and 2020 were heavily influenced by policy actions, trade negotiations, and pandemic-related events.")
st.header("Conclusion: ")
st.write("Interest rate and unemployment rate both have meaningful correlation with the stock index price.") 
st.write("Interest rate has strong positive linear correlation with the correlation coefficient 0.89 and regression equation (index price) =  3.915 + 310.02 * (interest rate).Unemployment rate has relatively weak negative linear correlation with the correlation coefficient -0.43 and regression equation (index price) = 3.915 - 106.786 * (unemployment rate).") 
st.write("Additionally, events like COVID-19 might highly affect the stock index price, thus the volatility of the stock index price should be analyzed in various perspectives.")
st.write("Based on these findings, investors can predict stock market volatility in response to interest rate changes, while policymakers can consider the impact on the stock market when formulating economic policies.")
st.write("Future research should further analyze the relationships with other economic indicators to develop more refined predictive models.")
