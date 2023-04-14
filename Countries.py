import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import date, timedelta
from datetime import datetime as dt
import plotly.express as px
from zipfile import ZipFile
from dateutil.relativedelta import relativedelta
import xgboost as xgb
import re
from sklearn.metrics import mean_squared_error

#df=pd.read_csv('//Users//reema//Desktop//Research//Streamlit//pages//COVID19_Data.csv')
with ZipFile('COVID19 Data 2.csv.zip', 'r') as zip:
  zip.extractall()

df=pd.read_csv('COVID19 Data 2.csv')

st.title('General Overview Per Country')

def map():
    c=df.country.unique()
    code=df.iso3.unique()
    ans=[]
    for i in c:
        ans.append(df[df['country']==i]['cumulative_cases'].iloc[-1])
        
    cases_df=pd.DataFrame(zip(ans,c,code),columns=['Cases','Countries','Code'])
    fig = px.scatter_geo(cases_df, locations="Code", color="Countries",hover_name="Countries", size="Cases", projection="natural earth")
    fig.update_layout(width= 1200,height=700, margin=dict(l=5))
    st.plotly_chart(fig)

map()

def preparing_data(df, country = 'Jordan', train_date = '01-01-2022', last_date='Nothing', target = 'cases'):
    df = df.copy()
    df = df[df['country'] == country]
    
    cumulative_target = 'cumulative_'+target
    
    s = df[cumulative_target].drop_duplicates()
    s = s.diff().fillna(s)
    dfFilter = pd.merge(s, df[cumulative_target], right_index=True, left_index=True)

    df[target] = None
    set1 = set()
    
    for i in range(len(df)):
        for j in range(len(dfFilter)):
            if df[cumulative_target].iloc[i] == dfFilter[cumulative_target+'_y'].iloc[j] and df[cumulative_target].iloc[i] not in set1:
                df[target].iloc[i] = dfFilter[cumulative_target+'_x'].iloc[j]
                set1.add(df[cumulative_target].iloc[i])
    
    df[target] = df[target].fillna(0)
    
    
    df['date2'] = df['date'].copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    df.index = pd.to_datetime(df.index)

    df = df.rename(columns={'date2': 'date'})

    df['day of the year'] = df.index.day_of_year
    df['quarter'] = df.index.quarter
    
    return df
    
def charts(df, country = 'Jordan', train_date = '01-01-2022', last_date='Nothing', target = 'cases'):
    if 'datetime64[ns]' != df.index.dtype:
        df = preparing_data(df, country, train_date, last_date, target)

    fig = px.scatter(df, x=df.index, y=target,width=500, height=350)
    fig.update_layout( yaxis_title=target, xaxis_title='Date')
    st.plotly_chart(fig)

    fig = px.box(df, x='quarter', y=target, width=500, height=350)
    fig.update_layout( yaxis_title=target, xaxis_title='Date')
    st.plotly_chart(fig)

def time_series(df, country = 'Jordan', train_date = '01-01-2022', last_date='Nothing', target = 'cases'):
    if 'datetime64[ns]' != df.index.dtype:
        df = preparing_data(df, country, train_date, last_date, target)
        
    train = df.loc[df.index < train_date]
    if last_date == 'Nothing':
      test = df.loc[df.index >= train_date]
    else:
      test = df.loc[(df.index >= train_date) & (df.index <= last_date)]

    FEATURES = ['day', 'day of the year', 'day of the week', 'quarter', 'month', 'year']
    TARGET = target

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]
    
    reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree',    
                           n_estimators=1000,
                           early_stopping_rounds=50,
                           objective='reg:linear',
                           max_depth=3,
                           learning_rate=0.01)
    reg.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=100)

    test['prediction'] = reg.predict(X_test)
    df = df.merge(test[['prediction']], how='left', left_index=True, right_index=True)

    if last_date != 'Nothing':
      for i in range(len(df)):
        if dt.strptime(df['date'].iloc[i], '%Y-%m-%d') > dt.strptime(last_date, '%d-%m-%Y'):
          df['prediction'].iloc[i] = None

    if last_date != 'Nothing':
      if dt.strptime(last_date, '%d-%m-%Y') > dt.strptime('11-09-2022', '%d-%m-%Y'):
        dates = list(pd.date_range(date(2022,9,12),date(dt.strptime(last_date, '%d-%m-%Y').year, dt.strptime(last_date, '%d-%m-%Y').month, dt.strptime(last_date, '%d-%m-%Y').day)-timedelta(days=1),freq='d'))
        df_future = pd.DataFrame(dates, columns=['date'])
        df_future['day of the week'] = df_future['date'].dt.dayofweek
        df_future['day of the year'] = df_future['date'].dt.dayofyear
        df_future['month'] = df_future['date'].dt.month
        df_future['year'] = df_future['date'].dt.year
        df_future['day'] = df_future['date'].dt.day
        df_future['quarter'] = df_future['date'].dt.quarter
        df_future['prediction'] = reg.predict(df_future[FEATURES])
        df_future['date'] = pd.to_datetime(df_future['date'])
        df = df.append(df_future, ignore_index=True)   

    score = np.sqrt(mean_squared_error(test[target], test['prediction']))

    fig = px.scatter(df, x=df.date, y=[target, 'prediction'] ,width=600, height=400)
    fig.update_layout( yaxis_title=target, xaxis_title='Date')
    st.plotly_chart(fig)
    return score


def regulations(df, country = 'Jordan', train_date = '01-01-2022', last_date='Nothing', target = 'cases',regulation='restriction_gatherings'):
    if 'datetime64[ns]' != df.index.dtype:
        df = preparing_data(df, country, train_date, last_date, target)
    df[regulation]=df[regulation].astype('object')
    fig = px.scatter(df, x=df.date, y=target, color=regulation,labels={regulation:'Regulation Label'},width=600, height=400)
    fig.add_scatter()
    fig.update_layout(yaxis_title=f"Number of COVID-19 {target}",xaxis_title='Date')
    st.plotly_chart(fig)

def regulations_special(df, country = 'Jordan', train_date = '01-01-2022', last_date='Nothing', target = 'cases',regulation='restriction_gatherings',special='workplaces'):
    if 'datetime64[ns]' != df.index.dtype:
        df = preparing_data(df, country, train_date, last_date, target)
    df[regulation]=df[regulation].astype('object')
    fig = px.scatter(df, x=df.date, color=special, y=target, facet_col=regulation,labels={special:'Index'},width=900, height=600)
    fig.add_scatter()
    fig.update_layout(yaxis_title=f"Number of COVID-19 {target}",xaxis_title='Date',margin=dict(l=15))
    st.plotly_chart(fig)

def stats(df,country,regulation,col):
  res=df[df['country']==country].groupby(regulation)[col].describe()[['mean','std']].reset_index().rename(columns={regulation:'Regulation Label','std':'Standard Deviation','mean':'Average'})
  return res

def viz(reg_name,option,reg):
    col1,col2=st.columns(2)
    with col1:
        st.subheader(f'{reg_name} Regulations VS COVID-19 Cases')
        regulations(df, country = option, train_date = '01-01-2022', target = 'cases',regulation=reg)
        x=stats(df,option,reg,'new_cases')
        st.write(x)
    with col2: 
        st.subheader(f'{reg_name} Regulations VS COVID-19 Deaths')
        regulations(df, country = option, train_date = '01-01-2022', target = 'deaths',regulation=reg)
        x=stats(df,option,reg,'new_deaths')
        st.write(x)


df1=df.copy()
res=df1[df1['country']=='Jordan'].groupby('restriction_gatherings')['new_cases'].describe()['mean'].to_frame()
res.rename(columns={'mean':'Average Number of Cases','restriction_gatherings':'Restriction Label'},inplace=True)
st.write('\n')
option=st.selectbox('Select Country',('India', 'Costa Rica', 'Germany', 'Mexico', 'Estonia', 'Myanmar',
       'Dominican Republic', 'Kenya', 'Botswana', 'Indonesia', 'Spain',
       'Uganda', 'Brazil', 'Togo', 'Zambia', 'Gabon', 'France',
       'El Salvador', 'Bangladesh', 'Palestine', 'Georgia', 'Honduras',
       'Peru', 'Angola', 'Italy', 'Cameroon', 'Rwanda', 'Cambodia',
       'Jamaica', 'Luxembourg', 'Japan', 'Ghana', 'Tajikistan',
       'Sri Lanka', 'Colombia', 'Jordan', 'Thailand', 'Argentina',
       'Senegal', 'Hungary', 'Malta', 'Iraq', 'Switzerland', 'Nigeria',
       'Sweden', 'Romania', 'Denmark', 'Belgium', 'Philippines',
       'South Africa', 'Mauritius', 'Zimbabwe', 'Canada', 'Ireland',
       'Pakistan', 'Kazakhstan', 'Malaysia', 'Paraguay', 'Serbia',
       'Mongolia', 'Poland', 'Latvia', 'Chile', 'Uruguay', 'Panama',
       'Belarus', 'Nepal', 'Portugal', 'Ecuador', 'Guatemala', 'Austria',
       'Australia', 'Lebanon', 'Croatia', 'Lithuania', 'Norway',
       'Mozambique', 'Greece', 'Bulgaria', 'Finland', 'Slovenia',
       'Papua New Guinea'))

st.write('\n ')
st.write('\n ')
st.write('\n ')
st.write(' \n -------------- ')

quartiles=pd.qcut(df.containment_index, 3, labels=["Low", "Medium", "High"])
df['cont_quar']=quartiles

st.title(option)
st.write('\n')
col1,col2,col3=st.columns(3)
with col1:
    st.metric(label='Population', value=df[df['country']==option]['pop2022'].iloc[-1])

with col2:
    st.metric(label='Average Containment Index', value=str(round(df[df['country']==option]['containment_index'].mean(),2)))

with col3:
    st.metric(label='Income Group', value=df[df['country']==option]['income group'].iloc[-1])  

col4,col5,col6=st.columns(3)
with col4:
    st.metric(label='LPI Rank*', value=df[df['country']==option]['legatumrank2020'].iloc[-1])

with col5:
    st.metric(label='Life Expectancy For Both Sexes', value=df[df['country']==option]['life expectancy(both sexes) '].iloc[-1])

# with col6:
#     st.metric(label='Pollution Index', value=df[df['country']==option]['Pollution Index'].iloc[-1])  

st.write('\n')
st.write('\n')
st.write('\n')

with st.expander("Deaths Due To COVID-19"):

    col1,col2= st.columns(2)
    with col1:
        st.subheader(f'COVID-19 Deaths In {option} Over Time')
        st.write('\n')
        charts(df, country = option, train_date = '01-01-2022', target = 'deaths')

    with col2:
         st.subheader(f'Predictive Model For COVID-19 Deaths In {option} Over Time')
         start_date= date(year=2022,month=1,day=2)
         end_date= dt.now().date()
         dat = st.slider("Select A Date:", min_value=start_date ,value=end_date,max_value=end_date,format='MMM DD, YYYY',key=1)
         final=re.sub(r'(\d{4})-(\d{1,2})-(\d{1,2})', '\\3-\\2-\\1', str(dat))
         sc=time_series(df, country = option, train_date = '01-01-2022', last_date= final, target = 'deaths')
         st.subheader('Model Evaluation')
         st.caption(f'The Models RMSE Score is {sc}')

st.write('\n')
st.write('\n')

with st.expander("COVID-19 Cases"):
    col1,col2= st.columns(2)
    with col1:
        st.subheader(f'COVID-19 Cases In {option} Over Time')
        st.write('\n')
        charts(df, country = option, train_date = '01-01-2022', target = 'cases')

    with col2:
        st.subheader(f'Predictive Model For COVID-19 Cases In {option} Over Time')
        start_date= date(year=2022,month=1,day=2)
        end_date= dt.now().date()
        dat = st.slider("Select the Date you want to view", min_value=start_date ,value=end_date,max_value=end_date,format='MMM DD, YYYY',key=2)
        final=re.sub(r'(\d{4})-(\d{1,2})-(\d{1,2})', '\\3-\\2-\\1', str(dat))
        sc=time_series(df, country = option, train_date = '01-01-2022', last_date= final, target = 'cases')
        st.subheader('Model Evaluation')
        st.caption(f'The Models RMSE Score is {sc}')

st.write('\n')
st.write('\n')

with st.expander("Regulations"):
    ans=st.selectbox(' Regulation ',['Facial Coverings','Vaccination','Workplace Closures','Public Events Cancelation','Internal Movement Restrictions','Gathgering Restrictions','Stay at Home Restictions'])
    if (ans=='Workplace Closures'):
        st.header(ans)
        st.caption("0: No restrictions \n")
        st.caption("1: Recommend closing (or work from home) \n")
        st.caption("2: Require closing (or work from home) for some sectors or categories of workers\n")
        st.caption("3: Require closing (or work from home) all but essential workplaces (e.g. grocery stores, doctors) \n")
        viz('Workplace Closures',option,'workplace_closures')
        st.subheader('COVID-19 Cases in Relation to Changes in Visitors to Workplaces')
        regulations_special(df, option, '01-01-2022', 'Nothing', 'cases',regulation='workplace_closures',special='workplaces')

    elif (ans=='Facial Coverings'):
        st.header(ans)
        st.caption("0: No policy \n")
        st.caption("1: Recommended \n")
        st.caption("2: Required in some specified shared/public spaces outside the home with other people present, or some situations when social distancing not possible \n")
        st.caption("3: Required in all shared/public spaces outside the home with other people present or all situations when social distancing not possible \n")
        st.caption("4: Required outside the home at all times, regardless of location or presence of other people \n")
        viz('Facial Coverings',option,'facial_coverings')

    elif (ans == 'Public Events Cancelation'):
        st.header(ans)
        st.caption("0: No measures \n")
        st.caption("1: Recommend cancelling \n")
        st.caption("2: Require cancelling\n")
        viz('Public Events Cancelation',option,'cancel_public_events')
            
    elif (ans == 'Internal Movement Restrictions'):
        st.header(ans)
        st.caption("0: No measures \n")
        st.caption("1: Recommend movement restriction \n")
        st.caption("2: Restrict movement \n")
        viz('Internal Movement Restrictions',option,'restrictions_internal_movements')
    
    elif (ans == 'Gathgering Restrictions'):
        st.header(ans)
        st.caption("0: No measures \n")
        st.caption("1: Restrictions on very large gatherings (the limit is above 1,000 people) \n")
        st.caption("2: Restrictions on gatherings between 100-1,000 people \n")
        st.caption("3: Restrictions on gatherings between 10-100 people \n")
        st.caption("4: Restrictions on gatherings of less than 10 people \n")
        viz('Gathgering Restrictions',option,'restriction_gatherings')

    elif (ans == 'Stay at Home Restictions'):
        st.header(ans)
        st.caption("0: No measures \n")
        st.caption("1: Recommend not leaving house\n")
        st.caption("2: Require not leaving house with exceptions for daily exercise, grocery shopping, and ‘essential’ trips \n")
        st.caption("3: Require not leaving house with minimal exceptions (e.g. allowed to leave only once every few days, or only one person can leave at a time, etc.)\n")
        viz('Stay at Home Restictions',option,'stay_home_requirements')
    
    elif (ans == 'Vaccination'):
        st.header(ans)
        st.header(ans)
        st.caption("0: No availability \n")
        st.caption("1: Availability for ONE of the following: key workers/ clinically vulnerable groups / elderly groups  \n")
        st.caption("2: Availability for TWO of the following: key workers/ clinically vulnerable groups / elderly groups \n")
        st.caption("3: Availability for ALL the following: key workers/ clinically vulnerable groups / elderly groups \n")
        st.caption("4: Availability for all three, plus partial additional availability (select broad groups/ages \n")
        st.caption("5: Universal availability \n")
        viz('Vaccination',option,'vaccination_policy')

st.write('\n')
st.write('\n')

with st.expander("Possible Impacting Factors"):
    st.write('PENDING')

