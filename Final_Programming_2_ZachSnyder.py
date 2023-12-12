#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import altair as alt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import streamlit as st


# ***

education_level = st.selectbox("Education Level:",
                    options = ["Less than high school",
                               "High school incomplete",
                               "High school graduate",
                               "Some college, no degree (includes some community college)",
                               "Two-year assoociate degree from a college or university",
                               "Four-year college or university degree/Bachelor's degree",
                               "Some postgraduate or professional schooling, no postgraduate degree",
                               "Postgraduate or professional degree, inclusing master's doctorate, medical, or law"])

st.write(f"Education level: {education_level}")

if education_level == "Less than high school":
    education_level = 1
elif education_level == "High school incomplete":
    education_level = 2
elif education_level == "High school graduate":
    education_level = 3
elif education_level == "Some college, no degree (includes some community college)":
    education_level = 4
elif education_level == "Two-year assoociate degree from a college or university":
    education_level = 5
elif education_level == "Four-year college or university degree/Bachelor's degree":
    education_level = 6
elif education_level == "Some postgraduate or professional schooling, no postgraduate degree":
    education_level = 7
elif education_level == "Postgraduate or professional degree, inclusing master's doctorate, medical, or law":
    education_level = 8


income = st.selectbox("Annual Household Income:",
                      options = ["Less than $10,000",
                                 "10 to under $20,000",
                                 "20 to under $30,000",
                                 "30 to under $40,000",
                                 "40 to under $50,000",
                                 "50 to under $75,000",
                                 "75 to under $100,000",
                                 "100 to under $150,000",
                                 "$150,000 or more"])
st.write(f"Annual Household Income: {income}")

if income == "Less than $10,000":
    income = 1
elif income == "10 to under $20,000":
    income = 2
elif income == "20 to under $30,000":
    income = 3
elif income == "30 to under $40,000":
    income = 4
elif income == "40 to under $50,000":
    income = 5
elif income == "50 to under $75,000":
    income = 6
elif income == "75 to under $100,000":
    income = 7
elif income == "100 to under $150,000":
    income = 8
elif income == "$150,000 or more":
    income = 9

parent = st.selectbox("Are you a parent?",
                      options = ["Yes",
                                 "No"])
if parent == "Yes":
    parent = 1
elif parent == "No":
    parent = 0

married = st.selectbox("Are you married?",
                      options = ["Yes",
                                 "No"])
if married == "Yes":
    married = 1
elif married == "No":
    married = 0


gender = st.selectbox("Are you male or female?",
                      options = ["Male",
                                 "Female"])
st.write(f"You are: {gender}")

if gender == "Male":
    gender = 0
elif gender == "Female":
    gender = 1


age = st.slider(label = "What is your age?",
                min_value = 1,
                max_value = 97,
                value = 22)
st.write("You are: ", age, "years old.")



inputs = pd.DataFrame({
    "income": [income],
    "education": [education_level],
    "parent": [parent],
    "married": [married],
    "female": [gender],
    "age": [age]
})


def clean_sm(x):
    return np.where(x == 1,
                   1,
                   0)


s = pd.read_csv("social_media_usage.csv")

ss = pd.DataFrame({
    "sm_li": clean_sm(s.web1h),
    "income": pd.Categorical(s.income, ordered = True),
    "income": np.where(s.income > 9, np.nan, s.income),
    "education": pd.Categorical(s.educ2, ordered = True),
    "education": np.where(s.educ2 > 8, np.nan, s.educ2),
    "parent": np.where(s.par == 1, 1, 0),
    "married": np.where(s.marital == 1, 1, 0),
    "female": np.where(s.gender == 2, 1, 0),
    "age": np.where(s.age > 98, np.nan, s.age),
    "age": pd.to_numeric(s.age)
})
ss = ss.dropna()

y = ss.sm_li
x = ss[["income", "education", "parent", "married", "female", "age"]]

x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                   stratify = y,
                                                   test_size = 0.2,
                                                   random_state = 5)


lr = LogisticRegression(class_weight = "balanced")
lr.fit(x_train, y_train)

linkedin_prediction = lr.predict(inputs)


st.write(f"Predicted Linkedin user: {'Yes' if linkedin_prediction == 1 else 'No'}")

linkedin_probability = lr.predict_proba(inputs)[:,1 ]
st.write(f"Probability of being a Linkedin user: {linkedin_probability}")





def clean_sm(x):
    return np.where(x == 1,
                   1,
                   0)


# In[16]:


toy_df = pd.DataFrame({'Column1': [0, 1, 2],
                       'Column2': [1, 0, 1]})

clean_sm(toy_df)


# ***

# #### Q3

# In[17]:


ss = pd.DataFrame({
    "sm_li": clean_sm(s.web1h),
    "income": pd.Categorical(s.income, ordered = True),
    "income": np.where(s.income > 9, np.nan, s.income),
    "education": pd.Categorical(s.educ2, ordered = True),
    "education": np.where(s.educ2 > 8, np.nan, s.educ2),
    "parent": np.where(s.par == 1, 1, 0),
    "married": np.where(s.marital == 1, 1, 0),
    "female": np.where(s.gender == 2, 1, 0),
    "age": np.where(s.age > 98, np.nan, s.age),
    "age": pd.to_numeric(s.age)
})
ss.tail()


# In[18]:


ss = ss.dropna()


# In[19]:


#ss_corr = ss.corr(numeric_only = False).style.background_gradient(cmap='coolwarm')
#ss_corr


# In[20]:


#alt.Chart(ss.groupby(["income", "education"], as_index=False)["sm_li"].mean()).mark_circle().\
#    encode(
#        x="education",
#        y="sm_li",
#        color="income",
#        tooltip=["income", "education", "sm_li"]
#    )


# ***

# #### Q4/5

# In[21]:


y = ss.sm_li
x = ss[["income", "education", "parent", "married", "female", "age"]]


# In[22]:


x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                   stratify = y,
                                                   test_size = 0.2,
                                                   random_state = 5)


# x_train contains 80% of the data, and is comprised of the feature variables used to make a prediction. \
# x_test contains 80% of the data, and is comprised of the target variable that is being predicted. \
# y_train contains 20% of the data, and is comprised of the feature variables used to make a prediction. \
# y_test contains 20% of the data, and is comprised of the target variable that is being predicted.

# ***

# #### Q6

# In[24]:


lr = LogisticRegression(class_weight = "balanced")


# In[25]:


lr.fit(x_train, y_train)


# ***

# #### Q7/8/9

# In[26]:


x_test.head()


# In[27]:


y_pred = lr.predict(x_test)


# In[28]:


pd.DataFrame(confusion_matrix(y_test, y_pred),
            columns = ["Predicted Negative", "Predicted Positive"],
            index = ["Actual Negative", "Actual Positive"]).style.background_gradient()


# In[29]:


accuracy = 174 / 255
print(accuracy)


# Put explanation here

# In[30]:


precision = round(65 / (65 + 61), 4)
print(f"precision = {precision}")


# In[31]:


recall = round(65 / (65 + 20), 4)
print(f"recall = {recall}")


# In[32]:


f_score = round((2 * (precision * recall)) / (precision + recall), 4)
print(f"f1-score = {f_score}")


# In[33]:


print(classification_report(y_test, y_pred))


# Put explanation here

# ***

# #### Q10

# In[34]:


predictions = pd.DataFrame({
    "income": [8.0, 8.0],
    "education": [7.0, 7.0],
    "parent": [0, 0],
    "married": [1, 1],
    "female": [1, 1],
    "age": [42, 82]
})
predictions.head()


# In[35]:


predictions["linkedin_prediction"] = lr.predict(predictions)


# In[37]:


print(predictions)

