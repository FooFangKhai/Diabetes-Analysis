import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import warnings
from scipy import stats
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from PIL import Image

df = pd.read_csv("/Users/_fangkhai/Downloads/Projects/Data Visualization/Diabetes-Analysis/diabetes.csv")

def page_home():
    st.title("Data Visualization 📊")
    st.write("The dataset comes from the Diabetes and Digestive and Kidney Disease National Institutes, focusing on women aged 21 and above of Pima Indian heritage. It includes medical factors like pregnancies, glucose, Blood Pressure, Skin Thickness, Diabetes Pedigree Function, BMI, insulin levels, and age, with the result showing if diabetes is present or not.")
    st.header("About 👤")
    col1, col2 = st.columns([1, 3.5])
    with col1:
        st.image("profile.jpg", width=200)
    with col2:
        st.write("Name: Foo Fang Khai")
        st.write("Student ID: 0134196")
        st.write("Email: 0134196@student.uow.edu.my")
    st.write("")
    st.header("Objective 🚀")
    st.write("Provide insights into factors affecting diabetes and utilize several machine learning algorithms to forecast diabetes.")
    st.image("get_started.gif", use_column_width = True)

def page_dataset():
    st.title("Dataset Observation and Processing")
    st.write("This page focuses on data processing and cleaning.")

    st.header("Pima Indians Diabetes Dataset 📂")
    st.write("The dataset can be obtain through Kaggle, which is available at https://www.kaggle.com/uciml/pima-indians-diabetes-database. ")
    st.write(df)
    st.write("This dataset consists of ",df.shape[0], " rows and ", df.shape[1], " columns")

    st.header("Data Observations 🔍")
    st.write(df.describe())
    st.write("It looks improbable that glucose, blood pressure, skin thickness, insulin, and BMI levels would be zero. In addition, insulin value of 846 are considered implausible, suggesting the presence of outliers. Hence, zero values for these parameters will be replaced by their respective means except for Pregnancies and z-score values will be used to remove the outliers.")

    df1 = df.copy()
    cols = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
    for i in cols:
        df1[i].replace(0,df1[i].mean(),inplace=True)
    z = np.abs(stats.zscore(df1))
    df1 = df1[(z < 3).all(axis=1)]

    st.header("Final Cleaned Dataset 🧹")
    st.write(df1.describe())
    st.write("This dataset consists of ",df1.shape[0], " rows and ", df1.shape[1], " columns")

    st.header("Initial Inspections 📋")
    col = list(df1.columns)
    fig, ax = plt.subplots(figsize=(12, 10), nrows=3, ncols=3)
    for i, column in enumerate(col):
        row = i // 3
        col_index = i % 3
        df1[column].hist(ax=ax[row, col_index], bins=20)
        ax[row, col_index].tick_params(axis='x', colors='white')
        ax[row, col_index].tick_params(axis='y', colors='white')
        ax[row, col_index].set_facecolor('none')

        ax[row, col_index].set_title(column, color='white')
        ax[row, col_index].grid(False)

    plt.tight_layout()
    fig.patch.set_facecolor('none')
    st.pyplot(fig)

def feature_selection():
    df1 = df.copy()
    cols = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
    for i in cols:
        df1[i].replace(0,df1[i].mean(),inplace=True)
    z = np.abs(stats.zscore(df1))
    df1 = df1[(z < 3).all(axis=1)]

    st.title("Features Relation")
    st.write("This page focuses on identifying features that have the strongest correlation with our target variable (outcome).")
    st.write("")

    # Using RandomForestClassifier
    st.header("Feature Importance Using RandomForestClassifier")
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    feature_importances = clf.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=True)
    colors = sns.color_palette('viridis', len(feature_importance_df))
    plt.figure(figsize=(16, 8)).set_facecolor('none')
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color=colors)
    plt.xlabel('Importance', color='white')
    plt.ylabel('Feature', color='white') 
    plt.title('Feature Importance', color='white')
    plt.tick_params(axis='x', colors='white')
    plt.tick_params(axis='y', colors='white') 
    plt.gca().set_facecolor('none')
    st.pyplot(plt.gcf())
    st.write("")

    # Using Heatmap
    st.header("Feature Importance Using Heatmap")
    corr = df1.corr()
    plt.figure(figsize = (18, 10))
    heatmap = sns.heatmap(corr, annot = True)
    plt.gcf().set_facecolor('none')
    heatmap_fig = plt.gcf()
    cbar = heatmap.collections[0].colorbar
    cbar.ax.yaxis.set_tick_params(color='white')
    for tick_label in cbar.ax.yaxis.get_ticklabels():
        tick_label.set_color('white')
    plt.tick_params(axis='x', colors='white')
    plt.tick_params(axis='y', colors='white') 
    st.pyplot(heatmap_fig)

    # Pairplot
    st.header("Features Correlation Using Pairplot")
    pairplot = sns.pairplot(df1, diag_kind='kde')
    for ax in pairplot.axes.flatten():
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
    pairplot.fig.set_facecolor("none")
    st.pyplot(pairplot)

    st.write("")
    st.header("Features Correlation Between Outcome Using Pairplot")
    df_cr = df1.copy()
    df_cr['Outcome'].astype('category')
    df_cr['Outcome'].replace(0, "Non-diabetic", inplace = True)
    df_cr['Outcome'].replace(1, "Diabetic", inplace = True)
    pairplots = sns.pairplot(df_cr, hue = 'Outcome', diag_kind = 'kde')
    for ax in pairplots.axes.flatten():
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
    pairplots._legend.get_title().set_color('white')
    pairplots._legend.get_texts()[0].set_color('white')
    pairplots._legend.get_texts()[1].set_color('white') 
    pairplots.fig.set_facecolor("none")
    st.pyplot(pairplots)

    # Hypothesis Testing
    st.header("Hypothesis Testing Using Pearson correlation coefficient")
    st.write("Null Hypothesis: Both sets of feature are uncorrelated.\n"
             "\nAlternative Hypothesis: Both sets of feature are somewhat correlated.")
    st.write("")
    independent_variable = st.selectbox("Select Independent Variable", ['Glucose', 'BMI', 'Age', 'DiabetesPedigreeFunction', 'BloodPressure', 'Pregnancies', 'Insulin', 'SkinThickness'])
    dependent_variable = 'Outcome'

    alpha = 0.05
    def hypothesis_test(p_value):
        if p_value < alpha:
            return "The p-value is {}, which is lower than the significance level (95%). Alternative hypothesis is accepted.".format(p_value)
        else:
            return "The p-value is {}, which is higher than the significance level (95%). Null hypothesis is not rejected.".format(p_value)

    s, p = stats.pearsonr(df1[independent_variable], df1[dependent_variable])
    st.write(hypothesis_test(p))
    st.write("The correlation coefficient between {} and the Dependent Variable is: {:.4f}".format(independent_variable, s))

def distribution_outcome():
    st.title("Distribution of Outcome")
    st.write("This page focuses on the distribution of Outcome, divided into two parts: dataset with diabetes and dataset without diabetes, each showing the distribution for Outcome 0 (no diabetes) and 1 (diabetes) respectively.")

    df_diabetes = df[df['Outcome']==1]
    df_withoutdiabetes = df[df['Outcome']==0]

    selected_dataset = st.selectbox("Select Dataset", ["Diabetes", "Without Diabetes"])
    if selected_dataset == "Diabetes":
        st.subheader("Histograms for People Who Had Positive Diabetes Findings")
        plot_data = df_diabetes
    else:
        st.subheader("Histograms for People Who Had Negative Diabetes Findings")
        plot_data = df_withoutdiabetes

    fig, ax = plt.subplots(4, 2, figsize=(20, 25), facecolor='none')
    sns.set(style="white", palette="muted")
    for axis in ax.flatten():
        axis.set_facecolor('none')
        axis.tick_params(axis='x', colors='white')
        axis.tick_params(axis='y', colors='white')
        axis.title.set_color('white')
        axis.xaxis.label.set_color('white')
        axis.yaxis.label.set_color('white')

    sns.distplot(plot_data.Age, bins=10, color='skyblue', ax=ax[0, 0])
    sns.distplot(plot_data.Pregnancies, bins=10, color='salmon', ax=ax[0, 1])
    sns.distplot(plot_data.Glucose, bins=10, color='lightgreen', ax=ax[1, 0])
    sns.distplot(plot_data.BloodPressure, bins=10, color='lightcoral', ax=ax[1, 1])
    sns.distplot(plot_data.SkinThickness, bins=10, color='cornflowerblue', ax=ax[2, 0])
    sns.distplot(plot_data.Insulin, bins=10, color='mediumorchid', ax=ax[2, 1])
    sns.distplot(plot_data.DiabetesPedigreeFunction, bins=10, color='lightpink', ax=ax[3, 0])
    sns.distplot(plot_data.BMI, bins=10, color='lightseagreen', ax=ax[3, 1])

    st.pyplot(fig)

def in_depth_analysis():
    st.title("In-depth Analysis")
    st.write("This page focuses on analysing several features that is correlated to our target variable (outcome).")
    df_diabetes = df[df['Outcome']==1]
    df_withoutdiabetes = df[df['Outcome']==0]

    df1 = df.copy()
    cols = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
    for i in cols:
        df1[i].replace(0,df1[i].mean(),inplace=True)
    z = np.abs(stats.zscore(df1))
    df1 = df1[(z < 3).all(axis=1)]

    dropdown_placeholder = st.empty()
    hovering = dropdown_placeholder.hovered
    if hovering:
        selected_variable = dropdown_placeholder.selectbox("Select Variable", ["Glucose", "Age", "Insulin", "SkinThickness", "Pregnancies"])
        if selected_variable == "Glucose":
            st.write("")
            st.title("Glucose Vs Outcome")
            st.write("**Glucose by Frequency (Patients with Diabetes)**", size = 20)
            fig, ax = plt.subplots(figsize=(20, 12))
            ax.hist(df_diabetes['Glucose'], histtype='stepfilled', bins=20, color='lightblue', density=True)
            sns.kdeplot(df_diabetes['Glucose'], color='mediumorchid').set_facecolor("none")
            ax.set_ylabel('Frequency', size=20)
            ax.set_xlabel('Glucose Levels', size=20)
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')
            ax.title.set_color('white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.grid(axis="y")
            fig.set_facecolor("none")
            st.pyplot(fig)  

            st.write("")
            st.write("**Glucose by Frequency (Patients without Diabetes)**", size = 20)
            fig, ax = plt.subplots(figsize=(20, 12))
            ax.hist(df_withoutdiabetes['Glucose'], histtype='stepfilled', bins=20, color='lightblue', density=True)
            sns.kdeplot(df_withoutdiabetes['Glucose'], color='mediumorchid').set_facecolor("none")
            ax.set_ylabel('Frequency', size=20)
            ax.set_xlabel('Glucose Levels', size=20)
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')
            ax.title.set_color('white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.grid(axis="y")
            fig.set_facecolor("none")
            st.pyplot(fig)

        elif selected_variable == "Age":
            st.write("")
            bins = [18, 30, 40, 50, 60, 70, 120]
            labels = ['21-29', '30-39', '40-49', '50-59', '60-69', '70+']

            df1['agerange'] = pd.cut(df1.Age, bins, labels=labels, include_lowest=True)
            df_diabetes['agerange'] = pd.cut(df_diabetes.Age, bins, labels=labels, include_lowest=True)
            df_withoutdiabetes['agerange'] = pd.cut(df_withoutdiabetes.Age, bins, labels=labels, include_lowest=True)

            age_groups_positive = df_diabetes.groupby('agerange').size().reset_index(name='positive_frequency')
            age_groups_negative = df_withoutdiabetes.groupby('agerange').size().reset_index(name='negative_frequency')
            age_frame_combined = pd.merge(age_groups_positive, age_groups_negative, on='agerange', how='outer')

            st.title('Age Groups vs Outcome')
            st.write("**Age Groups vs Frequency**", size = 20)
            fig, ax = plt.subplots(figsize=(15, 10))
            age_frame_combined.plot.barh(x='agerange', y=['positive_frequency', 'negative_frequency'], ax=ax, color=['red', 'blue'])
            ax.set_facecolor('none')  
            fig.set_facecolor('none')
            ax.tick_params(axis='x', colors='white')  
            ax.tick_params(axis='y', colors='white')  
            ax.xaxis.label.set_color('white') 
            ax.yaxis.label.set_color('white')
            ax.grid(axis='y')
            ax.set_xlabel('Frequency', color='white')
            st.pyplot(fig)

            st.write("")
            st.write('**Ratio of Positive and Negative Frequency by Age Group**')
            age_groups = df1.groupby('agerange').size().reset_index(name='frequency')
            age_groups_positive = df_diabetes.groupby('agerange').size().reset_index(name='positive_frequency')

            age_frame = pd.merge(age_groups, age_groups_positive, on='agerange', how='left')
            age_frame['ratio'] = age_frame['positive_frequency'] / age_frame['frequency']

            age_groups_2 = df1.groupby('agerange').size().reset_index(name='frequency')
            age_groups_negative = df_withoutdiabetes.groupby('agerange').size().reset_index(name='negative_frequency')

            age_frame_2 = pd.merge(age_groups_2, age_groups_negative, on='agerange', how='left')
            age_frame_2['ratio'] = age_frame_2['negative_frequency'] / age_frame_2['frequency']

            age_frame['symmetrical_ratio'] = (age_frame['ratio'] + age_frame_2['ratio']) / 2
            
            plt.figure(figsize=(10, 6))
            plt.plot(age_frame['agerange'], age_frame['ratio'], marker='o', color='blue', label='Positive Frequency')
            plt.plot(age_frame_2['agerange'], age_frame_2['ratio'], marker='o', color='red', label='Negative Frequency')
            plt.plot(age_frame['agerange'], age_frame['symmetrical_ratio'], linestyle='--', color='green', label='Midpoint')
            plt.xlabel('Age Group', color='white')
            plt.ylabel('Ratio of Frequency', color='white')
            plt.xticks(rotation=45, color='white')
            plt.yticks(color='white')
            plt.legend()
            plt.grid(axis="y")
            plt.gca().set_facecolor('none')
            plt.gcf().set_facecolor('none')
            st.pyplot(plt) 

        elif selected_variable == "Insulin":
            st.write("")
            df_positive = df1[(df1['Outcome'] == 1)]
            df_negative = df1[(df1['Outcome'] == 0)]

            st.title('Insulin vs Outcome')
            st.write('**Distribution of Insulin Levels for People with and without Diabetes**')
            
            df_combined = pd.concat([df_positive.assign(Diabetes='Diabetes'), 
                         df_negative.assign(Diabetes='No Diabetes')])
            plt.figure(figsize=(12, 8))
            sns.swarmplot(x='Diabetes', y='Insulin', data=df_combined, palette="muted")

            plt.xlabel('Diabetes Status', color='white')
            plt.ylabel('Insulin Level', color='white')
            plt.gca().set_facecolor('none')
            plt.gcf().set_facecolor('none')
            plt.tick_params(axis='x', colors='white')
            plt.tick_params(axis='y', colors='white') 
            st.pyplot(plt)
        
        elif selected_variable == "SkinThickness":
            st.write("")
            st.title("Skin Thickness Vs Outcome")
            st.write("**Box Plot for Skin Thickness by Outcome**")
            df1['Outcome'] = df1['Outcome'].map({0: 'No Diabetes', 1: 'Diabetes'})

            plt.figure(figsize=(16, 10))
            sns.boxplot(x="Outcome", y="SkinThickness", data=df1)
            plt.ylabel('Skin Thickness', size=15, color='white')
            plt.xlabel('Outcome', size=15, color='white')
            plt.xticks(size=12, rotation=45, color='white')
            plt.yticks(size=12, color='white')
            plt.gcf().set_facecolor('none')
            plt.grid(axis="y")
            st.pyplot(plt)
        
        elif selected_variable == "Pregnancies":
            df_positive = df1[(df1['Outcome'] == 1)]
            df_negative = df1[(df1['Outcome'] == 0)]
            st.write("")
            st.title("Pregnancies Vs Positive Outcome")
            st.write("**Box Plot for Skin Thickness by Outcome**")
            fig, ax = plt.subplots(figsize=(18, 10))

            pregnancy_groups = df1.groupby('Pregnancies')
            df_pregnancies = pd.DataFrame(pregnancy_groups.size()).reset_index()
            df_pregnancies.columns = ['Pregnancies', 'Overall']

            pregnancy_groups_positive = df_positive.groupby('Pregnancies')
            df_pregnancies_positive = pd.DataFrame(pregnancy_groups_positive.size()).reset_index()
            df_pregnancies_positive.columns = ['Pregnancies', 'Positive']

            df_merged = pd.merge(df_pregnancies, df_pregnancies_positive, on='Pregnancies', how='left')
            df_merged['Positive'] = df_merged['Positive'].fillna(0)

            df_merged['Diabetes'] = df_merged['Positive'] / df_merged['Overall']

            ax.set_title('Positive Pregnancies by Pregnancy Count')
            df_merged.plot(x='Pregnancies', y='Diabetes', ax=ax, marker='o')

            # Customize plot properties
            plt.xlabel('Number of Pregnancies', color = "white")
            plt.ylabel('Ratio of Positive Pregnancies to Overall Pregnancies', color = "white")
            plt.grid(axis="y")

            midpoint = df_merged['Diabetes'].mean()

            plt.axhline(y=midpoint, color='green', linestyle='--', label='Midpoint')
            plt.legend()
            ax.set_facecolor('none')
            fig.set_facecolor('none')
            plt.xticks(color='white')
            plt.yticks(color='white')
            st.pyplot(fig)

def prediction():
    st.title("Prediction")
    st.write("This page focuses on utilising several machine learning algorithms that gives us the best accuracy in predicting the outcome.")
    st.write("")
    st.header("Summary of Results")
    st.image("Results.png")

    X = df.drop(['Outcome'], axis=1)
    Y = df['Outcome']

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=60)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    def conf_mtx(y_act, y_pred):
        cm = metrics.confusion_matrix(y_act, y_pred, labels=[1, 0])
        df_cm = pd.DataFrame(cm, index=["Diabetic", "Non-Diabetic"],
                            columns=["Predict for Diabetic", "Predict for Non-Diabetic"])
        fig, ax = plt.subplots(figsize=(14, 4))
        sns.heatmap(df_cm, annot=True, fmt='g')
        st.pyplot(fig)
        Score_Accuracy = "%.2f%%" % (metrics.accuracy_score(y_act, y_pred) * 100)
        Score_Recall = "%.2f%%" % (metrics.recall_score(y_act, y_pred) * 100)
        Score_Precision = "%.2f%%" % (metrics.precision_score(y_act, y_pred) * 100)

        print("Model Accuracy Score: " + Score_Accuracy)
        print("Model Recall Score: " + Score_Recall)
        print("Model Precision Score: " + Score_Precision)

        return Score_Accuracy, Score_Recall, Score_Precision
    
    def ML_test(Mdl, Param_grid):
        if bool(Param_grid):
            Mdl = GridSearchCV(Mdl, Param_grid, cv=10)
            Mdl.fit(X_train_scaled, y_train)
            Mdl_params = Mdl.best_params_
            Mdl_train_sc = Mdl.cv_results_['mean_test_score'].mean()
            Mdl_test_sc = Mdl.score(X_test_scaled, y_test)
            probas = Mdl.predict_proba(X_test_scaled)

        else:
            Mdl = Mdl
            Mdl.fit(X_train_scaled, y_train)
            Mdl_train_sc = round(Mdl.score(X_train_scaled, y_train), 4)
            Mdl_test_sc = round(Mdl.score(X_test_scaled, y_test), 4)
            probas = Mdl.predict_proba(X_test_scaled)

        y_pred = Mdl.predict(X_test_scaled)

        print("Training score is: " + str(Mdl_train_sc))
        print("Test Mean score is: " + str(Mdl_test_sc))

        Score_Accuracy, Score_Recall, Score_Precision = conf_mtx(y_test, y_pred)
        Mdl_train_sc = "%.2f%%" % (Mdl_train_sc * 100)

        fpr, tpr, thresholds = roc_curve(y_test, probas[:, 1])
        roc_auc = round(auc(fpr, tpr), 4)
        print("AUC : " + str(roc_auc))

        return Mdl_train_sc, Score_Accuracy, Score_Recall, Score_Precision, roc_auc
    
    st.subheader("Statistics For Chosen Model")
    selected_model = st.selectbox("Select One Model", ["Logistic Regression", "Gaussian Naive Bayes", "K Neighbours Classifier", "Support Vector Machine","Decision Tree", "Random Forest", "Bagging Classifier", "AdaBoost Classifier", "GradientBoosting Classifier", "Linear Discriminant Analysis"])
    if selected_model == "Logistic Regression":
        Mdl_LogReg = LogisticRegression(solver="liblinear")
        Param_grid_LogReg = {'penalty': ['l1', 'l2'], 'C': np.linspace(0.1, 1.1, 10)}
        Mdl_train_sc, Score_Accuracy, Score_Recall, Score_Precision, roc_auc = ML_test(Mdl_LogReg, Param_grid_LogReg)
        st.write('**Model Results For Logistic Regression**')
        st.write('Training Accuracy:', Mdl_train_sc)
        st.write('Test Accuracy Score:', Score_Accuracy)
        st.write('Test Recall Score:', Score_Recall)
        st.write('Test Precision Score:', Score_Precision)
        st.write('AUC Score:', roc_auc)
    elif selected_model == "Gaussian Naive Bayes":
        Mdl = GaussianNB()
        Mdl_train_sc, Score_Accuracy, Score_Recall, Score_Precision, roc_auc = ML_test(Mdl,Param_grid={})
        st.write('**Model Results For Gaussian Naive Bayes**')
        st.write('Training Accuracy:', Mdl_train_sc)
        st.write('Test Accuracy Score:', Score_Accuracy)
        st.write('Test Recall Score:', Score_Recall)
        st.write('Test Precision Score:', Score_Precision)
        st.write('AUC Score:', roc_auc)
    elif selected_model == "K Neighbours Classifier":
        Mdl_kNeigh = KNeighborsClassifier()
        Param_grid_kNeigh = {'n_neighbors': list(np.arange(3, 8)), 'metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski']}
        Mdl_train_sc, Score_Accuracy, Score_Recall, Score_Precision, roc_auc = ML_test(Mdl_kNeigh, Param_grid_kNeigh)
        st.write('**Model Results For K Neighbours Classifier**')
        st.write('Training Accuracy:', Mdl_train_sc)
        st.write('Test Accuracy Score:', Score_Accuracy)
        st.write('Test Recall Score:', Score_Recall)
        st.write('Test Precision Score:', Score_Precision)
        st.write('AUC Score:', roc_auc)
    elif selected_model == "Support Vector Machine":
        Mdl = SVC(probability=True)
        Param_grid_SVC =  {'C': np.linspace(0.1,1.1,10), 'kernel': ['linear','poly','rbf',]}
        Mdl_train_sc, Score_Accuracy, Score_Recall, Score_Precision, roc_auc = ML_test(Mdl,Param_grid_SVC)
        st.write('**Model Results For Support Vector Machine**')
        st.write('Training Accuracy:', Mdl_train_sc)
        st.write('Test Accuracy Score:', Score_Accuracy)
        st.write('Test Recall Score:', Score_Recall)
        st.write('Test Precision Score:', Score_Precision)
        st.write('AUC Score:', roc_auc)
    elif selected_model == "Decision Tree":
        Mdl = DecisionTreeClassifier(random_state=1)
        Param_grid_dt = {'criterion':['gini','entropy'],'max_depth': [3, 4, 5, 6, 7, 8],
                    'min_impurity_decrease': [0.0001, 0.0003, 0.0005, 0.0007, 0.009]}
        Mdl_train_sc, Score_Accuracy, Score_Recall, Score_Precision, roc_auc = ML_test(Mdl,Param_grid_dt)
        st.write('**Model Results For Decision Tree**')
        st.write('Training Accuracy:', Mdl_train_sc)
        st.write('Test Accuracy Score:', Score_Accuracy)
        st.write('Test Recall Score:', Score_Recall)
        st.write('Test Precision Score:', Score_Precision)
        st.write('AUC Score:', roc_auc)
    elif selected_model == "Random Forest":
        Mdl = RandomForestClassifier(random_state=1,n_estimators=100)
        Param_grid_rf = {'criterion':['gini','entropy'],'max_depth': [3, 4, 5, 6, 7, 8],
                    'min_impurity_decrease': [0.0001, 0.0003, 0.0005, 0.0007, 0.009]}
        Mdl_train_sc, Score_Accuracy, Score_Recall, Score_Precision, roc_auc = ML_test(Mdl,Param_grid_rf)
        st.write('**Model Results For Random Forest**')
        st.write('Training Accuracy:', Mdl_train_sc)
        st.write('Test Accuracy Score:', Score_Accuracy)
        st.write('Test Recall Score:', Score_Recall)
        st.write('Test Precision Score:', Score_Precision)
        st.write('AUC Score:', roc_auc)
    elif selected_model == "Bagging Classifier":
        Mdl = BaggingClassifier(n_estimators=100, bootstrap=True)
        Param_grid_bc = {'max_samples': list(np.arange(0.1,1.1,0.1))}
        Mdl_train_sc, Score_Accuracy, Score_Recall, Score_Precision, roc_auc = ML_test(Mdl,Param_grid_bc)
        st.write('**Model Results For Bagging Classifier**')
        st.write('Training Accuracy:', Mdl_train_sc)
        st.write('Test Accuracy Score:', Score_Accuracy)
        st.write('Test Recall Score:', Score_Recall)
        st.write('Test Precision Score:', Score_Precision)
        st.write('AUC Score:', roc_auc)
    elif selected_model == "AdaBoost Classifier":
        Mdl = AdaBoostClassifier( n_estimators= 100)
        Param_grid_abc = {'learning_rate': list(np.arange(0.1,1.1,0.1))}
        Mdl_train_sc, Score_Accuracy, Score_Recall, Score_Precision, roc_auc = ML_test(Mdl,Param_grid_abc)
        st.write('**Model Results For AdaBoost Classifier**')
        st.write('Training Accuracy:', Mdl_train_sc)
        st.write('Test Accuracy Score:', Score_Accuracy)
        st.write('Test Recall Score:', Score_Recall)
        st.write('Test Precision Score:', Score_Precision)
        st.write('AUC Score:', roc_auc)
    elif selected_model == "GradientBoosting Classifier":
        Mdl = GradientBoostingClassifier(n_estimators= 100, learning_rate=0.1)
        Param_grid_gbc = {'max_depth': [3, 4, 5, 6, 7, 8],'min_impurity_decrease': [0.0001, 0.0003, 0.0005, 0.0007, 0.009]}
        Mdl_train_sc, Score_Accuracy, Score_Recall, Score_Precision, roc_auc = ML_test(Mdl,Param_grid_gbc)
        st.write('**Model Results For GradientBoosting Classifier**')
        st.write('Training Accuracy:', Mdl_train_sc)
        st.write('Test Accuracy Score:', Score_Accuracy)
        st.write('Test Recall Score:', Score_Recall)
        st.write('Test Precision Score:', Score_Precision)
        st.write('AUC Score:', roc_auc)
    elif selected_model == "Linear Discriminant Analysis":
        Mdl = LinearDiscriminantAnalysis()
        param_grid_lda = {}
        Mdl_train_sc, Score_Accuracy, Score_Recall, Score_Precision, roc_auc = ML_test(Mdl, param_grid_lda)
        st.write('**Model Results For Linear Discriminant Analysis**')
        st.write('Training Accuracy:', Mdl_train_sc)
        st.write('Test Accuracy Score:', Score_Accuracy)
        st.write('Test Recall Score:', Score_Recall)
        st.write('Test Precision Score:', Score_Precision)
        st.write('AUC Score:', roc_auc)

def conclusion():
    st.title("Conclusion")
    st.write("This page focuses on the overall discoveries regarding the factors on affecting diabetes and identifies the model that achieves the highest accuracy.")

    st.header("Overall Findings That Cause Diabetes")
    st.write("**Glucose Level (mg/dL):**\n"
             "\n    1. 120 - 130"
             "\n    2. 140 - 150")
    st.write("**Age Group:** \n"
             "\n    1. 50 - 59 (Highest)\n"
             "\n    2. 40 - 49 (Second Highest)")
    st.write("**Insulin:**\n"
             "\n    It doesn't exert a significant influence on diabetes.")
    st.image("Findings1.png")
    st.write("**Skin Thickness:**\n"
             "\n    Individuals with diabetes may exhibit a slightly thicker skin.")
    st.image("Findings2.png")
    st.write("**Pregnancies:**\n"
             "\n    1. Women with more than six pregnancies are more likely to develop diabetes.\n"
             "\n    2. The most significant risk occurs in women with nine pregnancies, followed by those with eleven.")

def main():
    st.set_page_config(layout="wide")
    st.sidebar.title("Diabetes Analysis")
    page_options = {
        "Home": "🏠",
        "Dataset Analysis": "💡",
        "Features Relation": "🛠️",
        "Distribution Based on Outcome": "🔀",
        "In-depth Analysis": "🕵️",
        "Prediction": "🔮",
        "Conclusion": "✨",
    }

    selected_page = st.sidebar.selectbox("", list(page_options.keys()), format_func=lambda x: f"{page_options[x]} {x}")

    if selected_page == "Home":
        st.sidebar.image("logo.gif", use_column_width=True)
        page_home()
    elif selected_page == "Dataset Analysis":
        st.sidebar.image("datacleaning.gif", use_column_width=True)
        page_dataset()
    elif selected_page == "Features Relation":
        st.sidebar.image("featureselection.gif", use_column_width=True)
        feature_selection()
    elif selected_page == "Distribution Based on Outcome":
        st.sidebar.image("distribution.gif", use_column_width=True)
        distribution_outcome()
    elif selected_page == "In-depth Analysis":
        st.sidebar.image("indepth.gif", use_column_width=True)
        in_depth_analysis()
    elif selected_page == "Prediction":
        st.sidebar.image("prediction.gif", use_column_width=True)
        prediction()
    elif selected_page == "Conclusion":
        st.sidebar.image("conclusion.gif", use_column_width=True)
        conclusion()

if __name__ == "__main__":
    main()
