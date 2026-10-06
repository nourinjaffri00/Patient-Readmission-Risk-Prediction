# Import libraries
import pandas as pd
import streamlit as st
from sklearn. model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Page setting 
st.set_page_config( 
    page_title= "Healthcare Patient Analytics",
    page_icon= "🩺",
    layout= "wide"
    )

# Importing Dataset
df= pd.read_csv("healthcare_patient_analytics.csv")
df['gender']= df['gender'].map(
    {
        "Male": 1,
        "Female":0
    }
)

# Choosing x features and target y 
feature= [
    "length_of_stay_days",
    "treatment_cost",
    "recovery_score",
    "gender"
]
x= df[feature]
y=df["readmission_risk"]

# train test split
x_train, x_test, y_train, y_test= train_test_split (x,y,test_size= 0.20, random_state=42,)

# train the linear regression model 
model= LinearRegression ()
model.fit(x_train,y_train)

# Test the model 
test_pred= model.predict (x_test)
mae= mean_absolute_error (y_test, test_pred)
r2= r2_score (y_test, test_pred)

# Prepare table used by app
test_outcomes= x_test.copy()
test_outcomes ["Actual"]= y_test
test_outcomes ["Predicted"]= test_pred.round(1)
test_outcomes ["Residual"]= (
    test_outcomes["Actual"]- test_outcomes["Predicted"]).round(1)
    
coeff_table= pd.DataFrame(
    {
        "feature": feature,
        "coefficients": model.coef_.round(4),
    }
)

#  streamlit interface
st.title("Patient Prediction")
st.write("Use input to predict")
st.info("Regression predicts a number. The target is readmission_risk")

# project summary
metric_1, metric_2, metric_3, metric_4= st.columns (4)
metric_1.metric("Data Rows", len(df))
metric_2.metric("Training Rows", len(x_train))
metric_3.metric("Test MAE", f"{mae:.2f} readmission_risk")
metric_4.metric("Test R2", f"{r2:.3f}")
st.caption (
    "For this dataset and split, MAE is about 23.55 orders and R2 is about 0.945"
)

# Tabs
predict_tab, insights_tab, data_tab= st.tabs(
    [
    "Predict readmission_risk",
    "Model_Inisghts", 
    "Explore Data"
    ]
)

# tab 1-Prediction
with predict_tab:
    st.subheader("create a new scenario!!")
    st.write("Choose patient details: ")

    with st.form("prediction_form"):
        left_col, right_col= st.columns (2)

        with left_col:
            length_of_stay_days=st.slider (
                "Length of Stays in Days", 
                min_value= int(df["length_of_stay_days"].min()),
                max_value= int(df["length_of_stay_days"].max()),
                value= 10,
                step = 1,
                help= "how much days stayed in hospital: ",
            )
            treatment_cost= st.slider(
                "Treatment Cost",
                min_value=int(df['treatment_cost'].min()),
                max_value=int(df["treatment_cost"].max()),
                value= 100000,
                step= 10000,
                help= "how much spent on treatment: ", 
            )
            
        with right_col:
            recovery_score= st.slider(
                "Recovery Score",
                min_value= int(df["recovery_score"].min()),
                max_value=int(df["recovery_score"].max()),
                value= 100,
                step=5,
                help="how much recovery score: "
            )
            gender=st.radio(
                "Gender",
                options= ["Male", "Female"],
                horizontal= True
            )
        predict_button= st.form_submit_button(
            "Predict Readmission Risk",
            type= "primary",
            use_container_width=True
        )
    if predict_button:
        gender= 1 if gender == "Male" else 0
        new_patient= pd.DataFrame(
            {
                "length_of_stay_days": [length_of_stay_days],
                "treatment_cost": [treatment_cost],
                "recovery_score": [recovery_score],
                "gender": [gender]
            }
        )
        # predict
        predicted_risk= model.predict(new_patient)[0]
        st.success ("Prediction Completed Successfully!!")
        st.metric(
            "Estimated Readmission_Risk",
            f"{round(predicted_risk):,} Risks",
        )
        st.caption (
            f"Raw Model Estimates: {predicted_risk:.2f} risk"
        )
        with st.expander ("See the Input Sent to Model"):
            st.dataframe (
                new_patient,
                use_container_width= True,
                hide_index= True
            )
        st.warning(
            "This is a model estimate based on available data"
        )
#  tab 2: Model Insights
with insights_tab:
    st.subheader ("Model Performance")
    score_1, score_2= st.columns (2)
    score_1.metric ("Mean Absolute Error", f"{mae:.2f} risk")
    score_2.metric("R2 Score", f"{r2:.3f}")

    st.write(
            "MAE prediction is about"
            f"{mae:.2f} risk away from actual value on average"
        )

    st.write(
        "R2: About"
        f"{r2*100:.1f}% risk away from actual value on average"
    )
st.divider ()

# intercept and coefficient
st.subheader ("What did linear regression learn? ")
st.metric ("Intercept", f"{model.intercept_:.2f}")
st.dataframe(
    coeff_table, 
    use_container_width= True,
    hide_index= True
)
st.bar_chart (
    coeff_table.set_index ("feature")
)
st.caption (
    "A coefficient describes the fitted relationship whil the other model"
    "features are held constant. It does not automatically prove causation"
)
st.divider()
st.subheader ("Actual vs. Predicted")

st.scatter_chart(
    test_outcomes,
    x="Actual",
    y="Predicted"
)
with st.expander ("Open the test-set prediction table"):
    st.dataframe(
        test_outcomes,
        use_container_width= True
    )
#  Exploring Dataset
with data_tab:
    st.subheader ("Explore the dataset")
    st.write(
        "Each row has historical record"
    )
    selected_feature= st.selectbox(
        "Choose feature to compare",
        options= feature
    )
    st.scatter_chart(
        df,
        x= selected_feature,
        y= "readmission_risk"
    )
    expand= st.checkbox("show the complete dataset")

    if expand:
        st.dataframe(
            df,
            use_container_width= True,
            hide_index= True
        )
    with st.expander("View Summary Statistics"):
        st.dataframe(
            df.describe(),
            use_container_width=True
        )



        

