import streamlit as st
import joblib
import pandas as pd

st.set_page_config(
    page_title="Malaysia House Price Analytics",
    page_icon="🏠",
    layout="centered"
)

@st.cache_data
def load_data():
    return pd.read_csv("malaysia_house_price_data_2025_cleaned.csv")

df = load_data()
states = sorted(df["State"].dropna().unique().tolist())
tenures = sorted(df["Tenure"].dropna().unique().tolist())
property_types = sorted(df["Type"].dropna().unique().tolist())

@st.cache_resource
def load_models():
    return joblib.load("malaysia_house_price_models.joblib")

model_bundle = load_models()
preprocessor = model_bundle["preprocessor"]
models = model_bundle["models"]
metrics = model_bundle["metrics"]

st.markdown("""
<style>
.stApp {
    background-color: #0f172a;
}

.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp p, .stApp label, .stApp span {
    color: #f8fafc;
}

.main-title {
    font-size: 44px;
    font-weight: 800;
    text-align: center;
    margin: 15px auto 12px auto;
    background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-description {
    max-width: 800px;
    margin: 0 auto 30px auto;
    text-align: center;
    font-size: 16px;
    line-height: 1.7;
    color: #cbd5e1;
}

div.stButton > button {
    background-color: #10b981;
    color: #ffffff;
    border: none;
    border-radius: 12px;
    font-size: 16px;
    font-weight: 700;
    padding: 12px 24px;
    box-shadow: 0 4px 14px rgba(16, 185, 129, 0.35);
    transition: all 0.2s ease-in-out;
}

div.stButton > button:hover {
    background-color: #059669;
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(16, 185, 129, 0.5);
}


hr {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, rgba(15,23,42,0), #38bdf8, rgba(15,23,42,0));
    margin: 25px 0;
    opacity: 0.7;
}
</style>

<div class="main-title">Malaysia House Price Analytics</div>
<div class="hero-description">
    This application analyses and predicts the median property price of residential properties in Malaysia using machine learning regression models.
    <br><br>
    Explore the data, compare model performance, and obtain an estimated median price based on your selected inputs.
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(
    [
        "🏠 Price Predictor",
        "📊 Model Evaluation",
        "📈 Exploratory Data Analysis"
    ]
)

with tab1:
    # =========================================================
    # MODEL SELECTION
    # =========================================================
    
    st.header("Select Machine Learning Model")
    
    selected_model = st.selectbox(
        "Choose a model:",
        [
            "Linear Regression",
            "Decision Tree",
            "Random Forest",
            "Gradient Boosting"
        ]
    )
    
    st.info(
        f"Selected model: **{selected_model}**"
    )
    
    
    # =========================================================
    # PROPERTY INFORMATION
    # =========================================================
    
    st.header("🏡 Property Information")
    
    st.write(
        "Please provide the following property details:"
    )
    
    
    # =========================================================
    # STATE AND AREA RELATIONSHIP
    # =========================================================
    
    # Initial State
    if "selected_state" not in st.session_state:
        st.session_state.selected_state = states[0]
    
    
    # Get areas for the initial state
    initial_areas = sorted(
        df.loc[
            df["State"] == st.session_state.selected_state,
            "Area"
        ].dropna().unique().tolist()
    )
    
    
    # Initial Area
    if "selected_area" not in st.session_state:
        if initial_areas:
            st.session_state.selected_area = initial_areas[0]
        else:
            st.session_state.selected_area = ""
    
    
    # ---------------------------------------------------------
    # When State changes, update Area
    # ---------------------------------------------------------
    
    def update_area_from_state():
    
        selected_state = st.session_state.selected_state
    
        available_areas = sorted(
            df.loc[
                df["State"] == selected_state,
                "Area"
            ].dropna().unique().tolist()
        )
    
        if available_areas:
            st.session_state.selected_area = available_areas[0]
    
    
    # ---------------------------------------------------------
    # When Area changes, update State
    # ---------------------------------------------------------
    
    def update_state_from_area():
    
        selected_area = st.session_state.selected_area
    
        matching_states = (
            df.loc[
                df["Area"] == selected_area,
                "State"
            ]
            .dropna()
            .unique()
            .tolist()
        )
    
        if matching_states:
            st.session_state.selected_state = matching_states[0]
    
    
    # =========================================================
    # STATE DROPDOWN
    # =========================================================
    
    state = st.selectbox(
        "State",
        states,
        key="selected_state",
        on_change=update_area_from_state
    )
    
    
    # =========================================================
    # AREA DROPDOWN
    # =========================================================
    
    filtered_areas = sorted(
        df.loc[
            df["State"] == state,
            "Area"
        ].dropna().unique().tolist()
    )
    
    
    # Make sure selected Area belongs to selected State
    if filtered_areas:
    
        if st.session_state.selected_area not in filtered_areas:
            st.session_state.selected_area = filtered_areas[0]
    
    else:
    
        st.session_state.selected_area = ""
    
    
    area = st.selectbox(
        "Area",
        filtered_areas,
        key="selected_area",
        on_change=update_state_from_area
    )
    
    
    # =========================================================
    # TENURE
    # =========================================================
    
    tenure = st.selectbox(
        "Tenure",
        tenures
    )
    
    
    # =========================================================
    # PROPERTY TYPE
    # =========================================================
    
    property_type = st.selectbox(
        "Property Type",
        property_types
    )
    
    
    # =========================================================
    # MEDIAN PSF
    # =========================================================
    
    median_psf = st.number_input(
        "Median Price Per Square Foot (RM)",
        min_value=38.0,
        max_value=1045.0,
        value=300.0,
        step=1.0
    )
    
    
    # =========================================================
    # TRANSACTIONS
    # =========================================================
    
    transactions = st.number_input(
        "Number of Transactions",
        min_value=10,
        max_value=76,
        value=20,
        step=1
    )
    
    
    # =========================================================
    # PREDICTION
    # =========================================================
    
    st.divider()
    
    st.header("💰 Price Prediction")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        predict_button = st.button(
            "💰 Predict Median Price",
            use_container_width=True
        )
    
    if predict_button:
    
        # -----------------------------------------------------
        # Create input data
        # -----------------------------------------------------
    
        input_data = pd.DataFrame({
            "Area": [area],
            "State": [state],
            "Tenure": [tenure],
            "Type": [property_type],
            "Median_PSF": [median_psf],
            "Transactions": [transactions]
        })
    
    
        # -----------------------------------------------------
        # Get selected model
        # -----------------------------------------------------
        
        model = models[selected_model]
        
        
        # -----------------------------------------------------
        # Apply preprocessing
        # -----------------------------------------------------
        
        input_processed = preprocessor.transform(input_data)
        
        
        # -----------------------------------------------------
        # Make prediction
        # -----------------------------------------------------
        
        prediction = model.predict(input_processed)[0]
        
    
        # -----------------------------------------------------
        # Display selected model
        # -----------------------------------------------------
    
        st.write(
            f"**Model used:** {selected_model}"
        )
        
        # -----------------------------------------------------
        # Display prediction
        # -----------------------------------------------------

        st.success(
            f"Estimated Median Property Price: "
            f"RM {prediction:,.2f}"
        )

# =========================================================
# TAB 2 - MODEL EVALUATION
# =========================================================

with tab2:

    st.header("📊 Model Evaluation")

    st.write(
        "Performance comparison and diagnostic analysis "
        "for all four trained regression models."
    )

    # ---------------------------------------------------------
    # MODEL METRICS
    # ---------------------------------------------------------

    st.subheader("Model Performance")

    try:

        metric_rows = []

        if isinstance(metrics, dict):

            for model_name, model_metrics in metrics.items():

                if isinstance(model_metrics, dict):

                    row = {
                        "Model": model_name
                    }

                    for metric_name, value in model_metrics.items():

                        if isinstance(value, (int, float)):

                            row[metric_name] = value

                    metric_rows.append(row)

        if metric_rows:

            metrics_df = pd.DataFrame(metric_rows)

            st.dataframe(
                metrics_df,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "Model metric values are available in the trained "
                "model bundle but could not be displayed as a table."
            )

    except Exception:

        st.info(
            "Model metrics could not be displayed."
        )

    # ---------------------------------------------------------
    # MODEL PERFORMANCE COMPARISON
    # ---------------------------------------------------------

    st.subheader("Model Performance Comparison")

    st.image(
        "final_benchmark.png",
        use_container_width=True
    )

    st.caption(
        "Figure: Comparison of the performance of the four "
        "tuned machine-learning regression models with untuned baseline model."
    )

    # ---------------------------------------------------------
    # ACTUAL VS PREDICTED
    # ---------------------------------------------------------

    st.subheader("Overfitting check")

    st.image(
        "overfitting_comparison.png",
        use_container_width=True
    )

    st.caption(
        "Figure: Overfitting check of Train and Test R² Score "
        "for all four regression models."
    )

    # ---------------------------------------------------------
    # RESIDUAL ANALYSIS
    # ---------------------------------------------------------

    st.subheader("Residual Analysis")

    st.image(
        "residual_plot_tuned_gradient_boosting.png",
        use_container_width=True
    )

    st.caption(
        "Figure: Residual analysis used to examine prediction errors "
        "and model behaviour."
    )

    # ---------------------------------------------------------
    # RANDOM FOREST FEATURE IMPORTANCE
    # ---------------------------------------------------------

    st.subheader("Top Features of Decision Tree")

    st.image(
        "top_feature_tuned_decision_tree.png",
        use_container_width=True
    )

    st.caption(
        "Figure: Feature importance produced by the Decision Tree model."
    )

    st.subheader("Top Features Of Gradient Boosting")

    st.image(
        "top_feature_tuned_gradient_boosting.png",
        use_container_width=True
    )

    st.caption(
        "Figure: Feature importance produced by the Gradient Boosting model."
    )
    
    st.subheader("Top Features Of Linear Regression")
    
    st.image(
        "top_feature_tuned_linear_regression.png",
        use_container_width=True
    )

    st.caption(
        "Figure: Feature importance produced by the Linear Regression model."
    )

    st.subheader("Top Features of Random Forest")

    st.image(
        "top_feature_tuned_random_forest.png",
        use_container_width=True
    )

    st.caption(
        "Figure: Feature importance produced by the Random Forest model."
    )

    st.subheader("Actual vs Predicted Median Price of Gradient Boosting model")

    st.image(
        "tuned_gradient_boosting_actual_predicted.png",
        use_container_width=True
    )

    st.caption(
        "Figure: Show how accurately of the model predicts the median price."
    )

# =========================================================
# TAB 3 - EXPLORATORY DATA ANALYSIS
# =========================================================

with tab3:

    st.header("📈 Exploratory Data Analysis")

    st.write(
        "Exploratory analysis of Malaysia housing prices, "
        "property characteristics, transactions, and relationships "
        "between important variables."
    )

    st.subheader("Distribution of Housing Price")

    st.image(
        "distribution_of_median_housing_price.png",
        use_container_width=True
    )

    st.caption(
        "Figure: Distribution of median housing prices."
    )
    
    st.subheader("Distribution of Property by State")

    st.image(
        "distribution_of_property_by_state.png",
        use_container_width=True
    )

    st.caption(
        "Figure: Distribution of the records group by state."
    )

    st.subheader("Distribution of Property Type")

    st.image(
        "distribution_of_property_type.png",
        use_container_width=True
    )

    st.caption(
        "Figure: Distribution of Property Type."
    )

    st.subheader("Distribution of Median Price Per Square Foot")

    st.image(
        "distribution_of_psf.png",
        use_container_width=True
    )

    st.caption(
        "Figure: Distribution of the Median Price per Square Foot."
    )

    st.subheader("Distibution of Property Transactions")

    st.image(
        "distribution_of_transaction.png",
        use_container_width=True
    )

    st.caption(
        "Figure: Distribution of the number of transactions."
    )
    
    st.subheader("Correlation Heatmap of Numerical Variable")

    st.image(
        "heatmap_num_variable.png",
        use_container_width=True
    )

    st.caption(
        "Figure: A correlation heatmap displaying the linear relationships between numerical variables."
    )

    st.subheader("Median Price by State and Type")

    st.image(
        "heatmap_state_type.png",
        use_container_width=True
    )

    st.caption(
        "Figure: A heatmap illustrating median property prices (in RM) across Malaysian states broken down by property type."
    )

    st.subheader("Median Price by Property Type")

    st.image(
        "median_price_by_type.png",
        use_container_width=True
    )

    st.caption(
        "Figure: Property prices across different property types."
    )

    st.subheader("Median Price per Square Foot by Type")

    st.image(
        "psf_by_type.png",
        use_container_width=True
    )

    st.caption(
        "Figure: A boxplot comparing the distribution of median price per square foot (RM) across different property types."
    )

    st.subheader("Relationship between Median Price per Square Foot and Median Price")
    
    st.image(
        "relationship_of_psf_price.png",
        use_container_width=True
    )

    st.caption(
        "Figure: Relationship between median property price "
        "and median price per square foot."
    )

    st.subheader("Relationship between Transaction and Median Price")
    
    st.image(
        "relationship_of_transaction_price.png",
        use_container_width=True
    )

    st.caption(
        "Figure: Relationship between median property price "
        "and number of transactions."
    )
