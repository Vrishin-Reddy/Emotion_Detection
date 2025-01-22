import streamlit as st
import pandas as pd
import pickle
import os
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import time

# Set page configuration
st.set_page_config(
    page_title="Emotion Recognition Chatbot",
    layout="wide",
    page_icon="🤖",
    initial_sidebar_state="expanded",
)

# Paths to the saved models and results
save_dir = os.path.join(os.path.dirname(__file__), "Emotion_Detection", "model_saves")

results_file = os.path.join(save_dir, "results.csv")
vectorizer_file = os.path.join(save_dir, "tfidf_vect.pk")

# Load model results
@st.cache_data
def load_model_results():
    if os.path.exists(results_file):
        return pd.read_csv(results_file)
    return pd.DataFrame(columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score", "Composite Score"])

model_results = load_model_results()

# Load vectorizer
@st.cache_resource
def load_vectorizer():
    if os.path.exists(vectorizer_file):
        with open(vectorizer_file, 'rb') as f:
            return pickle.load(f)
    return None

vectorizer = load_vectorizer()

# Load models dynamically
@st.cache_resource
def load_models():
    loaded_models = {}
    if os.path.exists(save_dir):
        for filename in os.listdir(save_dir):
            if filename.endswith("_text_emotion_model.sav"):
                try:
                    model_name = filename.split("_text_emotion_model.sav")[0].replace("_", " ").title()
                    with open(os.path.join(save_dir, filename), 'rb') as f:
                        loaded_models[model_name] = pickle.load(f)
                except Exception as e:
                    st.error(f"Error loading model {filename}: {e}")
    else:
        st.error(f"Model save directory '{save_dir}' does not exist.")
    return loaded_models

models = load_models()

# Load Cross-Validation Results
@st.cache_data
def load_cross_validation_results():
    cv_results = {}
    for filename in os.listdir(save_dir):
        if filename.endswith("_cross_validation_results.csv"):
            model_name = filename.split("_cross_validation_results.csv")[0].replace("_", " ").title()
            file_path = os.path.join(save_dir, filename)
            try:
                cv_data = pd.read_csv(file_path)
                cv_results[model_name] = cv_data
            except Exception as e:
                st.error(f"Error loading cross-validation results for {model_name}: {e}")
    return cv_results

cross_validation_results = load_cross_validation_results()

# Load Model Fit Evaluation
@st.cache_data
def load_fit_evaluation():
    fit_evaluation_file = os.path.join(save_dir, "model_fit_evaluation.csv")
    if os.path.exists(fit_evaluation_file):
        try:
            return pd.read_csv(fit_evaluation_file)
        except Exception as e:
            st.error(f"Error loading model fit evaluation data: {e}")
            return None
    else:
        st.error(f"Model fit evaluation file not found at {fit_evaluation_file}")
        return None

model_fit_evaluation = load_fit_evaluation()

# Custom CSS for Animations and Navbar
st.markdown(
    """
    <style>
    /* Sidebar Animation */
    .sidebar .sidebar-content {
        animation: slide-in 0.8s ease-out;
    }
    @keyframes slide-in {
        from {
            margin-left: -300px;
            opacity: 0;
        }
        to {
            margin-left: 0;
            opacity: 1;
        }
    }
    /* Header Animation */
    .main-header {
        animation: fade-in 1.5s ease-in-out;
        text-shadow: 0 0 10px rgba(255, 255, 255, 0.7), 0 0 20px rgba(255, 255, 255, 0.5);
    }
    @keyframes fade-in {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    /* Home Animation */
    .home-section {
        animation: zoom-in 1.5s ease;
    }
    @keyframes zoom-in {
        from {
            transform: scale(0.5);
            opacity: 0;
        }
        to {
            transform: scale(1);
            opacity: 1;
        }
    }
    /* Navigation Highlight */
    .stRadio > div {
        display: flex;
        justify-content: space-around;
    }
    .stRadio > div > label {
        transition: background-color 0.3s ease, transform 0.3s ease;
    }
    .stRadio > div > label:hover {
        background-color: #e8f0fe;
        transform: scale(1.05);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["Home", "Predict Emotion", "Model Insights", "Customize Metrics"],
    index=0,
    format_func=lambda x: f"📋 {x}" if x == "Home" else f"🔍 {x}" if x == "Predict Emotion" else f"📊 {x}" if x == "Model Insights" else f"⚙️ {x}",
)

# Emotion Mapping for Styling
emotion_colors = {
    "Anger": "#FF6B6B",
    "Fear": "#C06C84",
    "Joy": "#FFD700",
    "Love": "#FF85A1",
    "Sadness": "#87CEFA",
    "Surprise": "#FFA07A",
    "Unknown Emotion": "#B0B0B0",
}

# Home Page with Animation
if page == "Home":
    st.markdown("<h1 class='main-header'>Welcome to the Emotion Recognition Chatbot! 🤖</h1>", unsafe_allow_html=True)
    try:
        image = Image.open("static/chatbot_image.png")
        st.image(image, use_container_width=True, caption="Analyze Emotions with AI")
    except FileNotFoundError:
        st.warning("Image file not found. Add 'chatbot_image.png' to the 'static' folder.")

    st.markdown(
        """
        <div class='home-section'>
            <h3>How It Works:</h3>
            <ul>
                <li>Enter a sentence to analyze the emotion behind it.</li>
                <li>Select a model from the dropdown for analysis.</li>
                <li>View the predicted emotion and accuracy of the selected model.</li>
            </ul>
            <h3>Available Emotions:</h3>
            <ul>
                <li>😡 <b>Anger</b></li>
                <li>😱 <b>Fear</b></li>
                <li>😊 <b>Joy</b></li>
                <li>❤️ <b>Love</b></li>
                <li>😢 <b>Sadness</b></li>
                <li>😮 <b>Surprise</b></li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Predict Emotion Page
if page == "Predict Emotion":
    st.title("Predict Emotion from Text 🔍")

    
    default_model_name = "Lasso Regression"
    default_model = models.get(default_model_name)

    if not default_model:
        st.error(f"The model '{default_model_name}' could not be found in the directory.")
    else:
        if not model_results.empty:
            model_row = model_results[model_results["Model"].str.strip().str.title() == default_model_name]
            if not model_row.empty:
                accuracy = model_row["Accuracy"].values[0]
                st.markdown(
                    f"<span style='background-color:#D1ECF1;color:#0C5460;padding:5px;border-radius:5px;font-weight:bold;'>Model Accuracy: {accuracy:.2f}%</span>",
                    unsafe_allow_html=True,
                )
            else:
                st.warning("Model accuracy data not available for 'Lasso Regression'.")
        else:
            st.warning("No model results available.")

    sentence = st.text_area("Enter your sentence:", placeholder="Type something here...")

    if st.button("Predict Emotion"):
        if not sentence.strip():
            st.warning("Please enter a sentence.")
        else:
            cleaned_sentence = "".join([char.lower() for char in sentence if char.isalnum() or char.isspace()])
            try:
                vectorized_input = vectorizer.transform([cleaned_sentence])
            except Exception as e:
                st.error(f"Error in vectorizing input: {e}")
                st.stop()

            with st.spinner("Analyzing..."):
                for progress in range(0, 101, 10):
                    time.sleep(0.1)
                    st.progress(progress)

            try:
                prediction = default_model.predict(vectorized_input)
            except Exception as e:
                st.error(f"Error in model prediction: {e}")
                st.stop()

            emotion_mapping = {
                0: "Anger",
                1: "Fear",
                2: "Joy",
                3: "Love",
                4: "Sadness",
                5: "Surprise",
            }
            emotion = emotion_mapping.get(prediction[0], "Unknown Emotion")

            color = emotion_colors.get(emotion, "#B0B0B0")
            st.markdown(
                f"<h2 style='text-align: center; color: {color}; animation: fade-in 2s;'>Predicted Emotion: {emotion}</h2>",
                unsafe_allow_html=True,
            )

# Model Insights Page
if page == "Model Insights":
    st.title("Model Insights 📊")
    if model_results.empty:
        st.warning("No model insights available.")
    else:
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
            ["📋 Data View", "📈 Visual Insights", "🎯 Cross-Validation", "📉 Model Fit Evaluation", "📊 Detailed Comparisons", "🔄 Advanced Comparison"]
        )

        with tab1:
            st.dataframe(model_results)
            st.download_button(
                label="📥 Download Results",
                data=model_results.to_csv(index=False),
                file_name="model_results.csv",
                mime="text/csv",
            )

        # Tab 2: Visual Insights
        with tab2:
            # Bar Chart
            st.markdown("### Bar Chart of Model Accuracies")
            chart_data = model_results[["Model", "Accuracy"]]
            bar_fig = px.bar(
                chart_data,
                x="Accuracy",
                y="Model",
                color="Model",
                orientation="h",
                title="Model Accuracies",
                color_discrete_sequence=px.colors.qualitative.Plotly,
                text="Accuracy",
            )
            bar_fig.update_traces(texttemplate='%{text:.2f}', textposition='inside')
            st.plotly_chart(bar_fig, use_container_width=True)

            # Line Chart for Composite Scores
            st.markdown("### Line Chart of Composite Scores")
            line_fig = px.line(
                model_results,
                x="Model",
                y="Composite Score",
                title="Composite Scores Across Models",
                markers=True,
                text="Composite Score",
            )
            line_fig.update_traces(textposition="top center", texttemplate='%{y:.2f}')
            st.plotly_chart(line_fig, use_container_width=True)

        with tab3:
            if cross_validation_results:
                selected_cv_model = st.selectbox("Select a model to view cross-validation results:", list(cross_validation_results.keys()))
                if selected_cv_model in cross_validation_results:
                    cv_data = cross_validation_results[selected_cv_model]
                    if not cv_data.empty:
                        st.dataframe(cv_data)
                        if 'Model' in cv_data.columns:
                            x_labels = cv_data.columns[1:]  # Assuming all other columns after 'Model' are metrics
                            y_labels = cv_data['Model']  # Using 'Model' as the y-axis labels
                            # Generate heatmap
                            heatmap_fig = px.imshow(
                                cv_data[x_labels].values,  # Convert DataFrame section to numpy for imshow
                                labels=dict(x="Metrics", y="Models", color="Metric Score"),
                                x=x_labels,
                                y=y_labels,
                                aspect="auto",
                                title="Cross-Validation Metrics Heatmap",
                                color_continuous_scale="Viridis"
                            )
                            st.plotly_chart(heatmap_fig, use_container_width=True)
                        else:
                            st.error("Expected 'Model' column not found in data.")
                    else:
                        st.warning("No data available for selected model.")
                else:
                    st.warning("Selected model not found in cross-validation results.")
            else:
                st.warning("No cross-validation data available.")

        with tab4:  # Model Fit Evaluation
            st.header("Model Fit Evaluation 📉")
            st.markdown("### Metrics across different models and cross-validation folds")

            # Directory containing the model fit evaluation files
            save_dir = os.path.join(os.path.dirname(__file__), "Emotion_Detection", "model_saves")


            # Find all CSV files containing cross-validation results
            model_files = [f for f in os.listdir(save_dir) if f.endswith('_cross_validation_results.csv')]

            if not model_files:
                st.error("No model fit evaluation files found in the directory.")
            else:
                # Generate a dictionary of model names from files
                model_names = [f.replace('_cross_validation_results.csv', '').replace('_', ' ').title() for f in model_files]
                # Dropdown to select a model
                selected_model_name = st.selectbox("Select a model to display results:", model_names)

                # Construct the file path from the selected model, ensuring spaces are handled correctly
                file_path = os.path.join(save_dir, selected_model_name.replace(' ', '_') + '_cross_validation_results.csv')

                # Check if file exists before attempting to load
                if not os.path.exists(file_path):
                    corrected_file_path = os.path.join(save_dir, selected_model_name.replace(' ', ' ') + '_cross_validation_results.csv')  # Try with spaces
                    if not os.path.exists(corrected_file_path):
                        st.error(f"The file for '{selected_model_name}' does not exist at the path: {file_path}")
                    else:
                        file_path = corrected_file_path
                try:
                    fit_evaluation = pd.read_csv(file_path)
                    st.markdown(f"#### {selected_model_name}")
                    st.dataframe(fit_evaluation)

                    color_sequence = px.colors.qualitative.Vivid  # Using Plotly's 'Vivid' color palette
                    metrics = fit_evaluation.columns[1:]  # Assuming the first column is not a metric

                    # Create interactive line charts for each metric
                    for i, metric in enumerate(metrics):
                        fig = px.line(
                            fit_evaluation,
                            x=fit_evaluation.index,
                            y=metric,
                            title=f"{metric} for {selected_model_name} over Cross-Validation Folds",
                            markers=True,
                            labels={'x': 'CV Fold Index', 'y': metric},
                            color_discrete_sequence=[color_sequence[i % len(color_sequence)]]
                        )
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error loading or displaying data for {selected_model_name}: {e}")

        with tab5:  # Detailed Comparisons
            radar_fig = go.Figure()
            for _, row in model_results.iterrows():
                radar_fig.add_trace(go.Scatterpolar(
                    r=[row["Accuracy"], row["Precision"], row["Recall"], row["F1 Score"]],
                    theta=["Accuracy", "Precision", "Recall", "F1 Score"],
                    fill='toself',
                    name=row["Model"]
                ))
            radar_fig.update_layout(
                title="Model Metrics Comparison",
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
            )
            st.plotly_chart(radar_fig, use_container_width=True)

            st.markdown("### Heatmap of Model Metrics")
            heatmap_data = model_results.set_index("Model")[["Accuracy", "Precision", "Recall", "F1 Score"]]
            heatmap_fig = px.imshow(
                heatmap_data,
                text_auto=True,
                aspect="auto",
                title="Model Metrics Heatmap",
                color_continuous_scale="Blues",
            )
            st.plotly_chart(heatmap_fig, use_container_width=True)

        with tab6:  # Advanced Model Comparison
            st.markdown("## Advanced Model Comparison 🔄")
            if len(models) < 2:
                st.warning("At least two models are needed for comparison.")
            else:
                model_options = list(models.keys())
                model_1 = st.selectbox("Select the first model:", model_options, index=0)
                model_2 = st.selectbox("Select the second model:", model_options, index=1 if len(model_options) > 1 else 0)

                if model_1 and model_2:
                    if model_1 == model_2:
                        st.error("Please select two different models for comparison.")
                    else:
                        model_1_data = model_results[model_results['Model'].str.strip().str.title() == model_1]
                        model_2_data = model_results[model_results['Model'].str.strip().str.title() == model_2]

                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"#### Metrics for {model_1}")
                            st.dataframe(model_1_data)
                        with col2:
                            st.markdown(f"#### Metrics for {model_2}")
                            st.dataframe(model_2_data)

                        comparison_data = pd.concat([model_1_data, model_2_data])
                        comparison_fig = go.Figure(data=[
                            go.Bar(name=model_1, x=comparison_data.columns[1:], y=comparison_data.iloc[0, 1:], marker_color='indigo'),
                            go.Bar(name=model_2, x=comparison_data.columns[1:], y=comparison_data.iloc[1, 1:], marker_color='green')
                        ])
                        comparison_fig.update_layout(barmode='group', title=f"Comparison of {model_1} vs {model_2}")
                        st.plotly_chart(comparison_fig, use_container_width=True)

                        st.download_button(
                            label="📥 Download Comparison Results",
                            data=comparison_data.to_csv(index=False),
                            file_name=f"{model_1}_vs_{model_2}_comparison.csv",
                            mime="text/csv",
                        )

# Customize Metrics Page
if page == "Customize Metrics":
    st.title("Customize Metrics ⚙️")
    if model_results.empty:
        st.warning("No model results available.")
    else:
        metric = st.selectbox("Choose a Metric to Filter By:", ["Accuracy", "Precision", "Recall", "F1 Score"])
        min_threshold = st.slider("Set Minimum Threshold:", 0, 100, 80)

        filtered_results = model_results[model_results[metric] >= min_threshold / 100.0]
        if filtered_results.empty:
            st.warning(f"No models found with {metric} above {min_threshold}%. Try lowering the threshold.")
        else:
            st.dataframe(filtered_results)
            st.markdown(f"### Enhanced Visualization of {metric}")
            chart_data = filtered_results[["Model", metric]]
            bar_fig = px.bar(
                chart_data,
                x="Model",
                y=metric,
                color="Model",
                title=f"Models by {metric}",
                color_discrete_sequence=px.colors.qualitative.Vivid,
                text=metric,
            )
            bar_fig.update_traces(texttemplate='%{text:.2f}', textposition='inside')
            st.plotly_chart(bar_fig, use_container_width=True)

            st.markdown(f"### Bubble Chart for {metric} vs Composite Score")
            bubble_fig = px.scatter(
                filtered_results,
                x="Composite Score",
                y=metric,
                size="Accuracy",
                color="Model",
                title=f"Bubble Chart: {metric} vs Composite Score",
                hover_name="Model",
                color_discrete_sequence=px.colors.qualitative.Pastel,
                size_max=20,
            )
            st.plotly_chart(bubble_fig, use_container_width=True)

            st.download_button(
                label="📥 Download Filtered Results",
                data=filtered_results.to_csv(index=False),
                file_name="filtered_model_results.csv",
                mime="text/csv",
            )
