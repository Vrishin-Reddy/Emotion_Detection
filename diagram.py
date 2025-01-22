## To Generate the System Architecture 

import os
from diagrams import Cluster, Diagram, Edge
from diagrams.aws.analytics import Glue
from diagrams.aws.ml import SagemakerTrainingJob
from diagrams.aws.compute import Lambda
from diagrams.aws.ml import Personalize
from diagrams.aws.analytics import Quicksight
from diagrams.aws.database import Dynamodb
from diagrams.aws.network import APIGateway

# Define the output directory
output_dir = r"C:\temp\new_outputs"
os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

# Diagram attributes for 4K quality and advanced styles
graph_attr = {
    "dpi": "300",  # High resolution for 4K output
    "fontsize": "20",  # Larger font size for better readability
    "fontname": "Helvetica",  # Advanced font style
    "rankdir": "TB",  # Top-to-Bottom layout for better clarity
}

# Generate the diagram
with Diagram(
    "Machine Learning Workflow",
    show=False,
    outformat="png",
    filename=os.path.join(output_dir, "ml_workflow"),
    graph_attr=graph_attr,  # Apply graph attributes
):
    with Cluster("Input Layer"):
        dataset = Glue("Dataset")

    with Cluster("Preprocessing Module"):
        text_cleaning = Lambda("Text Cleaning")
        tokenization = Lambda("Tokenization")
        normalization = Lambda("Normalization")
        class_imbalance = Lambda("Imbalance Correction")

        # Arrange preprocessing steps sequentially
        dataset >> text_cleaning >> tokenization >> normalization >> class_imbalance

    with Cluster("Feature Extraction and Embedding"):
        feature_extraction = SagemakerTrainingJob("Feature Extraction")

    # Explicit connection to avoid penetration issues
    class_imbalance >> Edge(color="blue", style="solid") >> feature_extraction

    with Cluster("Model Training"):
        with Cluster("Baseline Models"):
            knn = SagemakerTrainingJob("KNN")
            svm = SagemakerTrainingJob("SVM")
            logistic_regression = SagemakerTrainingJob("Logistic Regression")
            decision_tree = SagemakerTrainingJob("Decision Tree")
            random_forest = SagemakerTrainingJob("Random Forest")
            xgboost = SagemakerTrainingJob("XGBoost")
            lasso_regression = SagemakerTrainingJob("Lasso Regression")
            naive_bayes = SagemakerTrainingJob("Naive Bayes")

        transformer_model = Personalize("Transformer-based Model")
        fine_tuned_bert = Personalize("Fine-tuned BERT")

        # Arrange connections from feature extraction
        feature_extraction >> [
            knn,
            svm,
            logistic_regression,
            decision_tree,
            random_forest,
            xgboost,
            lasso_regression,
            naive_bayes,
        ]
        feature_extraction >> Edge(color="green", style="dashed") >> transformer_model >> fine_tuned_bert

    with Cluster("Evaluation"):
        performance_metrics = Lambda("Performance Metric")
        classification_report = Lambda("Classification Reports")
        confusion_matrix = Lambda("Confusion Matrix")

        [
            knn,
            svm,
            logistic_regression,
            decision_tree,
            random_forest,
            xgboost,
            lasso_regression,
            naive_bayes,
            fine_tuned_bert,
        ] >> Edge(color="red", style="solid") >> performance_metrics
        performance_metrics >> [classification_report, confusion_matrix]

    with Cluster("Deployment"):
        real_time_module = APIGateway("Real-Time Module")
        output_dashboard = Quicksight("Output Dashboard")
        continuous_learning = Dynamodb("Continuous Learning")

        # Explicit connections for deployment
        performance_metrics >> Edge(color="purple", style="solid") >> real_time_module
        real_time_module >> Edge(color="purple", style="solid") >> output_dashboard
        output_dashboard >> Edge(color="purple", style="dotted") >> continuous_learning
