import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_fscore_support

# Set page configuration
st.set_page_config(
    page_title="ML Voyager Nexus",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3B82F6;
        margin-bottom: 1rem;
    }
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1E3A8A;
    }
    .metric-label {
        font-size: 1rem;
        color: #6B7280;
    }
    .stButton>button {
        background-color: #3B82F6;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        background-color: #1E3A8A;
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.markdown('<p class="main-header">🧠 ML Voyager Nexus</p>',
            unsafe_allow_html=True)
st.markdown("""
Explore different classification algorithms on classic datasets.
Adjust parameters, apply feature scaling and compare performance metrics in real-time.
""")

# Sidebar configuration
with st.sidebar:
    st.markdown('<p class="sidebar-header">Dataset Selection</p>',
                unsafe_allow_html=True)

    # Dataset selection with icons
    dataset_options = {
        'Iris': '🌸',
        'Wine': '🍷',
        'Breast Cancer': '🩺',
        'Digits': '🔢'
    }
    dataset_labels = [f"{icon} {name}" for name,
                      icon in dataset_options.items()]
    dataset_values = list(dataset_options.keys())
    dataset_index = st.selectbox(
        'Select Dataset',
        range(len(dataset_labels)),
        format_func=lambda i: dataset_labels[i]
    )
    dataset_name = dataset_values[dataset_index]

    st.markdown('<p class="sidebar-header">Preprocessing</p>',
                unsafe_allow_html=True)

    # Feature scaling options
    scaling_options = {
        'None': None,
        'Standard Scaler (Z-score)': 'standard',
        'Min-Max Scaler (0-1)': 'minmax',
        'Robust Scaler (IQR)': 'robust'
    }

    scaling_method = st.selectbox(
        'Feature Scaling',
        list(scaling_options.keys())
    )

    # Test set size
    test_size = st.slider('Test Set Size', 0.1, 0.5, 0.2, 0.05)

    st.markdown('<p class="sidebar-header">Algorithm Selection</p>',
                unsafe_allow_html=True)

    # Expanded classifier selection
    classifier_options = {
        'K-Nearest Neighbors': 'KNN',
        'Support Vector Machine': 'SVM',
        'Random Forest': 'RF',
        'Logistic Regression': 'LR',
        'Decision Tree': 'DT',
        'Gradient Boosting': 'GB',
        'AdaBoost': 'AB',
        'Gaussian Naive Bayes': 'NB',
        'Neural Network (MLP)': 'MLP'
    }

    classifier_name = st.selectbox(
        'Select Algorithm',
        list(classifier_options.keys())
    )
    clf_code = classifier_options[classifier_name]

# Function to load dataset


@st.cache_data
def get_dataset(name):
    if name == 'Iris':
        data = datasets.load_iris()
    elif name == 'Wine':
        data = datasets.load_wine()
    elif name == 'Digits':
        data = datasets.load_digits()
    else:  # Breast Cancer
        data = datasets.load_breast_cancer()

    X = data.data
    y = data.target
    feature_names = data.feature_names
    target_names = data.target_names

    return X, y, feature_names, target_names


# Load the selected dataset
X, y, feature_names, target_names = get_dataset(dataset_name)

# Display dataset information in columns
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown(
        f'<p class="metric-value">{X.shape[1]}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="metric-label">Features</p>',
                unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown(
        f'<p class="metric-value">{X.shape[0]}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="metric-label">Samples</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown(
        f'<p class="metric-value">{len(np.unique(y))}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="metric-label">Classes</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Parameter configuration based on classifier
with st.sidebar:
    st.markdown('<p class="sidebar-header">Model Parameters</p>',
                unsafe_allow_html=True)

    def add_parameter_ui(clf_name):
        params = {}
        if clf_name == 'Support Vector Machine':
            C = st.slider('Regularization (C)', 0.01, 10.0, 1.0, 0.01)
            kernel = st.selectbox(
                'Kernel', ['linear', 'rbf', 'poly', 'sigmoid'], index=1)
            gamma = st.select_slider(
                'Gamma (Kernel Coefficient)',
                options=['scale', 'auto'] + [0.001, 0.01, 0.1, 1, 10, 100],
                value='scale'
            )
            degree = st.slider('Polynomial Degree', 2, 10,
                               3, disabled=(kernel != 'poly'))

            params['C'] = C
            params['kernel'] = kernel
            params['gamma'] = gamma
            if kernel == 'poly':
                params['degree'] = degree

        elif clf_name == 'K-Nearest Neighbors':
            K = st.slider('Number of Neighbors (K)', 1, 20, 5)
            weights = st.selectbox('Weight Function', ['uniform', 'distance'])
            algorithm = st.selectbox(
                'Algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute'])
            metric = st.selectbox('Distance Metric', [
                                  'euclidean', 'manhattan', 'minkowski', 'chebyshev'])

            params['K'] = K
            params['weights'] = weights
            params['algorithm'] = algorithm
            params['metric'] = metric

        elif clf_name == 'Random Forest':
            n_estimators = st.slider('Number of Trees', 10, 500, 100, 10)
            max_depth = st.slider('Max Depth', 2, 30, 10)
            min_samples_split = st.slider('Min Samples Split', 2, 20, 2)
            min_samples_leaf = st.slider('Min Samples Leaf', 1, 10, 1)
            criterion = st.selectbox(
                'Split Criterion', ['gini', 'entropy', 'log_loss'])

            params['n_estimators'] = n_estimators
            params['max_depth'] = max_depth
            params['min_samples_split'] = min_samples_split
            params['min_samples_leaf'] = min_samples_leaf
            params['criterion'] = criterion

        elif clf_name == 'Logistic Regression':
            C = st.slider('Regularization (C)', 0.01, 10.0, 1.0, 0.01)
            solver = st.selectbox(
                'Solver', ['lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga'])
            max_iter = st.slider('Max Iterations', 100, 2000, 100, 100)
            penalty = st.selectbox('Penalty', ['l2', 'l1', 'elasticnet', 'none'],
                                   disabled=(solver in ['newton-cg', 'lbfgs', 'sag']))

            params['C'] = C
            params['solver'] = solver
            params['max_iter'] = max_iter
            if solver not in ['newton-cg', 'lbfgs', 'sag']:
                params['penalty'] = penalty
            else:
                params['penalty'] = 'l2'

        elif clf_name == 'Decision Tree':
            max_depth = st.slider('Max Depth', 2, 30, 10)
            min_samples_split = st.slider('Min Samples Split', 2, 20, 2)
            min_samples_leaf = st.slider('Min Samples Leaf', 1, 10, 1)
            criterion = st.selectbox(
                'Split Criterion', ['gini', 'entropy', 'log_loss'])
            splitter = st.selectbox('Splitter', ['best', 'random'])

            params['max_depth'] = max_depth
            params['min_samples_split'] = min_samples_split
            params['min_samples_leaf'] = min_samples_leaf
            params['criterion'] = criterion
            params['splitter'] = splitter

        elif clf_name == 'Gradient Boosting':
            n_estimators = st.slider(
                'Number of Boosting Stages', 10, 500, 100, 10)
            learning_rate = st.slider('Learning Rate', 0.01, 1.0, 0.1, 0.01)
            max_depth = st.slider('Max Depth', 2, 20, 3)
            subsample = st.slider('Subsample Ratio', 0.5, 1.0, 1.0, 0.05)
            loss = st.selectbox('Loss Function', ['log_loss', 'exponential'])

            params['n_estimators'] = n_estimators
            params['learning_rate'] = learning_rate
            params['max_depth'] = max_depth
            params['subsample'] = subsample
            params['loss'] = loss

        elif clf_name == 'AdaBoost':
            n_estimators = st.slider('Number of Estimators', 10, 500, 50, 10)
            learning_rate = st.slider('Learning Rate', 0.01, 2.0, 1.0, 0.01)
            algorithm = st.selectbox('Algorithm', ['SAMME', 'SAMME.R'])

            params['n_estimators'] = n_estimators
            params['learning_rate'] = learning_rate
            params['algorithm'] = algorithm

        elif clf_name == 'Gaussian Naive Bayes':
            var_smoothing = st.slider(
                'Variance Smoothing', 1e-12, 1e-6, 1e-9, format="%.0e")

            params['var_smoothing'] = var_smoothing

        elif clf_name == 'Neural Network (MLP)':
            hidden_layer_sizes = st.text_input(
                'Hidden Layer Sizes (comma separated)', '100')
            try:
                hidden_layer_sizes = tuple(int(x.strip())
                                           for x in hidden_layer_sizes.split(','))
            except:
                hidden_layer_sizes = (100,)

            activation = st.selectbox('Activation Function', [
                                      'relu', 'tanh', 'logistic', 'identity'])
            solver = st.selectbox('Solver', ['adam', 'sgd', 'lbfgs'])
            alpha = st.slider('Alpha (L2 penalty)', 0.0001,
                              0.01, 0.0001, 0.0001, format="%.4f")
            learning_rate = st.selectbox(
                'Learning Rate', ['constant', 'invscaling', 'adaptive'])
            max_iter = st.slider('Max Iterations', 200, 2000, 200, 100)

            params['hidden_layer_sizes'] = hidden_layer_sizes
            params['activation'] = activation
            params['solver'] = solver
            params['alpha'] = alpha
            params['learning_rate'] = learning_rate
            params['max_iter'] = max_iter

        return params

    params = add_parameter_ui(classifier_name)

# Get classifier based on selection and parameters


def get_classifier(clf_name, params):
    if clf_name == 'Support Vector Machine':
        if params['kernel'] == 'poly':
            return SVC(
                C=params['C'],
                kernel=params['kernel'],
                gamma=params['gamma'],
                degree=params['degree'],
                random_state=42,
                probability=True
            )
        else:
            return SVC(
                C=params['C'],
                kernel=params['kernel'],
                gamma=params['gamma'],
                random_state=42,
                probability=True
            )

    elif clf_name == 'K-Nearest Neighbors':
        return KNeighborsClassifier(
            n_neighbors=params['K'],
            weights=params['weights'],
            algorithm=params['algorithm'],
            metric=params['metric']
        )

    elif clf_name == 'Random Forest':
        return RandomForestClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            min_samples_leaf=params['min_samples_leaf'],
            criterion=params['criterion'],
            random_state=42
        )

    elif clf_name == 'Logistic Regression':
        return LogisticRegression(
            C=params['C'],
            solver=params['solver'],
            max_iter=params['max_iter'],
            penalty=params['penalty'],
            random_state=42
        )

    elif clf_name == 'Decision Tree':
        return DecisionTreeClassifier(
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            min_samples_leaf=params['min_samples_leaf'],
            criterion=params['criterion'],
            splitter=params['splitter'],
            random_state=42
        )

    elif clf_name == 'Gradient Boosting':
        return GradientBoostingClassifier(
            n_estimators=params['n_estimators'],
            learning_rate=params['learning_rate'],
            max_depth=params['max_depth'],
            subsample=params['subsample'],
            loss=params['loss'],
            random_state=42
        )

    elif clf_name == 'AdaBoost':
        return AdaBoostClassifier(
            n_estimators=params['n_estimators'],
            learning_rate=params['learning_rate'],
            algorithm=params['algorithm'],
            random_state=42
        )

    elif clf_name == 'Gaussian Naive Bayes':
        return GaussianNB(
            var_smoothing=params['var_smoothing']
        )

    elif clf_name == 'Neural Network (MLP)':
        return MLPClassifier(
            hidden_layer_sizes=params['hidden_layer_sizes'],
            activation=params['activation'],
            solver=params['solver'],
            alpha=params['alpha'],
            learning_rate=params['learning_rate'],
            max_iter=params['max_iter'],
            random_state=42
        )


# Create classifier
clf = get_classifier(classifier_name, params)

# Apply feature scaling if selected


def apply_scaling(X_train, X_test, scaling_type):
    if scaling_type == 'standard':
        scaler = StandardScaler()
    elif scaling_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaling_type == 'robust':
        scaler = RobustScaler()
    else:
        return X_train, X_test

    X_train_scaled = scaler.fit_transform(X_train)
    # Handle empty test set case
    if X_test.size == 0:
        # Create empty 2D array with correct feature count
        X_test_scaled = np.empty((0, X_train.shape[1]))
    else:
        # Ensure X_test is 2D
        if X_test.ndim == 1:
            X_test = X_test.reshape(-1, X_train.shape[1])
        X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled


# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)

# Apply scaling if selected
scaling_type = scaling_options[scaling_method]
X_train_scaled, X_test_scaled = apply_scaling(X_train, X_test, scaling_type)

# Train the model
try:
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(
        y_test, y_pred, output_dict=True, target_names=target_names)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted')

    # Check if the classifier has feature importances
    has_feature_importance = hasattr(
        clf, 'feature_importances_') or hasattr(clf, 'coef_')

    # Get feature importances if available
    if hasattr(clf, 'feature_importances_'):
        feature_importances = clf.feature_importances_
    elif hasattr(clf, 'coef_'):
        if len(clf.coef_.shape) == 1:
            feature_importances = clf.coef_
        else:
            feature_importances = np.mean(np.abs(clf.coef_), axis=0)
    else:
        feature_importances = None

    training_success = True
except Exception as e:
    st.error(f"Error training model: {e}")
    training_success = False

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Visualization", "📈 Performance", "🔍 Feature Analysis", "📋 Dataset Preview"])

with tab1:
    st.markdown('<p class="sub-header">Data Visualization</p>',
                unsafe_allow_html=True)

    # Visualization options
    viz_col1, viz_col2 = st.columns([1, 3])

    with viz_col1:
        n_components = st.radio("Dimensions", [2, 3], horizontal=True)
        plot_type = st.radio("Plot Type", ["Scatter", "3D Surface"], horizontal=True,
                             disabled=(n_components != 3))

        # Apply the same scaling to visualization data
        X_scaled = apply_scaling(X, np.array([]), scaling_type)[0]

    with viz_col2:
        # Apply PCA
        pca = PCA(n_components)
        X_projected = pca.fit_transform(X_scaled if scaling_type else X)

        # Create the visualization
        if n_components == 2:
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(
                X_projected[:, 0],
                X_projected[:, 1],
                c=y,
                cmap='viridis',
                alpha=0.8,
                s=50,
                edgecolors='w'
            )

            # Add explanatory info
            explained_variance = pca.explained_variance_ratio_
            ax.set_xlabel(
                f'Principal Component 1 ({explained_variance[0]*100:.1f}%)')
            ax.set_ylabel(
                f'Principal Component 2 ({explained_variance[1]*100:.1f}%)')
            ax.set_title(f'PCA of {dataset_name} Dataset' +
                         f' ({scaling_method})' if scaling_type else '',
                         fontsize=15, pad=20)

            # Add a legend
            legend = ax.legend(*scatter.legend_elements(),
                               title="Classes",
                               loc="best")
            ax.add_artist(legend)

            # Add a grid
            ax.grid(alpha=0.3)

            # Improve appearance
            fig.tight_layout()
            st.pyplot(fig)

        else:  # 3D plot
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

            p = ax.scatter(
                X_projected[:, 0],
                X_projected[:, 1],
                X_projected[:, 2],
                c=y,
                cmap='viridis',
                s=60,
                alpha=0.8
            )

            # Add explanatory info
            explained_variance = pca.explained_variance_ratio_
            ax.set_xlabel(f'PC1 ({explained_variance[0]*100:.1f}%)')
            ax.set_ylabel(f'PC2 ({explained_variance[1]*100:.1f}%)')
            ax.set_zlabel(f'PC3 ({explained_variance[2]*100:.1f}%)')
            ax.set_title(f'3D PCA of {dataset_name} Dataset' +
                         f' ({scaling_method})' if scaling_type else '',
                         fontsize=15)

            # Add a colorbar
            fig.colorbar(p, ax=ax, shrink=0.7, aspect=20)

            st.pyplot(fig)

        # Show explained variance
        st.markdown("#### Explained Variance")
        total_var = sum(pca.explained_variance_ratio_)
        st.progress(total_var)
        st.text(f"Total Explained Variance: {total_var*100:.2f}%")

if training_success:
    with tab2:
        st.markdown('<p class="sub-header">Model Performance</p>',
                    unsafe_allow_html=True)

        # Display metrics in columns
        perf_col1, perf_col2, perf_col3 = st.columns([1, 1, 1])

        with perf_col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(
                f'<p class="metric-value">{acc:.4f}</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="metric-label">Accuracy Score</p>',
                        unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with perf_col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(
                f'<p class="metric-value">{precision:.4f}</p>', unsafe_allow_html=True)
            st.markdown(
                f'<p class="metric-label">Precision (Weighted)</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with perf_col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(
                f'<p class="metric-value">{recall:.4f}</p>', unsafe_allow_html=True)
            st.markdown(
                f'<p class="metric-label">Recall (Weighted)</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Confusion matrix and class metrics in columns
        cm_col1, cm_col2 = st.columns([1, 1])

        with cm_col1:
            # Plot confusion matrix
            st.markdown("#### Confusion Matrix")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted Labels')
            ax.set_ylabel('True Labels')
            st.pyplot(fig)

        with cm_col2:
            # Plot class metrics
            st.markdown("#### Class-wise Performance")

            # Extract class metrics
            class_metrics = {'Class': [], 'Precision': [],
                             'Recall': [], 'F1-Score': []}
            for i, class_name in enumerate(target_names):
                class_metrics['Class'].append(class_name)
                if str(i) in class_report:
                    class_metrics['Precision'].append(
                        class_report[str(i)]['precision'])
                    class_metrics['Recall'].append(
                        class_report[str(i)]['recall'])
                    class_metrics['F1-Score'].append(
                        class_report[str(i)]['f1-score'])
                elif class_name in class_report:
                    class_metrics['Precision'].append(
                        class_report[class_name]['precision'])
                    class_metrics['Recall'].append(
                        class_report[class_name]['recall'])
                    class_metrics['F1-Score'].append(
                        class_report[class_name]['f1-score'])

            metrics_df = pd.DataFrame(class_metrics)

            # Plot metrics
            fig, ax = plt.subplots(figsize=(8, 6))
            metrics_df_long = pd.melt(metrics_df, id_vars=['Class'],
                                      value_vars=['Precision',
                                                  'Recall', 'F1-Score'],
                                      var_name='Metric', value_name='Value')

            sns.barplot(x='Class', y='Value', hue='Metric',
                        data=metrics_df_long, ax=ax)
            ax.set_ylim(0, 1)
            ax.set_title('Class-wise Metrics')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)

    with tab3:
        st.markdown('<p class="sub-header">Feature Analysis</p>',
                    unsafe_allow_html=True)

        if has_feature_importance and feature_importances is not None:
            # Create a DataFrame with feature names and importances
            feature_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': feature_importances
            })

            # Sort by importance
            feature_df = feature_df.sort_values('Importance', ascending=False)

            # Display top features
            st.markdown("#### Feature Importance")

            # Plot feature importance
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.barplot(x='Importance', y='Feature',
                        data=feature_df.head(15), ax=ax)
            ax.set_title(f'Top Feature Importances for {classifier_name}')
            plt.tight_layout()
            st.pyplot(fig)

            # Feature comparison
            st.markdown("#### Feature Comparisons")

            # Let user select features to compare
            top_features = feature_df['Feature'].head(5).tolist()

            feat_col1, feat_col2 = st.columns(2)

            with feat_col1:
                x_feature = st.selectbox('X-axis Feature', feature_names,
                                         index=feature_names.tolist().index(
                                             top_features[0])
                                         if top_features[0] in feature_names else 0)

            with feat_col2:
                y_feature = st.selectbox('Y-axis Feature', feature_names,
                                         index=feature_names.tolist().index(
                                             top_features[1])
                                         if len(top_features) > 1 and top_features[1] in feature_names else 1)

            # Create a scatter plot of the two selected features
            fig, ax = plt.subplots(figsize=(10, 8))

            # Get the indices of the selected features
            x_idx = feature_names.tolist().index(x_feature)
            y_idx = feature_names.tolist().index(y_feature)

            scatter = ax.scatter(
                X_scaled[:, x_idx] if scaling_type else X[:, x_idx],
                X_scaled[:, y_idx] if scaling_type else X[:, y_idx],
                c=y,
                cmap='viridis',
                alpha=0.8,
                edgecolors='w'
            )

            ax.set_xlabel(x_feature)
            ax.set_ylabel(y_feature)
            ax.set_title(f'{x_feature} vs {y_feature} by Class')

            # Add a legend
            legend = ax.legend(*scatter.legend_elements(),
                               title="Classes",
                               loc="best")
            ax.add_artist(legend)

            # Add a grid
            ax.grid(alpha=0.3)

            st.pyplot(fig)
        else:
            st.info(
                f"Feature importance is not available for {classifier_name}.")

            # Instead, show a correlation heatmap
            st.markdown("#### Feature Correlation")

            # Create a DataFrame
            df = pd.DataFrame(X, columns=feature_names)

            # Correlation matrix
            corr = df.corr()

            # Plot heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=False,
                        square=True, linewidths=.5, cbar_kws={"shrink": .5})
            plt.tight_layout()
            st.pyplot(fig)

            # Let user select features
            feat_col1, feat_col2 = st.columns(2)

            with feat_col1:
                x_feature = st.selectbox(
                    'X-axis Feature', feature_names, index=0)

            with feat_col2:
                y_feature = st.selectbox(
                    'Y-axis Feature', feature_names, index=1)

            # Create a scatter plot of the two selected features
            fig, ax = plt.subplots(figsize=(10, 8))

            # Get the indices of the selected features
            x_idx = feature_names.index(x_feature)
            y_idx = feature_names.index(y_feature)

            scatter = ax.scatter(
                X_scaled[:, x_idx] if scaling_type else X[:, x_idx],
                X_scaled[:, y_idx] if scaling_type else X[:, y_idx],
                c=y,
                cmap='viridis',
                alpha=0.8,
                edgecolors='w'
            )

            ax.set_xlabel(x_feature)
            ax.set_ylabel(y_feature)
            ax.set_title(f'{x_feature} vs {y_feature} by Class')

            # Add a legend
            legend = ax.legend(*scatter.legend_elements(),
                               title="Classes",
                               loc="best")
            ax.add_artist(legend)

            # Add a grid
            ax.grid(alpha=0.3)

            st.pyplot(fig)

with tab4:
    st.markdown('<p class="sub-header">Dataset Preview</p>',
                unsafe_allow_html=True)

    # Create a DataFrame for display
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    df['target_name'] = [target_names[i] for i in y]

    # Show sample data
    st.markdown("#### Dataset Sample")
    st.dataframe(df.head(10), use_container_width=True)

    # Dataset statistics
    st.markdown("#### Feature Statistics")
    stats_df = df.drop(['target', 'target_name'], axis=1).describe().T
    stats_df = stats_df.round(2)
    st.dataframe(stats_df, use_container_width=True)

    # Feature correlation heatmap
    st.markdown("#### Feature Correlation")
    corr = df.drop(['target', 'target_name'], axis=1).corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=False,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.tight_layout()
    st.pyplot(fig)

    # Class distribution
    st.markdown("#### Class Distribution")
    class_dist = df['target_name'].value_counts().reset_index()
    class_dist.columns = ['Class', 'Count']

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Class', y='Count', data=class_dist, ax=ax)
    ax.set_title('Class Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

# Add a tab for model comparison
st.markdown('<p class="sub-header">Model Comparison</p>',
            unsafe_allow_html=True)

# Create a form for model comparison
with st.form("model_comparison"):
    st.markdown("#### Compare Multiple Models")
    st.write("Select models to compare on the current dataset:")

    # Create checkboxes for model selection
    model_selection = {}
    col1, col2, col3 = st.columns(3)

    with col1:
        model_selection['K-Nearest Neighbors'] = st.checkbox(
            'K-Nearest Neighbors', value=True)
        model_selection['Support Vector Machine'] = st.checkbox(
            'Support Vector Machine', value=True)
        model_selection['Random Forest'] = st.checkbox(
            'Random Forest', value=True)

    with col2:
        model_selection['Logistic Regression'] = st.checkbox(
            'Logistic Regression', value=False)
        model_selection['Decision Tree'] = st.checkbox(
            'Decision Tree', value=False)
        model_selection['Gradient Boosting'] = st.checkbox(
            'Gradient Boosting', value=False)

    with col3:
        model_selection['AdaBoost'] = st.checkbox('AdaBoost', value=False)
        model_selection['Gaussian Naive Bayes'] = st.checkbox(
            'Gaussian Naive Bayes', value=False)
        model_selection['Neural Network (MLP)'] = st.checkbox(
            'Neural Network (MLP)', value=False)

    # Submit button
    submitted = st.form_submit_button("Compare Selected Models")

# Run comparison if submitted
if 'submitted' in locals() and submitted:
    # List of selected models
    selected_models = [model for model,
                       selected in model_selection.items() if selected]

    if not selected_models:
        st.warning("Please select at least one model to compare.")
    else:
        st.markdown("#### Model Comparison Results")

        # Prepare data for comparison
        comparison_results = []

        # Define default parameters for each model
        default_params = {
            'K-Nearest Neighbors': {'K': 5, 'weights': 'uniform', 'algorithm': 'auto', 'metric': 'euclidean'},
            'Support Vector Machine': {'C': 1.0, 'kernel': 'rbf', 'gamma': 'scale'},
            'Random Forest': {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 2, 'min_samples_leaf': 1, 'criterion': 'gini'},
            'Logistic Regression': {'C': 1.0, 'solver': 'lbfgs', 'max_iter': 100, 'penalty': 'l2'},
            'Decision Tree': {'max_depth': 10, 'min_samples_split': 2, 'min_samples_leaf': 1, 'criterion': 'gini', 'splitter': 'best'},
            'Gradient Boosting': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3, 'subsample': 1.0, 'loss': 'log_loss'},
            'AdaBoost': {'n_estimators': 50, 'learning_rate': 1.0, 'algorithm': 'SAMME.R'},
            'Gaussian Naive Bayes': {'var_smoothing': 1e-9},
            'Neural Network (MLP)': {'hidden_layer_sizes': (100,), 'activation': 'relu', 'solver': 'adam', 'alpha': 0.0001, 'learning_rate': 'constant', 'max_iter': 200}
        }

        # Progress bar
        progress_bar = st.progress(0)

        # Train and evaluate each selected model
        for i, model_name in enumerate(selected_models):
            # Create classifier with default parameters
            model = get_classifier(model_name, default_params[model_name])

            try:
                # Train model
                model.fit(X_train_scaled, y_train)

                # Predict
                y_pred = model.predict(X_test_scaled)

                # Calculate metrics
                acc = accuracy_score(y_test, y_pred)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_test, y_pred, average='weighted')

                # Add to results
                comparison_results.append({
                    'Model': model_name,
                    'Accuracy': acc,
                    'Precision': precision,
                    'Recall': recall,
                    'F1-Score': f1,
                    'Training Time': 'N/A'  # Could add timing if needed
                })

            except Exception as e:
                st.error(f"Error training {model_name}: {e}")

            # Update progress bar
            progress_bar.progress((i + 1) / len(selected_models))

        # Create comparison table
        if comparison_results:
            comparison_df = pd.DataFrame(comparison_results)
            comparison_df = comparison_df.sort_values(
                'Accuracy', ascending=False)

            # Display comparison table
            st.dataframe(comparison_df.style.highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1-Score']),
                         use_container_width=True)

            # Plot comparison chart
            fig, ax = plt.subplots(figsize=(12, 6))
            comparison_df_long = pd.melt(comparison_df, id_vars=['Model'],
                                         value_vars=[
                                             'Accuracy', 'Precision', 'Recall', 'F1-Score'],
                                         var_name='Metric', value_name='Value')

            sns.barplot(x='Model', y='Value', hue='Metric',
                        data=comparison_df_long, ax=ax)
            ax.set_title('Model Comparison')
            ax.set_ylim(0, 1)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)

            # Best model
            best_model = comparison_df.iloc[0]['Model']
            best_accuracy = comparison_df.iloc[0]['Accuracy']

            st.success(
                f"Best model: {best_model} with accuracy {best_accuracy:.4f}")

# Add hyperparameter tuning section
st.markdown('<p class="sub-header">Hyperparameter Tuning</p>',
            unsafe_allow_html=True)

with st.expander("Hyperparameter Tuning"):
    st.write("""
    Hyperparameter tuning helps find the optimal parameters for your model.
    Select a search method and parameter ranges to tune.
    """)

    if classifier_name in ['K-Nearest Neighbors', 'Support Vector Machine', 'Random Forest']:
        # Create a form for hyperparameter tuning
        with st.form("hyperparameter_tuning"):
            st.markdown(f"#### Tune {classifier_name}")

            # Tuning method
            tuning_method = st.radio(
                "Tuning Method", ["Grid Search", "Random Search"])

            # CV folds
            cv_folds = st.slider("Cross-validation Folds", 2, 10, 5)

            # Scoring metric
            scoring = st.selectbox("Scoring Metric", [
                                   "accuracy", "precision_weighted", "recall_weighted", "f1_weighted"])

            # Set parameter ranges based on classifier
            param_ranges = {}

            if classifier_name == 'K-Nearest Neighbors':
                param_ranges['n_neighbors'] = st.multiselect(
                    "K Values", list(range(1, 21)), default=[3, 5, 7, 9])
                param_ranges['weights'] = st.multiselect(
                    "Weights", ['uniform', 'distance'], default=['uniform', 'distance'])
                param_ranges['metric'] = st.multiselect(
                    "Metrics", ['euclidean', 'manhattan', 'minkowski'], default=['euclidean'])

            elif classifier_name == 'Support Vector Machine':
                param_ranges['C'] = st.multiselect(
                    "C Values", [0.1, 1, 10, 100], default=[0.1, 1, 10])
                param_ranges['kernel'] = st.multiselect(
                    "Kernels", ['linear', 'rbf', 'poly'], default=['rbf'])
                param_ranges['gamma'] = st.multiselect(
                    "Gamma Values", ['scale', 'auto', 0.1, 0.01], default=['scale', 'auto'])

            elif classifier_name == 'Random Forest':
                param_ranges['n_estimators'] = st.multiselect(
                    "Number of Trees", [10, 50, 100, 200], default=[50, 100])
                param_ranges['max_depth'] = st.multiselect(
                    "Max Depths", [None, 5, 10, 15], default=[None, 10])
                param_ranges['min_samples_split'] = st.multiselect(
                    "Min Samples Split", [2, 5, 10], default=[2, 5])

            # Submit button
            tune_submitted = st.form_submit_button("Start Tuning")

    else:
        st.info(f"Hyperparameter tuning interface is currently available for KNN, SVM and Random Forest models. Please select one of these models in the sidebar.")

# Add footer with info
st.markdown("---")
st.markdown(
    "This advanced ML explorer tool helps you analyze datasets, compare models and tune hyperparameters. "
    "Adjust parameters in the sidebar and explore different visualization options to gain insights into your data and models."
)

# Add data export option
with st.expander("Export Options"):
    st.write("Export trained model or results")

    # Add export buttons
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Export Results as CSV"):
            st.info(
                "In a real application, this would allow downloading results as CSV.")

    with col2:
        if st.button("Export Trained Model"):
            st.info(
                "In a real application, this would allow downloading the trained model.")
