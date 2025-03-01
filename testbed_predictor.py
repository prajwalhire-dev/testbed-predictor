import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import joblib
import os


class TestbedPredictor:
    def __init__(self):
        self.models = {
            'XGBoost': xgb.XGBClassifier(
                n_estimators=200,        # More trees for better performance
                max_depth=3,             # Prevent overfitting
                learning_rate=0.1,       # Conservative learning rate
                subsample=0.8,           # Prevent overfitting
                colsample_bytree=0.8,    # Feature subsampling
                min_child_weight=1,      # Minimum sum of instance weight
                gamma=0,                 # Minimum loss reduction
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            ),

            'LightGBM': LGBMClassifier(
                n_estimators=200,
                learning_rate=0.1,
                num_leaves=31,           # Controls tree complexity
                class_weight='balanced',
                subsample=0.8,           # Sample ratio for training
                colsample_bytree=0.8,    # Feature sampling
                min_child_samples=20,    # Minimum samples per leaf
                random_state=42
            ),

            'Random Forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=None,          # Let trees grow fully
                min_samples_split=5,     # Minimum samples to split
                min_samples_leaf=2,      # Minimum samples per leaf
                max_features='sqrt',     # Use sqrt(n_features)
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),

            'SVM': SVC(
                kernel='rbf',
                C=1.0,                   # Regularization parameter
                gamma='scale',           # Kernel coefficient
                class_weight='balanced',
                probability=True,
                random_state=42
            ),

            'KNN': KNeighborsClassifier(
                n_neighbors=5,
                weights='distance',
                metric='minkowski',
                p=2,                     # Euclidean distance
                n_jobs=-1
            ),

            'Decision Tree': DecisionTreeClassifier(
                max_depth=5,             # Limit depth to prevent overfitting
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                criterion='gini',
                random_state=42
            ),

            # 'Logistic Regression': LogisticRegression(
            #     class_weight='balanced',
            #     max_iter=1000,
            #     C=1.0,                   # Inverse of regularization strength
            #     solver='lbfgs',
            #     multi_class='auto',
            #     n_jobs=-1
            # ),

            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=3,
                min_samples_split=5,
                min_samples_leaf=2,
                subsample=0.8,
                random_state=42
            )
        }

        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_model_name = None
        self.feature_order = None

    def preprocess_data(self, df, training=True):
        """Preprocess the dataset"""
        df_processed = df.copy()

        # Define feature groups
        self.categorical_features = ['PID', 'family', 'load']
        self.binary_features = ['TGN', 'TGN required', 'servers', 'servers required',
                              'EOR', 'EOR required', 'I', 'K', 'M', 'N', 'O', 'P']
        self.numerical_features = ['devices', 'device required']

        try:
            # Handle categorical features
            for feature in self.categorical_features:
                if training:
                    self.label_encoders[feature] = LabelEncoder()
                    df_processed[feature] = self.label_encoders[feature].fit_transform(df_processed[feature].astype(str))
                else:
                    df_processed[feature] = self.label_encoders[feature].transform(df_processed[feature].astype(str))

            # Convert binary features
            for feature in self.binary_features:
                df_processed[feature] = (df_processed[feature].astype(str).str.lower() == 'yes').astype(int)

            # Scale numerical features
            if training:
                df_processed[self.numerical_features] = self.scaler.fit_transform(df_processed[self.numerical_features])
            else:
                df_processed[self.numerical_features] = self.scaler.transform(df_processed[self.numerical_features])

            # Store or enforce feature order
            if training:
                self.feature_order = [col for col in df_processed.columns
                                    if col not in ['Target', 'Testbed']]
            else:
                # Ensure all required features are present
                missing_features = set(self.feature_order) - set(df_processed.columns)
                if missing_features:
                    raise ValueError(f"Missing features in prediction data: {missing_features}")

                # Reorder columns to match training data
                df_processed = df_processed[self.feature_order]

            return df_processed

        except Exception as e:
            print(f"Error in preprocessing: {str(e)}")
            raise
    def plot_model_comparison(self, results):
        """Plot model comparison"""
        models = list(results.keys())
        test_scores = [results[model]['test_score'] for model in models]
        cv_scores = [results[model]['cv_score'] for model in models]

        x = np.arange(len(models))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))
        rects1 = ax.bar(x - width/2, test_scores, width, label='Test Score')
        rects2 = ax.bar(x + width/2, cv_scores, width, label='CV Score')

        ax.set_ylabel('Accuracy')
        ax.set_title('Model Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45)
        ax.legend()

        plt.tight_layout()
        plt.show()

    def plot_confusion_matrices(self, results):
        """Plot confusion matrices for all models"""
        n_models = len(results)
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.ravel()

        for idx, (name, result) in enumerate(results.items()):
            sns.heatmap(result['confusion_matrix'],
                       annot=True,
                       fmt='d',
                       cmap='Blues',
                       ax=axes[idx])
            axes[idx].set_title(f'{name} Confusion Matrix')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')

        plt.tight_layout()
        plt.show()

    def plot_roc_curves(self, X_test, y_test):
        """Plot ROC curves for all models"""
        plt.figure(figsize=(10, 8))

        for name, model in self.models.items():
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(loc="lower right")
        plt.show()

    def train(self, df):
        """Train all models and select the best one"""
        print("Starting training process...")

        # Preprocess data
        df_processed = self.preprocess_data(df, training=True)
        print("Data preprocessing completed.")

        # Prepare features and target
        X = df_processed[self.feature_order]
        y = (df['Target'].astype(str).str.lower() == 'yes').astype(int)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print("Data split completed. Training set size:", len(X_train))

        # Balance training data
        oversampler = RandomOverSampler(random_state=42)
        X_train_balanced, y_train_balanced = oversampler.fit_resample(X_train, y_train)
        print("Data balancing completed. Balanced training set size:", len(X_train_balanced))

        # Train and evaluate all models
        best_score = 0
        results = {}

        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            try:
              #special handling for XGBoost
              if name == 'XGBoost':
                model.fit(X_train_balanced, y_train_balanced,
                          eval_set=[(X_test, y_test)],
                          # early_stopping_rounds=10,
                          verbose=True)
                #Manual cross-validation for XGBoost
                cv_scores = []
                from sklearn.model_selection import KFold
                kf = KFold(n_splits=5, shuffle=True, random_state=42)

                for train_idx, val_idx in kf.split(X_train_balanced):
                    X_fold_train, X_fold_val = X_train_balanced.iloc[train_idx], X_train_balanced.iloc[val_idx]
                    y_fold_train, y_fold_val = y_train_balanced.iloc[train_idx], y_train_balanced.iloc[val_idx]

                    # Create and train a new XGBoost model for each fold
                    fold_model = xgb.XGBClassifier(**self.models['XGBoost'].get_params())
                    fold_model.fit(X_fold_train, y_fold_train, verbose=False)

                    # Get fold score
                    fold_pred = fold_model.predict(X_fold_val)
                    fold_score = (fold_pred == y_fold_val).mean()
                    cv_scores.append(fold_score)

                cv_score = np.mean(cv_scores)


              else:
                  model.fit(X_train_balanced, y_train_balanced)
                  cv_score = cross_val_score(model, X_train_balanced, y_train_balanced, cv=5).mean()

              # Evaluate
              y_pred = model.predict(X_test)
              test_score = (y_pred == y_test).mean()
              # cv_score = cross_val_score(model, X_train_balanced, y_train_balanced, cv=5).mean()

              results[name] = {
                  'cv_score': cv_score,
                  'test_score': test_score,
                  'confusion_matrix': confusion_matrix(y_test, y_pred),
                  'classification_report': classification_report(y_test, y_pred)
              }

              print(f"{name} - Test Score: {test_score:.4f}, CV Score: {cv_score:.4f}")
              print("\nClassification Report:")
              print(results[name]['classification_report'])

              # Update best model
              if test_score > best_score:
                  best_score = test_score
                  self.best_model = model
                  self.best_model_name = name
            except Exception as e:
              print(f"Error occurred while training {name}: {str(e)}")
              continue


        # Plot comparisons only if we have results
        if results:
          try:
              self.plot_model_comparison(results)
              self.plot_confusion_matrices(results)
              self.plot_roc_curves(X_test, y_test)
          except Exception as e:
              print(f"Error in plotting: {str(e)}")

        if self.best_model is None:
          raise ValueError("No models were successfully trained")

        print(f"\nTraining completed. Best model: {self.best_model_name} (Score: {best_score:.4f})")
        return results

    def save_model(self, directory='models'):
        """Save the trained model and preprocessors"""
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Save the best model
        model_path = os.path.join(directory, f'{self.best_model_name}.joblib')
        joblib.dump(self.best_model, model_path)

        # Save label encoders
        encoders_path = os.path.join(directory, 'label_encoders.joblib')
        joblib.dump(self.label_encoders, encoders_path)

        # Save scaler
        scaler_path = os.path.join(directory, 'scaler.joblib')
        joblib.dump(self.scaler, scaler_path)

        # Save feature order
        feature_order_path = os.path.join(directory, 'feature_order.joblib')
        joblib.dump(self.feature_order, feature_order_path)

        print(f"Model and preprocessors saved in {directory}/")

    @classmethod
    def load_model(cls, directory='models'):
        """Load a trained model and preprocessors"""
        predictor = cls()

        # Load the best model
        model_files = [f for f in os.listdir(directory) if f.endswith('.joblib')
                      and not f.startswith('label_encoders')
                      and not f.startswith('scaler')
                      and not f.startswith('feature_order')]

        if not model_files:
            raise ValueError("No model file found")

        model_name = model_files[0].replace('.joblib', '')
        model_path = os.path.join(directory, model_files[0])
        predictor.best_model = joblib.load(model_path)
        predictor.best_model_name = model_name

        # Load label encoders
        encoders_path = os.path.join(directory, 'label_encoders.joblib')
        predictor.label_encoders = joblib.load(encoders_path)

        # Load scaler
        scaler_path = os.path.join(directory, 'scaler.joblib')
        predictor.scaler = joblib.load(scaler_path)

        # Load feature order
        feature_order_path = os.path.join(directory, 'feature_order.joblib')
        predictor.feature_order = joblib.load(feature_order_path)

        return predictor