from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
import joblib
import matplotlib
# Set matplotlib to use 'Agg' backend (non-interactive) to avoid GUI issues
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from testbed_predictor import TestbedPredictor  # Import your TestbedPredictor class

app = Flask(__name__)

# Define the path to the model and data
MODEL_DIR = 'models'
DATA_PATH = 'data/data_testbed.xlsx'  # Location of your Excel file
PLOT_DIR = 'static/plots'  # Directory to save plots

# Global variables
predictor = None
pids = []
families = []
load_values = ['high', 'low']

# Create directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs('static', exist_ok=True)

def load_predictor():
    """Load the trained model"""
    global predictor
    try:
        predictor = TestbedPredictor.load_model(MODEL_DIR)
        print(f"Model loaded successfully: {predictor.best_model_name}")
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

def load_options():
    """Load selection options from the dataset"""
    global pids, families
    try:
        if os.path.exists(DATA_PATH):
            df = pd.read_excel(DATA_PATH)
            pids = sorted(df['PID'].unique().tolist())
            families = sorted(df['family'].unique().tolist())
            print(f"Options loaded: {len(pids)} PIDs, {len(families)} families")
            return True
        else:
            print(f"Data file not found: {DATA_PATH}")
            return False
    except Exception as e:
        print(f"Error loading options: {str(e)}")
        return False

# Monkey patch the plotting methods to save figures instead of displaying them
def patch_testbed_predictor():
    original_plot_model_comparison = TestbedPredictor.plot_model_comparison
    original_plot_confusion_matrices = TestbedPredictor.plot_confusion_matrices
    original_plot_roc_curves = TestbedPredictor.plot_roc_curves
    
    def new_plot_model_comparison(self, results):
        try:
            plt.figure(figsize=(12, 6))
            models = list(results.keys())
            test_scores = [results[model]['test_score'] for model in models]
            cv_scores = [results[model]['cv_score'] for model in models]
            
            x = np.arange(len(models))
            width = 0.35
            
            plt.bar(x - width/2, test_scores, width, label='Test Score')
            plt.bar(x + width/2, cv_scores, width, label='CV Score')
            
            plt.ylabel('Accuracy')
            plt.title('Model Comparison')
            plt.xticks(x, models, rotation=45)
            plt.legend()
            plt.tight_layout()
            
            # Save instead of showing
            plt.savefig(os.path.join(PLOT_DIR, 'model_comparison.png'))
            plt.close()
        except Exception as e:
            print(f"Error in plot_model_comparison: {str(e)}")
    
    def new_plot_confusion_matrices(self, results):
        try:
            n_models = len(results)
            rows = (n_models + 3) // 4  # Ceiling division
            plt.figure(figsize=(20, 5*rows))
            
            for idx, (name, result) in enumerate(results.items()):
                plt.subplot(rows, 4, idx+1)
                plt.imshow(result['confusion_matrix'], interpolation='nearest', cmap='Blues')
                plt.title(f'{name} Confusion Matrix')
                plt.colorbar()
                plt.ylabel('True label')
                plt.xlabel('Predicted label')
                
                # Add text annotations
                cm = result['confusion_matrix']
                thresh = cm.max() / 2.
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        plt.text(j, i, format(cm[i, j], 'd'),
                                horizontalalignment="center",
                                color="white" if cm[i, j] > thresh else "black")
            
            plt.tight_layout()
            plt.savefig(os.path.join(PLOT_DIR, 'confusion_matrices.png'))
            plt.close()
        except Exception as e:
            print(f"Error in plot_confusion_matrices: {str(e)}")
    
    def new_plot_roc_curves(self, X_test, y_test):
        try:
            plt.figure(figsize=(10, 8))
            
            for name, model in self.models.items():
                if hasattr(model, 'predict_proba'):
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
            
            plt.savefig(os.path.join(PLOT_DIR, 'roc_curves.png'))
            plt.close()
        except Exception as e:
            print(f"Error in plot_roc_curves: {str(e)}")
    
    TestbedPredictor.plot_model_comparison = new_plot_model_comparison
    TestbedPredictor.plot_confusion_matrices = new_plot_confusion_matrices
    TestbedPredictor.plot_roc_curves = new_plot_roc_curves
    
    from sklearn.metrics import roc_curve, auc
    import numpy as np
    print("TestbedPredictor plotting methods patched to save figures instead of displaying them")

# Patch TestbedPredictor on startup
patch_testbed_predictor()

@app.route('/')
def index():
    """Render the main page"""
    plot_files = []
    for plot_type in ['model_comparison.png', 'confusion_matrices.png', 'roc_curves.png']:
        plot_path = os.path.join(PLOT_DIR, plot_type)
        if os.path.exists(plot_path):
            plot_files.append('/static/plots/' + plot_type)
    
    return render_template('index.html', 
                          pids=pids, 
                          families=families,
                          load_values=load_values,
                          model_loaded=predictor is not None,
                          plot_files=plot_files)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    if predictor is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Get data from form
        input_data = {
            'PID': request.form.get('pid'),
            'family': request.form.get('family'),
            'devices': int(request.form.get('devices')),
            'device required': int(request.form.get('devices_required')),
            'TGN': request.form.get('tgn'),
            'TGN required': request.form.get('tgn_required'),
            'servers': request.form.get('servers'),
            'servers required': request.form.get('servers_required'),
            'EOR': request.form.get('eor'),
            'EOR required': request.form.get('eor_required'),
            'I': request.form.get('i_support'),
            'K': request.form.get('k_support'),
            'M': request.form.get('m_support'),
            'N': request.form.get('n_support'),
            'O': request.form.get('o_support'),
            'P': request.form.get('p_support'),
            'load': request.form.get('load')
        }
        
        # Log input data for debugging
        print(f"Input data for prediction: {input_data}")
        
        # Create DataFrame from input data
        input_df = pd.DataFrame([input_data])
        
        # Preprocess the input data
        df_processed = predictor.preprocess_data(input_df, training=False)
        
        # Use the best model to make predictions
        prediction = predictor.best_model.predict(df_processed)
        probability = predictor.best_model.predict_proba(df_processed)
        
        # Create result dictionary
        result = {
            'prediction': 'yes' if prediction[0] == 1 else 'no',
            'probability': probability[0].tolist(),
            'confidence': float(max(probability[0]) * 100),
            'model_used': predictor.best_model_name
        }
        
        return jsonify(result)
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/train', methods=['POST'])
def train_model():
    """Train a new model from the dataset"""
    try:
        if not os.path.exists(DATA_PATH):
            return jsonify({'error': f'Data file not found: {DATA_PATH}'}), 404
            
        # Load data
        df = pd.read_excel(DATA_PATH)
        print(f"Data loaded for training: {len(df)} records")
        
        # Initialize predictor
        global predictor
        predictor = TestbedPredictor()
        
        # Train model
        results = predictor.train(df)
        
        # Save model
        predictor.save_model(MODEL_DIR)
        
        # Reload options
        load_options()
        
        return jsonify({
            'success': True, 
            'message': f'Model trained successfully! Best model: {predictor.best_model_name}',
            'plots': ['/static/plots/model_comparison.png', 
                      '/static/plots/confusion_matrices.png', 
                      '/static/plots/roc_curves.png']
        })
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({
        'status': 'ok',
        'model_loaded': predictor is not None,
        'data_available': os.path.exists(DATA_PATH)
    })


if __name__ == '__main__':
    # Load the model and options on startup
    model_loaded = load_predictor()
    options_loaded = load_options()
    
    if not model_loaded:
        print("Warning: Model not loaded. Training will be required.")
    
    if not options_loaded:
        print("Warning: Options not loaded. Default values will be used.")
        # Set default values if options weren't loaded
        pids = ['HX', 'GI', 'FX3', 'WF', 'FX2', 'FX'] 
        families = ['ararat', 'sundown', 'wolfridge', 'fretta', 'N3K']
    
    # Start the app
    app.run(debug=True, host='0.0.0.0', port=5000)