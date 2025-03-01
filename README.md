# Testbed Predictor

## Project Structure

Create the following directory structure:

```
testbed-predictor/
├── app.py                      # Flask application
├── testbed_predictor.py        # Your TestbedPredictor class file
├── requirements.txt            # Dependencies
├── templates/                  # HTML templates
│   └── index.html              # Main UI page
├── static/                     # Static assets (if needed)
├── data/                       # Data directory
│   └── data_testbed.xlsx       # Your dataset file
└── models/                     # Directory for saved models
```
## Step 1: Create the Files

1. Save your `TestbedPredictor` class as `testbed_predictor.py`
2. Create `app.py` using the provided Flask code
3. Create `templates/index.html` with the provided HTML template
4. Create a `requirements.txt` file (content below)


## Step 2: Create requirements.txt or get it from my github

```
flask==2.3.3
pandas==2.0.3
numpy==1.25.2
scikit-learn==1.3.0
xgboost==1.7.6
lightgbm==4.0.0
imbalanced-learn==0.11.0
joblib==1.3.2
openpyxl==3.1.2
gunicorn==21.2.0
```

## Step 3: Set Up Environment

```bash
# Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Step 4: Prepare Data

Place your `data_testbed.xlsx` file in the `data/` directory:

```bash
mkdir -p data
cp /path/to/your/data_testbed.xlsx data/
```

## Step 5: Start the Application

```bash
# Run the Flask app
python app.py
```

The application will be available at [http://localhost:5000](http://localhost:5000)

## Step 6: Production Deployment

### Option 1: Gunicorn + Nginx (for Linux/MacOS)

1. Install Gunicorn:
```bash
pip install gunicorn
```

2. Create a systemd service file (testbed-predictor.service):
```
[Unit]
Description=Testbed Predictor Service
After=network.target

[Service]
User=<your-user>
WorkingDirectory=/path/to/testbed-predictor
ExecStart=/path/to/venv/bin/gunicorn -w 4 -b 127.0.0.1:5000 app:app
Restart=always

[Install]
WantedBy=multi-user.target
```

3. Set up Nginx as a reverse proxy:
```
server {
    listen 80;
    server_name your-domain-or-ip;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Option 2: Docker Deployment

1. Create a Dockerfile:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p models

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
```

2. Build and run the Docker container:
```bash
docker build -t testbed-predictor .
docker run -d -p 80:5000 -v $(pwd)/data:/app/data --name testbed-app testbed-predictor
```

## Troubleshooting

1. **Model loading fails**: Make sure you've trained a model first using the "Train Model" button or pre-train a model and place the files in the `models/` directory.

2. **Missing dependencies**: If you get import errors, ensure all dependencies are installed with `pip install -r requirements.txt`.

3. **File not found errors**: Check that your `data_testbed.xlsx` file is correctly placed in the `data/` directory.

4. **Prediction errors**: Check the logs for detailed error messages. Most common issues are related to feature mismatch between training and prediction.