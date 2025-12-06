import os
import glob
from flask import Flask, render_template, send_file, jsonify, request
import pandas as pd
from datetime import datetime
import requests
import json
import re
import io

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
DATA_DIR = 'data'
CACHE_FILE = 'cache.json'

# --- API & Data Logic ---

def get_references_count(title):
    """
    Fetches the Wikipedia page content for the given title and counts '<ref' tags.
    """
    try:
        url = "https://ml.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "prop": "revisions",
            "rvprop": "content",
            "format": "json",
            "titles": title
        }
        # Timeout to prevent hanging
        response = requests.get(url, params=params, timeout=5)
        data = response.json()
        
        pages = data.get("query", {}).get("pages", {})
        for _, page in pages.items():
            if "missing" in page:
                return 0
            revisions = page.get("revisions", [])
            if revisions:
                content = revisions[0].get("*", "")
                # Count occurrences of <ref
                return len(re.findall(r'<ref', content, re.IGNORECASE))
    except Exception as e:
        print(f"Error fetching data for {title}: {e}")
    return 0

def enrich_data_with_refs(df):
    """
    Enriches the dataframe with a 'References' column by fetching data from MediaWiki API.
    Uses a local cache to store results.
    """
    if df.empty:
        return df

    cache = {}
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                cache = json.load(f)
        except Exception:
            cache = {}

    titles = df['Title'].unique()
    cache_updated = False

    for title in titles:
        # Simple caching mechanism
        if title not in cache:
            # print(f"Fetching references for: {title}") # Reduced logs
            count = get_references_count(title)
            cache[title] = count
            cache_updated = True
    
    if cache_updated:
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False, indent=4)

    # Map the cache to the dataframe
    df['References'] = df['Title'].map(cache).fillna(0).astype(int)
    return df

def load_dataframe(filename):
    """
    Reads a specific CSV file from the data directory.
    """
    csv_path = os.path.join(DATA_DIR, filename)
    
    if not os.path.exists(csv_path):
        return pd.DataFrame()

    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
        # Ensure column names match expected standard (simple normalization)
        # Assuming format is consistent for now based on user files
        
        if 'CreateTime' in df.columns:
            df['CreateTime'] = pd.to_datetime(df['CreateTime'], format='%Y%m%d%H%M%S', errors='coerce')
        
        # Enrich
        df = enrich_data_with_refs(df)
        return df
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return pd.DataFrame()

# --- Routes ---

@app.route('/')
def index():
    """
    Lists all CSV files in the data directory and extracts summary metadata.
    """
    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    events = []
    
    # Sort files by modification time (newest first)
    files.sort(key=os.path.getmtime, reverse=True)

    for f in files:
        basename = os.path.basename(f)
        clean_name = basename.replace('.csv', '').replace('_', ' ').replace('-', ' ').title()
        
        # Default metadata
        meta = {
            "name": clean_name,
            "filename": basename,
            "articles": 0,
            "editors": 0,
            "date_range": "Unknown Date",
            "bytes": 0
        }

        try:
            # Quick read for metadata
            df = pd.read_csv(f, encoding='utf-8')
            
            if not df.empty:
                meta['articles'] = len(df)
                
                if 'Creator' in df.columns:
                    meta['editors'] = df['Creator'].nunique()
                
                if 'LastSize' in df.columns:
                    meta['bytes'] = int(df['LastSize'].sum())
                
                if 'CreateTime' in df.columns:
                    # Convert to datetime to find range
                    dates = pd.to_datetime(df['CreateTime'], format='%Y%m%d%H%M%S', errors='coerce').dropna()
                    if not dates.empty:
                        start = dates.min().strftime('%b %d, %Y')
                        end = dates.max().strftime('%b %d, %Y')
                        if start == end:
                            meta['date_range'] = start
                        else:
                            meta['date_range'] = f"{start} - {end}"
        except Exception as e:
            print(f"Error reading {basename}: {e}")

        events.append(meta)
    
    return render_template('index.html', events=events)

@app.route('/event/<filename>')
def event_dashboard(filename):
    """
    Renders the dashboard for a specific file.
    """
    # Just render the template. The template will call the stats API with the filename.
    return render_template('dashboard.html', filename=filename)

@app.route('/api/stats/<filename>')
def stats(filename):
    """
    Returns JSON stats for the specific file.
    """
    df = load_dataframe(filename)
    
    if df.empty:
        return jsonify({
            "summary": {"total_articles": 0, "total_bytes": 0, "total_edits": 0, "active_editors": 0},
            "chart_data": [],
            "articles": []
        })

    # Summary KPIs
    total_articles = len(df)
    total_bytes = int(df['LastSize'].sum()) if 'LastSize' in df.columns else 0
    total_edits = int(df['Edits'].sum()) if 'Edits' in df.columns else 0
    active_editors = df['Creator'].nunique() if 'Creator' in df.columns else 0

    # Chart Data
    chart_data = []
    if 'Creator' in df.columns and 'LastSize' in df.columns:
        top_creators = df.groupby('Creator')['LastSize'].sum().nlargest(10).reset_index()
        chart_data = top_creators.to_dict(orient='records')

    # Detailed List
    df['status'] = 'Created'
    if 'CreateTime' in df.columns:
        df['CreateTime'] = df['CreateTime'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    articles = df.to_dict(orient='records')

    response_data = {
        "summary": {
            "total_articles": total_articles,
            "total_bytes": total_bytes,
            "total_edits": total_edits,
            "active_editors": active_editors
        },
        "chart_data": chart_data,
        "articles": articles
    }
    
    return jsonify(response_data)

@app.route('/export/<filename>')
def export(filename):
    """
    Generates and downloads the processed report for the specific file.
    """
    df = load_dataframe(filename)
    if df.empty:
        return "No data available to export", 404

    buffer = io.StringIO()
    df.to_csv(buffer, index=False, encoding='utf-8-sig')
    buffer.seek(0)
    
    mem = io.BytesIO()
    mem.write(buffer.getvalue().encode('utf-8-sig'))
    mem.seek(0)
    
    safe_name = filename.replace('.csv', '_report.csv')

    return send_file(
        mem,
        mimetype='text/csv',
        as_attachment=True,
        download_name=safe_name
    )

@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Handles CSV file uploads.
    """
    from werkzeug.utils import secure_filename
    
    if 'file' not in request.files:
        return 'No file part', 400
    
    file = request.files['file']
    
    if file.filename == '':
        return 'No selected file', 400
    
    if file and file.filename.lower().endswith('.csv'):
        filename = secure_filename(file.filename)
        # secure_filename might return empty if filename is only special chars or slashes
        if not filename:
            filename = f"upload_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
            
        file.save(os.path.join(DATA_DIR, filename))
        return 'File uploaded successfully', 200
    else:
        return 'Invalid file type. Only CSV allowed.', 400

@app.route('/delete/<filename>', methods=['POST'])
def delete_file(filename):
    """
    Deletes a specific CSV file.
    """
    from werkzeug.utils import secure_filename
    
    filename = secure_filename(filename)
    file_path = os.path.join(DATA_DIR, filename)
    
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            return 'File deleted successfully', 200
        except Exception as e:
            return f'Error deleting file: {str(e)}', 500
    else:
        return 'File not found', 404

if __name__ == '__main__':
    # Ensure data directory exists
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    app.run(debug=True)
    app.run(debug=True)
