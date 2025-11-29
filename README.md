# MoviCue (v1.7)
Personal Project to learn ML basics

# How to Demo
### Prerequisites
```bash
- Python 3.12
- pip
```
### Setting Up
```bash
git clone https://github.com/rjd14me/MoviCue.git  
cd MoviCue
```
```bash
pip install -r requirements.txt
```
### Getting Started
```bash
python src/api.py
```
Then go to [this web address](http://127.0.0.1:8000)
### Try it Out
- Type any movie title in the search box,and pick one from the drop-down suggestions.
- See the list of recommended movies based on similarity scores.  
## Features
- Uses data from the MovieLens set  (`movies.csv`, `links.csv` and `ratings.csv`)
- Learns movie embeddings from movie metadata.
- Displays a simple web UI where you type a movie title and get similar movies.
- Includes a live search dropdown that updates as you type, allowing users to select specific movies quickly.
- Fuzzy Search Capabilities
- Random Movie searching
- Filters to filter by year and ratings

## Key skills demonstrated:
- Data Handling
- Python and project structure
- FastAPI
- Simple deep learning (PyTorch)
- Frontend (HTML/CSS)

```bash
Modules: pandas, numpy, fastapi, uvicorn[standard], jinja2, torch, scikit-learn
```

## Project layout

```text
movie-recommender/
  data/
    movies.csv
    ratings.csv
    links.csv            
  src/
    api.py
    data_preparation.py
    models/
      deep_model.py
  templates/
    index.html
  static/
    style.css
  requirements.txt
  README.md
  
```