create  a environment
```bash
conda create -n env name python=3.7 
```
activate environment
```bash
conda activate env
```
install requirements file
```bash 
pip install -r requirements.txt
```

```bash
git init
```

```bash
dvc init
```

```bash
git add .
```

```bash
git commit -m "commit"
```

```bash
git remote add origin https://github.com/dk6304273089/DHEE.git
git branch -M main
git push origin main
```
retraining approach
```bash
dvc repro
```
To host our prediction page on browser
```bash
streamlit run app.py
```
 