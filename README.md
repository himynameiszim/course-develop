# course-develop
Bespoke Course Creator: From Needs analysis to Course Development

## Set-up
1. Clone the repository
```
git clone https://github.com/himynameiszim/course-develop.git
cd course-develop/
```

2. Virtual environment
```
conda create -n <env-name> python=3.10
conda activate <env-name>
```

3. Installing requirements
```
pip3 install -r requirements.txt
```

4. Set-up environment for LLM <br/>
- Download [Ollama](http://ollama.com/) and pull/run some models for *ChatOllama*, *OllamaEmbeddings* and *KeyBERT* (this list might get updated soon).
- One might need to change `base_url` and other parameters in model initialization process of `main.py`.

## Run
Run the pipeline with
```
python3 main.py
```