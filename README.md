# PocketVAE

*Code for "PocketVAE: A Two-Step Model for Groove Generation and Control", Kyungyun Lee, Wonil Kim, Juhan Nam, May 2021*

## Links 
* [Paper]() 
* [Accompanying website]()

## Demo
[![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19d9BwENCxiokZBh6qtwLO8siDIzZhrWc#scrollTo=7JfM7bEFtoXV&forceEdit=true&sandboxMode=true)


## Setup
```
pip install -e .  
pip install -r requirements.txt
```

### Directory layout
* `src/run_{model_type}.py` : training code for model
* `src/models/{model_type}.py` : model architecture code 
* `src/config/` : configuration files 
* `datasets/` : directory for data related code
* `weights/`: trained model weights 

model_type 
- `onestep` : one-step model 
- `vae`: pocketVAE (vae)
- `vqvae`: pocketVAE (vqvae) 


## To train
### 1. Prepare dataset & preprocess

### Datasets 

* [GrooveMIDI](https://magenta.tensorflow.org/datasets/groove)
* [Reddit drums](https://www.reddit.com/r/WeAreTheMusicMakers/comments/3anwu8/the_drum_percussion_midi_archive_800k/) (Filtered extensively)
* [BFD3](https://www.fxpansion.com/products/bfd3/) (not free)
* [GrooveMonkee Mega pack](https://groovemonkee.com/products/mega-pack) (not free)

Need to setup the path to data directory in `config/config.yaml`

```
cd datasets
python prepare_datasets.py
```


### 2. Train 
```
cd src
python main.py model=vqvae model_name=some_model_name use_genre_cond=True
"""
Options 

split_numer=0,1,..,9 (default=0)
gpus=? (default='0')
use_genre_cond=True (default=False)
use_vel_cond=True (default=False)
use_time_cond=True (default=False)
"""
```
### 3. Evaluate 

```
cd src
python evaluate.py model=vqvae model_name=some_model_name
```




## Cite 
```
NOT YET
```


