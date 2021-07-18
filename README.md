# PocketVAE

This is a drum groove generation and control model, which uses VQ-VAE to first generate the note scores (without any playing style) and then uses VAEs to add velocity and microtiming values on top of the generated note scores. 


## Links 
* [Paper](https://arxiv.org/abs/2107.05009) : Presented as my master's thesis in June 2021  
* [WIP - Accompanying website]()


## Setup
Clone the repo first and `cd` into the directory.  
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
I apologize in advance that some of the  data mentioned below are not publicly available. However, I believe that the results will not degrade too much as long as you don't use the "genres" to generate with control.  

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
@article{lee2021pocketvae,
  title={PocketVAE: A Two-step Model for Groove Generation and Control},
  author={Lee, Kyungyun and Kim, Wonil and Nam, Juhan},
  journal={arXiv preprint arXiv:2107.05009},
  year={2021}
}
```


