# imdb-sympathizers

## Requirements
```bash
Python 2.7.X 3.5.X
Google Cloud SDK 212.0.0
Ruby 2.5.X
```

## Usage
```bash
# Show tasks by make
make

# Train for development
make train-repl MODEL=cnn_maxpool OPT=adam

# Show tasks by rake
rake -T

# Train
nohup rake tasks:train &
tail -f nohup.out
```
## Results
https://gist.github.com/chck/c8dce4201778d78b8b409b135810cb49
