ADL Final
===

## Installation
Clone this repo and run
```
cd ADL-Final
apt-get update && apt-get install -y libsndfile1 ffmpeg
bash ./reinstall.sh
```

## Generate Predictions
To generate predictions from test dataset, run
```
python test.py \
  model.dataset.data_dir=<data_dir_with_dialogue_data> \
  split=<dev/test_seen/test_unseen>
```

The output files would be located in folder `predictions`.

## Notes

The structure of the data directory should be like:
```
data_dir/
    schemas.json
    train/
    dev/
    test_seen/
    test_unseen/
```

The files that are modified from the original NeMo repo to generate predictions are listed below.
```
NeMo/nemo/collections/nlp/data/dialogue_state_tracking/sgd/
    prediction_utils.py
    dataset.py
NeMo/nemo/collections/nlp/models/dialogue_state_tracking/
    sgdqa_model.py
```