

Here, we provide an example of training a Bert Model with ColossalAI from scratch. The data processing step is adapted from MLPerf.
```
dependencies --> ColossalAI, transformers
```

## Download Wiki dataset

Download files after the uncompress, extract, clean up and dataset seperation steps from a [Google Drive location](https://drive.google.com/corp/drive/u/0/folders/1cywmDnAsrP5-2vsr8GDc6QUc7VWe-M3v).  The total size is about 4GB.

### Files in ./results directory:

| File                | Size (bytes) | MD5                              |
| ------------------- | -----------: | -------------------------------- |
| eval.md5            |       330000 | 71a58382a68947e93e88aa0d42431b6c |
| eval.txt            |     32851144 | 2a220f790517261547b1b45ed3ada07a |
| part-00000-of-00500 |     27150902 | a64a7c31eff5cd38ae6d94f7a6229dab |
| part-00001-of-00500 |     27198569 | 549a9ed4f805257245bec936563abfd0 |
| part-00002-of-00500 |     27395616 | 1a1366ddfc03aef9d41ce552ee247abf |
| ...                 |              |                                  |
| part-00497-of-00500 |     24775043 | 66835aa75d4855f2e678e8f3d73812e9 |
| part-00498-of-00500 |     24575505 | e6d68a7632e9f4aa1a94128cce556dc9 |
| part-00499-of-00500 |     21873644 | b3b087ad24e3770d879a351664cebc5a |

Each of `part-00xxx-of-00500` and eval.txt contains one sentence of an article in one line and different articles separated by blank line.

## Generate the TFRecords for Wiki dataset

The second step is to preprocess the dataset to randomly generate inputs for MLM and NSP tasks. The output pre-training dataset size is 365GB of tfrecords, and usually takes dozens of hours to generate.

The [create_pretraining_data.py](./cleanup_scripts/create_pretraining_data.py) script tokenizes the words in a sentence using [tokenization.py](./creanup_scripts/tokenization.py) and `vocab.txt` file. Then, random tokens are masked using the strategy where 80% of time, the selected random tokens are replaced by `[MASK]` tokens, 10% by a random word and the remaining 10% left as is. This process is repeated for `dupe_factor` number of times, where an example with `dupe_factor` number of different masks are generated and written to TFRecords.

```shell
# Generate one TFRecord for each part-00XXX-of-00500 file. The following command is for generating one corresponding TFRecord
python3 create_pretraining_data.py \
   --input_file=<path to ./results of previous step>/part-00XXX-of-00500 \
   --output_file=<tfrecord dir>/part-00XXX-of-00500 \
   --vocab_file=<path to downloaded vocab.txt> \
   --do_lower_case=True \
   --max_seq_length=512 \
   --max_predictions_per_seq=76 \
   --masked_lm_prob=0.15 \
   --random_seed=12345 \
   --dupe_factor=10
```
where

- `dupe_factor`:  Number of times to duplicate the dataset and write to TFrecords. Each of the duplicate example has a different random mask
- `max_sequence_length`: Maximum number of tokens to be present in a single example
-`max_predictions_per_seq`: Maximum number of tokens that can be masked per example
- `masked_lm_prob`: Masked LM Probability
- `do_lower_case`: Whether the tokens are to be converted to lower case or not

After the above command is called 500 times, once per `part-00XXX-of-00500` file, there would be 500 TFrecord files totalling to ~365GB.

**Note: It is extremely critical to set the value of `random_seed` to `12345` so that th examples on which the training is evaluated is consistent among users.**

Use the following steps for the eval set:

```shell
python3 create_pretraining_data.py \
  --input_file=<path to ./results>/eval.txt \
  --output_file=<output path for eval_intermediate> \
  --vocab_file=<path to vocab.txt> \
  --do_lower_case=True \
  --max_seq_length=512 \
  --max_predictions_per_seq=76 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=10

python3 pick_eval_samples.py \
  --input_tfrecord=<path to eval_intermediate from the previous command> \
  --output_tfrecord=<output path for eval_10k> \
  --num_examples_to_pick=10000
```

### TFRecord Features

The examples in the TFRecords have the following key/values in its features dictionary:

| Key                  |    Type | Value                                                        |
| -------------------- | ------: | ------------------------------------------------------------ |
| input_ids            |   int64 | Input token ids, padded with 0's to max_sequence_length.     |
| input_mask           |   int32 | Mask for padded positions. Has 0's on padded positions and 1's elsewhere in TFRecords. |
| segment_ids          |   int32 | Segment ids. Has 0's on the positions corresponding to the first segment and 1's on positions corresponding to the second segment. The padded positions correspond to 0. |
| masked_lm_ids        |   int32 | Ids of masked tokens, padded with 0's to max_predictions_per_seq to accommodate a variable number of masked tokens per sample. |
| masked_lm_positions  |   int32 | Positions of masked tokens in the input_ids tensor, padded with 0's to max_predictions_per_seq. |
| masked_lm_weights    | float32 | Mask for masked_lm_ids and masked_lm_positions. Has values 1.0 on the positions corresponding to actually masked tokens in the given sample and 0.0 elsewhere. |
| next_sentence_labels |   int32 | Carries the next sentence labels.                            |

### Some stats of the generated tfrecords:

| File                |    Size (bytes) |
| ------------------- | --------------: |
| eval_intermediate   |     843,343,183 |
| eval_10k            |      25,382,591 |
| part-00000-of-00500 |     514,241,279 |
| part-00499-of-00500 |     898,392,312 |
| part-00XXX-of-00500 | 391,434,110,129 |

## Load tfrecords

1. Make index files
   

  To run the script setup a virtualenv with the following libraries installed.

  - `nvidia.dali`: See [documentation](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/installation.html)

  ```
  python3 make_idx.py --tfrecord_train="path/to/train_tfrecords" --tfrecord_eval="path/to/eval_tfrecords" --output_idx="path/to/output_idx"
  ```

  The index files can be obtained from TFRecord files by using the make_idx.py script that is distributed with DALI. There should be one index file for every TFRecord file.
2. Instantiate a DALIdataloader to read tfrecords

  ```python
import glob
import os
from bert_dali_dataloader import DaliDataloader

# We also provide a DALI dataloader which can read the processed dataset. 
# 2 inputs: the repo that stores 500 training tfrecords, the corresponding index repo
def build_dali_train('path/to/train_tfrecords', 'path/to/train_tfrecords_idx'):
    train_pat = os.path.join('path/to/train_tfrecords')
    train_idx_pat = os.path.join('path/to/train_tfrecords_idx')
    return DaliDataloader(
        sorted(glob.glob(train_pat)),
        sorted(glob.glob(train_idx_pat)),
        batch_size=BATCH_SIZE,
        shard_id=SHARD_ID,
        num_shards=NUM_SHARDS,
        num_threads=16,
        training=True,
        cuda=True,  
    )
  ```
