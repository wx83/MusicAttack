# Adversarial Attack on Music Generation

Weihan Xu


Introduction:

Recent advancements in text-to-music generation models, such as MusicGen and AudioLM, pose a potential threat to musicians in the music industry. These models can learn to mimic artistic styles, raising significant copyright concerns. In this paper, we first explore these concerns by gathering insights from both musicians and the broader community regarding music generation models. We then propose a novel perturbation-attack-and-finetuning paradigm designed to mitigate style replication in text-to-music generation models. Our evaluations, including both subjective and objective assessments, demonstrate that our approach effectively addresses the issue of style mimicry while maintaining the quality of the generated music tracks. 


### Step 1: Dataset Construction
Prepare your dataset
- **my_dataset**: Root directory of the dataset.
- **my_dataset/data/train**: Contains training data files.
- **my_dataset/data/valid**: Contains validation data files.
- **my_dataset/metadata.csv**: data_prep.py

    A CSV file with two columns:
    - `file_name`: The name of the data file.
    - `text`: The corresponding text or label for the file.

More datail can be found: https://huggingface.co/docs/datasets/audio_dataset

### Step 2: Perturb your data
**attack.py**

### Step 3: Finetune with MusicGen
**musicgen.py**

### Step 4: Evaluation
**eval.py**