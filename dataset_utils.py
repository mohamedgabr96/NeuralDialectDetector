import torch
import logging
import os
from torch.utils.data import TensorDataset, RandomSampler

logger = logging.getLogger(__name__)

class NADIDataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, sentences, labels):
        'Initialization'
        self.sentences = sentences
        self.labels = labels

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'

        X = self.sentences[index]
        y = self.labels[index]

        return X, y


def parse_classes_list(path_to_folder):
    classes_path = os.path.join(path_to_folder, "classes.txt")
    with open(classes_path, encoding="utf-8") as file_open:
        lines = file_open.readlines()
    return [line.strip("\n") for line in lines]


def parse_data(path_to_file, separator="\t"):
    with open(path_to_file, encoding="utf-8") as file_open:
        lines = file_open.readlines()
    
    lines_split = [line.split(separator)[1:] for line in lines]
    return lines_split


def parse_and_generate_loader(path_to_data_folder, tokenizer, params, classes_list, split_set="train", locale="ar"):
    data_examples = parse_data(os.path.join(path_to_data_folder, f"{split_set}-{locale}.tsv"))
    dataset = load_and_cache_examples(data_examples, tokenizer, classes_list)
    data_sampler = RandomSampler(dataset)
    generator = torch.utils.data.DataLoader(dataset, sampler=data_sampler, **params)

    return generator


def parse_and_generate_loaders(path_to_data_folder, tokenizer, batch_size=2):
    params = {'batch_size': batch_size}
    classes_list = parse_classes_list(path_to_data_folder)
    training_generator = parse_and_generate_loader(path_to_data_folder, tokenizer, params, classes_list, split_set="train", locale="ar")
    dev_generator = parse_and_generate_loader(path_to_data_folder, tokenizer, params, classes_list, split_set="dev", locale="ar")
    test_generator = parse_and_generate_loader(path_to_data_folder, tokenizer, params, classes_list, split_set="test", locale="ar")

    return training_generator, dev_generator, test_generator, len(classes_list)

# From : https://github.com/monologg/JointBERT/blob/master/predict.py
def load_and_cache_examples(examples, tokenizer, classes_list, pad_token_ignore_index=0, max_seq_len=128):

        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
    pad_token_label_id = pad_token_ignore_index
    features = convert_examples_to_features(examples, classes_list, max_seq_len, tokenizer,
                                            pad_token_label_id=pad_token_label_id)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f[0] for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f[1] for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f[2] for f in features], dtype=torch.long)
    all_class_label_ids = torch.tensor([f[3] for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask,
                            all_token_type_ids, all_class_label_ids)
    return dataset


# From : https://github.com/monologg/JointBERT/blob/master/predict.py
def convert_examples_to_features(examples, classes_list, max_seq_len, 
                                 tokenizer,
                                 pad_token_label_id=-100,
                                 cls_token_segment_id=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 mask_padding_with_zero=True):
    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        # Tokenize word by word (for NER)
        tokens = []
        sentence_whitespace = example[0].split(' ')
        for word in sentence_whitespace:
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token]  # For handling the bad-encoded word
            tokens.extend(word_tokens)

        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        if len(tokens) > max_seq_len - special_tokens_count:
            tokens = tokens[:(max_seq_len - special_tokens_count)]

        # Add [SEP] token
        tokens += [sep_token]
        token_type_ids = [sequence_a_segment_id] * len(tokens)

        # Add [CLS] token
        tokens = [cls_token] + tokens
        token_type_ids = [cls_token_segment_id] + token_type_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_len)
        assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_ids), max_seq_len)

        class_label_id = classes_list.index(example[1].strip("\n"))

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("class_label: %s (id = %d)" % (example[1].strip("\n"), class_label_id))

        features.append((input_ids,
                          attention_mask,
                          token_type_ids,
                          class_label_id
                          ))

    return features