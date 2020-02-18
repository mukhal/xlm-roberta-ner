import os
import logging



logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.DEBUG)

logger = logging.getLogger(__name__)

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, label_id, valid_ids=None, label_mask=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.label_id = label_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask


class NerProcessor:
    """Processor for the CoNLL-2003 data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_file(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_file(os.path.join(data_dir, "valid.txt")), "valid")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_file(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self):
        return ["IGNORE", "O", "B-PERS", "I-PERS", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]

    def _read_file(self, filename):
        '''
        read file
        '''
        f = open(filename)
        data = []
        sentence = []
        label = []

        for line in f:
            if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n" or line[0] == '.':
                if len(sentence) > 0:
                    data.append((sentence, label))
                    sentence = []
                    label = []
                continue

            splits = line.split(' ')
            sentence.append(splits[0])
            label.append(splits[-1][:-1])

        if len(sentence) > 0:
            data.append((sentence, label))
            sentence = []
            label = []
        return data

    def _create_examples(self, lines, set_type):
        examples = []

        for i, (sentence, label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join(sentence)
            text_b = None
            label = label
            examples.append(InputExample(
                guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, encode_method):
    """Converts a set of examples into XLMR compatible format

    * Labels are only assigned to the positions correspoinding to the first BPE token of each word.
    * Other positions are labeled with 0 ("IGNORE")


    """

    label_map = {label: i for i, label in enumerate(label_list)} # 0 label is to be ignored
    logger.debug("label_map = ")
    label_map
    features = []

    for (ex_index, example) in enumerate(examples):

        textlist = example.text_a.split(' ')
        labellist = example.label
        labels = []
        valid = []
        label_mask = []
        token_ids = []

        for i, word in enumerate(textlist):

            tokens = encode_method(word.strip()) # word token ids
            token_ids.extend(tokens) # all sentence token ids

            label_1 = labellist[i]
            

            for m in range(len(tokens)):

                if m == 0:  # only label the first BPE token of each work
                    labels.append(label_1)
                    valid.append(1)
                    label_mask.append(1)
                else:
                    labels.append("IGNORE") # unlabeled BPE token
                    label_mask.append(0)
                    valid.append(0)
        
        logger.info("token ids = ")
        logger.info(token_ids)
        logger.info("labels = ")
        logger.info(labels)
        logger.info("valid = ")
        logger.info(valid)

        assert len(valid) == len(labels)

        if len(token_ids) >= max_seq_length - 1: # trim extra tokens
            token_ids = token_ids[0:(max_seq_length)]
            labels = labels[0:(max_seq_length)]
            valid = valid[0:(max_seq_length)]
            label_mask = label_mask[0:(max_seq_length)]


        label_ids = []

        for i, _ in enumerate(token_ids):
            label_ids.append(label_map[labels[i]])
        
       
        assert len(token_ids) == len(label_ids)
        assert len(valid) == len(label_ids)

        input_mask = [1] * len(token_ids)
        label_mask = [1] * len(label_ids)

        while len(token_ids) < max_seq_length:
            token_ids.append(1) # padding idx
            input_mask.append(0)
            label_ids.append(0) # label padding idx
            valid.append(1)
            label_mask.append(0)

        while len(label_ids) < max_seq_length:
            label_ids.append(0)
            label_mask.append(0)
        
        assert len(token_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(valid) == max_seq_length
        assert len(label_mask) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" %
                        " ".join([str(x) for x in token_ids]))
            logger.info("input_mask: %s" %
                        " ".join([str(x) for x in input_mask]))
            # logger.info("label: %s (id = %d)" % (example.label, label_ids))

        features.append(
            InputFeatures(input_ids=token_ids,
                          input_mask=input_mask,
                          label_id=label_ids,
                          valid_ids=valid,
                          label_mask=label_mask))

    return features
