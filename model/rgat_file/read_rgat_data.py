import json
from collections import Counter, defaultdict
from nltk import word_tokenize
from copy import copy, deepcopy

def read_sentence_depparsed(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
        return data

def get_dataset(dataset_name):
    rest_train = './rgat_data/semeval14/Restaurants_Train_v2_biaffine_depparsed_with_energy.json'
    rest_test = './rgat_data/semeval14/Restaurants_Test_Gold_biaffine_depparsed_with_energy.json'

    laptop_train = './rgat_data/semeval14/Laptop_Train_v2_biaffine_depparsed.json'
    laptop_test = './rgat_data/semeval14/Laptops_Test_Gold_biaffine_depparsed.json'

    twitter_train = './rgat_data/twitter/train_biaffine.json'
    twitter_test = './rgat_data/twitter/test_biaffine.json'
    ds_train = {'14semeval_rest': rest_train,
                '14semeval_laptop': laptop_train, 'Twitter': twitter_train}
    ds_test = {'14semeval_rest': rest_test,
               '14semeval_laptop': laptop_test, 'Twitter': twitter_test}
    train = list(read_sentence_depparsed(ds_train[dataset_name]))
    test = list(read_sentence_depparsed(ds_test[dataset_name]))
    return train, test

def get_rolled_and_unrolled_data(input_data, multi_hop,add_non_connect,max_hop):

    # A hand-picked set of part of speech tags that we see contributes to ABSA.
    opinionated_tags = ['JJ', 'JJR', 'JJS', 'RB', 'RBR',
                        'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']

    all_rolled = []
    all_unrolled = []
    mixed_rolled = []
    mixed_unrolled = []

    unrolled = []
    mixed = []
    unrolled_ours = []
    mixed_ours = []

    # Make sure the tree is successfully built.
    zero_dep_counter = 0

    # Sentiment counters
    total_counter = defaultdict(int)
    mixed_counter = defaultdict(int)
    sentiments_lookup = {'negative': 0, 'positive': 1, 'neutral': 2}

    tree_samples = []
    # for seeking 'but' examples
    for e in input_data:
        e['tokens'] = [x.lower() for x in e['tokens']]
        aspects = []
        sentiments = []
        froms = []
        tos = []
        dep_tags = []
        dep_index = []
        dep_dirs = []

        # Classify based on POS-tags

        pos_class = e['tags']

        # Iterate through aspects in a sentence and reshape the dependency tree.
        for i in range(len(e['aspect_sentiment'])):
            aspect = e['aspect_sentiment'][i][0].lower()
            # We would tokenize the aspect while at it.
            aspect = word_tokenize(aspect)
            sentiment = sentiments_lookup[e['aspect_sentiment'][i][1]]
            frm = e['from_to'][i][0]
            to = e['from_to'][i][1]

            aspects.append(aspect)
            sentiments.append(sentiment)
            froms.append(frm)
            tos.append(to)

            # Center on the aspect.
            dep_tag, dep_idx, dep_dir = reshape_dependency_tree_new(frm, to, e['dependencies'],
                                                       multi_hop=multi_hop, add_non_connect=add_non_connect, tokens=e['tokens'], max_hop=max_hop)

            # Because of tokenizer differences, aspect opsitions are off, so we find the index and try again.
            if len(dep_tag) == 0:
                zero_dep_counter += 1
                as_sent = e['aspect_sentiment'][i][0].split()
                as_start = e['tokens'].index(as_sent[0])
                # print(e['tokens'], e['aspect_sentiment'], e['dependencies'],as_sent[0])
                as_end = e['tokens'].index(
                    as_sent[-1]) if len(as_sent) > 1 else as_start + 1
                print("Debugging: as_start as_end ", as_start, as_end)
                dep_tag, dep_idx, dep_dir = reshape_dependency_tree_new(as_start, as_end, e['dependencies'],
                                                           multi_hop=multi_hop, add_non_connect=add_non_connect, tokens=e['tokens'], max_hop=max_hop)
                if len(dep_tag) == 0:  # for debugging
                    print("Debugging: zero_dep",
                          e['aspect_sentiment'][i][0], e['tokens'])
                    print("Debugging: ". e['dependencies'])
                else:
                    zero_dep_counter -= 1

            dep_tags.append(dep_tag)
            dep_index.append(dep_idx)
            dep_dirs.append(dep_dir)

            total_counter[e['aspect_sentiment'][i][1]] += 1

            # Unrolling
            all_unrolled.append(
                {'sentence': e['tokens'], 'tags': e['tags'], 'pos_class': pos_class, 'aspect': aspect, 'sentiment': sentiment,
                    'predicted_dependencies': e['predicted_dependencies'], 'predicted_heads': e['predicted_heads'],
                 'from': frm, 'to': to, 'dep_tag': dep_tag, 'dep_idx': dep_idx, 'dep_dir':dep_dir,'dependencies': e['dependencies']})


        # All sentences with multiple aspects and sentiments rolled.
        all_rolled.append(
            {'sentence': e['tokens'], 'tags': e['tags'], 'pos_class': pos_class, 'aspects': aspects, 'sentiments': sentiments,
             'from': froms, 'to': tos, 'dep_tags': dep_tags, 'dep_index': dep_index, 'dependencies': e['dependencies']})

        # Ignore sentences with single aspect or no aspect
        if len(e['aspect_sentiment']) and len(set(map(lambda x: x[1], e['aspect_sentiment']))) > 1:
            mixed_rolled.append(
                {'sentence': e['tokens'], 'tags': e['tags'], 'pos_class': pos_class, 'aspects': aspects, 'sentiments': sentiments,
                 'from': froms, 'to': tos, 'dep_tags': dep_tags, 'dep_index': dep_index, 'dependencies': e['dependencies']})

            # Unrolling
            for i, as_sent in enumerate(e['aspect_sentiment']):
                mixed_counter[as_sent[1]] += 1
                mixed_unrolled.append(
                    {'sentence': e['tokens'], 'tags': e['tags'], 'pos_class': pos_class, 'aspect': aspects[i], 'sentiment': sentiments[i],
                     'from': froms[i], 'to': tos[i], 'dep_tag': dep_tags[i], 'dep_idx': dep_index[i], 'dependencies': e['dependencies']})


    return all_rolled, all_unrolled, mixed_rolled, mixed_unrolled


def reshape_dependency_tree_new(as_start, as_end, dependencies, multi_hop=False, add_non_connect=False, tokens=None, max_hop = 5):
    '''
    Adding multi hops
    This function is at the core of our algo, it reshape the dependency tree and center on the aspect.
    In open-sourced edition, I choose not to take energy(the soft prediction of dependency from parser)
    into consideration. For it requires tweaking allennlp's source code, and the energy is space-consuming.
    And there are no significant difference in performance between the soft and the hard(with non-connect) version.

    '''
    dep_tag = []
    dep_idx = []
    dep_dir = []
    # 1 hop

    for i in range(as_start, as_end):
        for dep in dependencies:
            if i == dep[1] - 1:
                # not root, not aspect
                if (dep[2] - 1 < as_start or dep[2] - 1 >= as_end) and dep[2] != 0 and dep[2] - 1 not in dep_idx:
                    if str(dep[0]) != 'punct':  # and tokens[dep[2] - 1] not in stopWords
                        dep_tag.append(dep[0])
                        dep_dir.append(1)
                    else:
                        dep_tag.append('<pad>')
                        dep_dir.append(0)
                    dep_idx.append(dep[2] - 1)
            elif i == dep[2] - 1:
                # not root, not aspect
                if (dep[1] - 1 < as_start or dep[1] - 1 >= as_end) and dep[1] != 0 and dep[1] - 1 not in dep_idx:
                    if str(dep[0]) != 'punct':  # and tokens[dep[1] - 1] not in stopWords
                        dep_tag.append(dep[0])
                        dep_dir.append(2)
                    else:
                        dep_tag.append('<pad>')
                        dep_dir.append(0)
                    dep_idx.append(dep[1] - 1)

    if multi_hop:
        current_hop = 2
        added = True
        while current_hop <= max_hop and len(dep_idx) < len(tokens) and added:
            added = False
            dep_idx_temp = deepcopy(dep_idx)
            for i in dep_idx_temp:
                for dep in dependencies:
                    if i == dep[1] - 1:
                        # not root, not aspect
                        if (dep[2] - 1 < as_start or dep[2] - 1 >= as_end) and dep[2] != 0 and dep[2] - 1 not in dep_idx:
                            if str(dep[0]) != 'punct':  # and tokens[dep[2] - 1] not in stopWords
                                dep_tag.append('ncon_'+str(current_hop))
                                dep_dir.append(1)
                            else:
                                dep_tag.append('<pad>')
                                dep_dir.append(0)
                            dep_idx.append(dep[2] - 1)
                            added = True
                    elif i == dep[2] - 1:
                        # not root, not aspect
                        if (dep[1] - 1 < as_start or dep[1] - 1 >= as_end) and dep[1] != 0 and dep[1] - 1 not in dep_idx:
                            if str(dep[0]) != 'punct':  # and tokens[dep[1] - 1] not in stopWords
                                dep_tag.append('ncon_'+str(current_hop))
                                dep_dir.append(2)
                            else:
                                dep_tag.append('<pad>')
                                dep_dir.append(0)
                            dep_idx.append(dep[1] - 1)
                            added = True
            current_hop += 1

    if add_non_connect:
        for idx, token in enumerate(tokens):
            if idx not in dep_idx and (idx < as_start or idx >= as_end):
                dep_tag.append('non-connect')
                dep_dir.append(0)
                dep_idx.append(idx)

    # add aspect and index, to make sure length matches len(tokens)
    for idx, token in enumerate(tokens):
        if idx not in dep_idx:
            dep_tag.append('<pad>')
            dep_dir.append(0)
            dep_idx.append(idx)

    index = [i[0] for i in sorted(enumerate(dep_idx), key=lambda x:x[1])]
    dep_tag = [dep_tag[i] for i in index]
    dep_idx = [dep_idx[i] for i in index]
    dep_dir = [dep_dir[i] for i in index]

    assert len(tokens) == len(dep_idx), 'length wrong'
    return dep_tag, dep_idx, dep_dir

def _default_unk_index():
    return 1

def build_dep_tag_vocab(data, vocab_size=1000, min_freq=0):
    counter = Counter()
    for d in data:
        tags = d['dep_tag']
        counter.update(tags)

    itos = ['<pad>', '<unk>']
    min_freq = max(min_freq, 1)

    # sort by frequency, then alphabetically
    words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
    words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

    for word, freq in words_and_frequencies:
        if freq < min_freq or len(itos) == vocab_size:
            break
        if word == '<pad>':
            continue
        itos.append(word)
    # stoi is simply a reverse dict for itos
    stoi = defaultdict(_default_unk_index)
    stoi.update({tok: i for i, tok in enumerate(itos)})

    return {'itos': itos, 'stoi': stoi, 'len': len(itos)}

def convert_data(dataset):
    new_dataset=[]
    sid=0
    for i in dataset:
        record={}
        d=[]
        for pos in range(len(i['sentence'])):
            if pos < i['from']:
                d.append(i['to'] - pos)
            else:
                d.append(pos - i['from'])
        record['y']=i['sentiment']
        record['sent'] = ' '.join(i['sentence'])
        record['words'] = i['sentence']
        record['twords'] = i['aspect']
        record['dep_tag'] = i['dep_tag']
        record['deg_dir'] = i['dep_dir']
        record['wc'] = len(i['sentence'])
        record['wct'] = len(record['twords'])
        record['dist'] = d.copy()
        record['sid'] = sid
        record['beg'] = i['from']
        record['end'] = i['to']
        sid=sid+1
        new_dataset.append(record)
    return new_dataset
