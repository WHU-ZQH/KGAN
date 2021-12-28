import numpy as np
import spacy
from spacy import displacy
from model.rgat_file.senticnet5 import senticnet

from spacy.tokens import Doc

class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split()
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)

nlp = spacy.load('en_core_web_sm')
nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)


class Tree:
    def __init__(self, value, parent=None):
        if isinstance(value, list):
            self.value = 0
            self.parent = None
            self.children = []
            # 获取列表里每个路径
            for path in value:
                # 构建父结点和孩子结点
                parent = self
                for i,v in enumerate(path):
                    node = None
                    for child in parent.children:
                        if v == child.getValue():
                            node = child
                            break
                    if node == None:
                        node = Tree(v, parent)
                        parent.children.append(node)
                    parent = node
        else:
            # 该逻辑一般只由此构造器执行，而不由外部创建对象时直接执行
            self.value = value
            self.parent = parent
            self.children = []

    def getValue(self):
        """获取结点值"""
        return self.value

    def getChildren(self):
        """获取孩子结点"""
        return self.children

    def getParent(self):
        """获取父结点"""
        return self.parent

def get_senticnet_tree():
    values=[]
    for k,v in senticnet.items():
        value=[]
        value.append(k)
        for i,j in enumerate(v):
            if '#' in j:
                value.append(j.replace('#',''))
            if i>=8:
                value.append(j)
        values.append(value)
    get_senticnet_tree=Tree(value=values[:2],parent='a_little')

    print()



def dependency_adj_matrix(text):
    tokens = nlp(text)
    words = text.split()
    matrix = np.zeros((len(words), len(words))).astype('float32')
    assert len(words) == len(list(tokens))

    for token in tokens:
        matrix[token.i][token.i] = 1
        for child in token.children:
            matrix[token.i][child.i] = 1
            matrix[child.i][token.i] = 1
    return matrix

# def knowledge_adj_matrix2(text):
#     tokens = nlp(text)
#     words = text.split()
#     tags=[]
#     matrix=[[]]
#     for word in tokens:
#         matrix[word][word] = 1
#         semantic_tag=senticnet[word][:-5]
#         mood_tag=senticnet[word][4,5]
#         for i in semantic_tag:
#             matrix[word][i] = 1
#             matrix[i][word] = 1
#         for i in mood_tag:
#             i=i.replace('#','')
#             matrix[word][i] = 1
#             matrix[i][word] = 1
#
#     matrix = np.zeros((len(words), len(words))).astype('float32')
#     assert len(words) == len(list(tokens))
#     return matrix

def knowledge_adj_matrix(text):
    tokens = nlp(text)
    words = text.split()
    assert len(words) == len(list(tokens))
    tags={}
    num=len(words)
    for token in tokens:
        try:
            for sem in senticnet[token.text][8:]:
                if sem not in tags.keys():
                    tags[sem]=num
                    num+=1
            for mod in [senticnet[token.text][4].replace('#',''),senticnet[token.text][5].replace('#','')]:
                if mod not in tags.keys():
                    tags[mod] = num
                    num+=1
            tag=senticnet[token.text][8:]+[senticnet[token.text][4].replace('#',''),senticnet[token.text][5].replace('#','')]
        except KeyError:
            continue
    matrix = np.zeros((len(words)+len(tags.keys()), len(words)+len(tags.keys()))).astype('float32')
    for token in tokens:
        matrix[token.i][token.i] = 1
        for t in tag:
            matrix[token.i][tags[t]] = 1
            matrix[tags[t]][token.i] = 1
    return matrix

def process_graph(text):
    syntactic_adj_matrix = dependency_adj_matrix(text)
    # common_adj_matrix = knowledge_adj_matrix(text)
    return  syntactic_adj_matrix

# dependency_adj_matrix(('The apple is good'))
# # knowledge_adj_matrix('The apple is good')
# get_senticnet_tree()
