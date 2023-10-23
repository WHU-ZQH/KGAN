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

nlp = spacy.load('en_core_web_lg')
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

    for num,token in enumerate(tokens):
        matrix[token.i][token.i] = 1
        if num!=0 and num!=(len(tokens)-1):
            for child in token.children:
                if child.string != '[CLS] ' and child.string != '[SEP]':
                    matrix[token.i][child.i] = 1
                    matrix[child.i][token.i] = 1
    return matrix


def knowledge_adj_matrix(text):
    tokens = nlp(text)
    words = text.split()
    assert len(words) == len(list(tokens))
    tags={}
    tag={}
    num=len(words)
    for token in tokens:
        try:
            for sem in senticnet[token.text][8:]:
                if sem not in tags.keys():
                    tags[sem]=num
                    num+=1
            tag[token]=senticnet[token.text][8:]
        except KeyError:
            continue
    matrix = np.zeros((len(words)+len(tags.keys()), len(words)+len(tags.keys()))).astype('float32')
    for token in tokens:
        matrix[token.i][token.i] = 1
        if token in tag.keys():
            for t in tag[token]:
                if t != '[CLS] ' and t != '[SEP]':
                    matrix[token.i][tags[t]] = 1
                    matrix[tags[t]][token.i] = 1
    return matrix, words+list(tags.keys())

def process_graph(text):
    syntactic_adj_matrix = dependency_adj_matrix(text)
    common_adj_matrix, words_know = knowledge_adj_matrix(text)
    return  syntactic_adj_matrix, common_adj_matrix, words_know

