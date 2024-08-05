import nltk
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

def extract_keywords(text):
    # 分词
    words = word_tokenize(text)

    # 词性标注
    pos_tags = pos_tag(words)

    # 过滤停用词
    stop_words = set(stopwords.words('english'))
    pos_tags = [(word, pos) for word, pos in pos_tags if word.lower() not in stop_words]

    # 仅保留动词和名词
    # keywords = [word for word, pos in pos_tags if pos.startswith('N') or pos.startswith('V')]
    flag = False
    keywords = []
    for word,pos in pos_tags:
        if pos.startswith('N') or pos.startswith('V'):
            flag = False
            keywords.append(word)
        else:
            if flag == False:
                keywords.append("<pad>")
                flag = True

    return ' '.join(keywords)

def create_init_summ(filename,save_file):
    f = open(filename,"r")
    summs = f.readlines()
    f1 = open(save_file,"w")
    for summ in summs:
        init_summ = extract_keywords(summ)
        f1.write(init_summ+'\n')
    f.close()
    f1.close()

create_init_summ("../../data/java/test_output", "../../data/java/test1.init.summ")
create_init_summ("../../data/java/train_output", "../../data/java/train1.init.summ")
create_init_summ("../../data/java/dev_output", "../../data/java/valid1.init.summ")
