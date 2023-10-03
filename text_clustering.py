import pandas as pd
import MeCab
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib import rcParams

def mecab_sep(text):
	m = MeCab.Tagger("-Ochasen")
	node = m.parseToNode(text)
	words_list = []
	while node:
		if node.feature.split(",")[0] in ["名詞"]:
			words_list.append(node.surface)
		node = node.next
	return words_list

### Bag-of-words / tf-idf
def calc_vecs(docs):
	vectorizer = TfidfVectorizer(analyzer=mecab_sep)
	vecs = vectorizer.fit_transform(docs)
	return vecs.toarray()


target_docs_df = pd.read_csv("R1R2R3.csv",encoding='UTF-8')
target_docs = target_docs_df["題目"].tolist()
all_docs_vecs = calc_vecs(target_docs_df["題目"])

vec = KMeans(n_clusters=16)
vec.fit(all_docs_vecs)

vec.labels_

target_docs_df ["クラス"] = vec.labels_

pd.crosstab(target_docs_df["研究室"],target_docs_df["クラス"])

cross_survive= pd.crosstab(target_docs_df['クラス'], target_docs_df['研究室'], 
normalize='index')

#文字化け
plt.rcParams["font.family"] = "MS Gothic"

#plt.legend(loc = "best")

#カラー食
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", ["red", "blue", "green", 
"orange", "purple", "brown", "pink", "gray", "olive", "cyan", "black", "gold", "maroon", 
"lime", "fuchsia", "tan"])
n = 16
for i in range(n):
	x = (i, i+1)
	y = (i, i)
	plt.plot(x, y, "o-", label=i)

plt.legend( loc='center left', bbox_to_anchor=(1., .5))
#plt.legend()

cross_survive.plot.bar(stacked=True)
#plot_colortable(mcolors.CSS4_COLORS)
plt.legend( loc='center left', bbox_to_anchor=(1., .5))
#plt.xticks(rotation=45)