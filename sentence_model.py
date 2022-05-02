from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from clean_sentences import *
import numpy as np
from graficos import *


def get_resul_df(csimdf):
    resul_df = pd.DataFrame()
    item1_list = []
    item2_list = []
    score = []
    for i, j in csimdf.iterrows():
        for item1 in range(len(select_list)):
            for item2 in range(len(select_list)):
                item1_list.append(select_list[item1])
                item2_list.append(select_list[item2])
                score.append(csimdf[item1][item2])
    resul_df['ITE_ITEM_TITLE1'] = item1_list
    resul_df['ITE_ITEM_TITLE2'] = item2_list
    resul_df['Score Similitud (0,1)'] = score
    return resul_df


def calculate_cosine(sentence_embeddings):
    # calculate similarities (will store in array)
    score = np.zeros((sentence_embeddings.shape[0], sentence_embeddings.shape[0]))
    for i in range(sentence_embeddings.shape[0]):
        score[i, :] = cosine_similarity(
            [sentence_embeddings[i]],
            sentence_embeddings
        )[0]
    return score


if __name__ == "__main__":
    model = SentenceTransformer('neuralmind/bert-base-portuguese-cased')

    df_titles = pd.read_csv('items_titles.csv')

    df_titles['ITE_ITEM_TITLE'] = df_titles['ITE_ITEM_TITLE'].apply(remove_stopwords_br)
    df_titles['ITE_ITEM_TITLE'] = df_titles['ITE_ITEM_TITLE'].apply(clean)
    vocabulary = create_vocab(df_titles['ITE_ITEM_TITLE'])
    # print(dict(sorted(vocabulary.items(), key=lambda item: item[1], reverse=True)))

    select_list = df_titles['ITE_ITEM_TITLE'].tolist()
    select_list = select_list[0:5]
    embeddings = model.encode(select_list)
    doc_list = list(range(5))

    scores = calculate_cosine(embeddings)
    csim_df = pd.DataFrame(scores, index=sorted(doc_list), columns=sorted(doc_list))

    print(csim_df)

    plot_heatmap(scores, doc_list)

    final_df = get_resul_df(csim_df)
    print(final_df)
    final_df.to_csv('sentence_similarity.csv')
