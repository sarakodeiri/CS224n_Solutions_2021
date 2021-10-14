# Q1.1
# ------------------
    flattened_list = [y for x in corpus for y in x]
    corpus_words = sorted(set(flattened_list))
    num_corpus_words = len(corpus_words)
# ------------------

# Q1.2
# ------------------
    M = np.zeros((num_words, num_words))
    
    for i in range(num_words):
        word2ind[words[i]] = i
    
    for file in corpus:
        for index in range(len(file)):
            x = word2ind[file[index]]
            for j in range(index - window_size, index + window_size + 1):
                if j >= 0 and j < len(file) and j != index:
                    y = word2ind[file[j]]
                    M[x, y] += 1
# ------------------

# Q1.3
# ------------------
    SVD = TruncatedSVD(k, n_iter = n_iters)
    M_reduced = SVD.fit_transform(M)
# ------------------

# Q1.4
# ------------------
    for word in words:
        x = M_reduced[word2ind[word],:][0]
        y = M_reduced[word2ind[word],:][1]
        plt.scatter(x, y, marker='x', color='blue')
        plt.text(x, y, word, fontsize=9)
# ------------------

# Q1.5-Explanation

#  As expected, Iraq, Ecuador and Kuwait are near each other as they are all countries.
#  Petroleum, Industry Energy and Oil are near eachother because they are often used next to each other when it comes to global news.
#  I expected Barrels and BPD to be closer to oil and industry. It might be because they aren't used that often (low frequency) or because they were used in other contexts as well.

# ------------------

# Q2.1-Explanation

#  Both the method and the corpus have changed. GloVe is pretrained and isn't fitted to the data whereas we trained SVD on the given corpus.

# ------------------

# Q2.2
# ------------------
    pprint.pprint(wv_from_bin.most_similar("light"))
    pprint.pprint(wv_from_bin.most_similar("run"))
    
    
    pprint.pprint(wv_from_bin.most_similar("bright"))
# ------------------

# Q2.2-Explanation

# "Light" is the antonym of both heavy and dark. Both meanings are seen in the top 10.
# "Run" means both starting and walking fast. Both are seen.
# On the other hand, "bright" means both light and smart. The only context that is shown in the top 10 is about the former meaning.
# It depends on the data and how the words are used. For example, in the corpus "light" has probably been used for describing weight almost as much as it's been used in the context of color; So we see both in the list. But this is not true for "bright".

# ------------------

# Q2.3
# ------------------
    pprint.pprint(wv_from_bin.distance("big", "small"))
    pprint.pprint(wv_from_bin.distance("naive", "small"))
# ------------------

# Q2.3-Explanation

#  The distance between big and small is much less than big naive and small, because they are more often used together, probably in comparison. And also because naive is a rarely used word on its own.

# ------------------

# Q2.4-Explanation

#  k - m + w

# ------------------

# Q2.5
# ------------------
pprint.pprint(wv_from_bin.most_similar(positive=['woman', 'actor'], negative=['man']))
# ------------------

# Q2.5-Explanation

#  actor:man :: actress:woman

# ------------------

# Q2.6
# ------------------
pprint.pprint(wv_from_bin.most_similar(positive=['dog', 'cat'], negative=['meow']))
# ------------------

# Q2.6-Explanation

#  cat:meow :: dog:dogs   correct value of b: bark

# ------------------

# Q2.7-Explanation

#  Women's work is often taking care of others (as nurses, teachers, or mothers) while men are seen in a more technical light (assuming from the words mechanic, laborer and factory)

# ------------------

# Q2.8
# ------------------
    concept = 'assault'
    one = 'man'
    two = 'woman'
    pprint.pprint(wv_from_bin.most_similar(positive=[one, concept], negative=[two]))
    pprint.pprint(wv_from_bin.most_similar(positive=[two, concept], negative=[one]))
# ------------------

# Q2.8-Explanation

#  When it comes to assault, according to the data, men are often seen as attackers and women as victims.

# ------------------

# Q2.9-Explanation

# Bias gets into vectors because of the data provided. No algorithm or method is biased by nature, it's just math. But when the data talks about a group in a different light, the bias takes place. In the first example, the data probably didn't include lots of data about women in the workplace except for in schools as teachers or in hospitals as nurses, and of course, mothers. Whereas it's easy to predict from the words "factory" and "mechanic" that men are seen in more technical work positions.
# A naive approach to test for bias is testing every instance of bias we can think of on our data (racial, sexist, ageist, etc.) But the best approach is making sure we make an unbiased dataset in the first place. Another great idea is the back-bone of the what-if you which tests hypothetical situations: https://pair-code.github.io/what-if-tool/

# ------------------