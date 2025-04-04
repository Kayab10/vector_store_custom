from store import VectorStore
import numpy as np

store = VectorStore()

sentences = [
    "World War I began in 1914 and ended in 1918.",
    "World War II started in 1939 and concluded in 1945.",
    "The assassination of Archduke Franz Ferdinand sparked World War I.",
    "The Treaty of Versailles officially ended World War I.",
    "Germany invaded Poland in 1939, marking the start of World War II.",
    "The Allies defeated the Axis powers in World War II.",
    "Atomic bombs were dropped on Hiroshima and Nagasaki in 1945.",
    "The League of Nations was formed after World War I.",
    "The United Nations was established following World War II.",
    "D-Day was a major Allied invasion during World War II in 1944."
]

# Tokenization and Vocabulary Creation
vocabulary = set()
for sentence in sentences:
    tokens = sentence.lower().split()
    vocabulary.update(tokens)

# Assign unique indices to words in the vocabulary
word_to_index = {word: i for i, word in enumerate(vocabulary)}

# Vectorization
sentence_vectors = {}
for sentence in sentences:
    tokens = sentence.lower().split()
    vector = np.zeros(len(vocabulary))
    for token in tokens:
        vector[word_to_index[token]] += 1
    sentence_vectors[sentence] = vector

# Storing in VectorStore
for sentence, vector in sentence_vectors.items():
    store.add_vector(sentence, vector)

# Searching for Similarity
query_sentence = "When did the Second World War begin and end?"
query_vector = np.zeros(len(vocabulary))
query_tokens = query_sentence.lower().split()
for token in query_tokens:
    if token in word_to_index:
        query_vector[word_to_index[token]] += 1

similar_sentences = store.find_similar_vectors(query_vector, num_results=2)

# Print similar sentences
print("Query Sentence:", query_sentence)
print("Similar Sentences:")
for sentence, similarity in similar_sentences:
    print(f"{sentence}: Similarity = {similarity:.4f}")