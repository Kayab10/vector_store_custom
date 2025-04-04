import numpy as np

class VectorStore:
    def __init__(self):
        self.vector_data = {} # dict to store vector
        self.vector_index = {} # indexing for retrieval
        
    def add_vector(self, vector_id, vector):
        """
        Add a vector to the store.

        Args:
            vector_id (str or int): A unique identifier for the vector.
            vector (numpy.ndarray): The vector data to be stored.
        """
        self.vector_data[vector_id] = vector
        self.update_index(vector_id, vector)
    
    def update_index(self,vector_id, vector):
        '''
        update the indexing structure for the vector store'''
        for existing_id, existing_vector in self.vector_data.items():
            #cosine similarity
            similarity = np.dot(vector,existing_vector)/(np.linalg.norm(vector)*np.linalg.norm(existing_vector))
            if existing_id not in self.vector_index:
                self.vector_index[existing_id] = {}
            self.vector_index[existing_id][vector_id] = similarity

    def get_vector(self, vector_id):
        """
        Retrieve a vector from the store.

        Args:
            vector_id (str or int): The identifier of the vector to retrieve.

        Returns:
            numpy.ndarray: The vector data if found, or None if not found.
        """
        return self.vector_data.get(vector_id) 

    def find_similar_vectors(self, query_vector, num_results=5):
        """
        Find similar vectors to the query vector using brute-force search.

        Args:
            query_vector (numpy.ndarray): The query vector for similarity search.
            num_results (int): The number of similar vectors to return.

        Returns:
            list: A list of (vector_id, similarity_score) tuples for the most similar vectors.
        """
        results = []
        for vector_id, vector in self.vector_data.items():
            similarity = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
            results.append((vector_id, similarity))

        # Sort by similarity in descending order
        results.sort(key=lambda x: x[1], reverse=True)

        # Return the top N results
        return results[:num_results]
    

