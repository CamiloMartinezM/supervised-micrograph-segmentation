# original_shape = X[-2::].astype(int)
# X_2d = X[:-2].reshape(original_shape)

# # The image is segmented using the given algorithm for superpixels generation
# # using the input X as a 2d array
# segments = cp.asarray(self._generate_superpixels(X_2d))
# superpixels = cp.unique(segments)
# n_superpixels = int(superpixels[-1]) + 1

# # Response vector for X as a 2d numpy array, with shape (*X_2d.shape, 8).
# # This response vector is flattened so that a numpy array of shape
# # (X_2d.size, 8) is obtained
# responses = self._get_response_vector(X_2d).reshape(-1, 8)

# # Numpy array of shape (n_superpixels, X.size) where each position has a boolean
# # mask that represents where that superpixel appears (position == superpixel)
# # Remember that X's last two values are actually its shape and are not actually
# # part of the image.
# flat_X = cp.asarray(X[:-2])
# superpixel_booleanmasks = (flat_X[:, np.newaxis] == superpixels)

# # Using the boolean masks, it is possible to extract the corresponding responses 
# # of every pixel in every superpixel (a pixel of a superpixel would be a 'True'
# # value in the boolean mask)
# superpixels_responses = [
#     responses[superpixel_booleanmasks[:, i], :] 
#     for i in range(n_superpixels)
# ]

# # Size of every superpixel response. This corresponds to the number of pixels each
# # superpixel has
# lens = cp.array([
#     superpixels_responses[i].shape[0] # cp.sum(superpixel_booleanmasks[:, i]) 
#     for i in range(n_superpixels)
# ])

# # Concatenation of superpixels_responses in order to obtain a big matrix of shape
# # (n_superpixels, max(lens), 8). This allows to store the feature vectors of every
# # superpixel, eventhough they do not have the same dimension in the first axis
# # (because they all have a different number of pixels). The remaining values are 
# # filled with np.nan
# mask = lens[:, np.newaxis] > cp.arange(cp.max(lens))
# feature_vectors = cp.full((*mask.shape, 8), np.nan)
# feature_vectors[mask] = cp.concatenate(superpixels_responses)

# # T is broadcasted in order to account for all superpixels
# big_T = cp.broadcast_to(self.T, (n_superpixels, *self.T.shape))

# # Distance matrices for every superpixel, with shape 
# # (n_superpixels, self.classes, max(lens), K)
# distance_matrices = cp.linalg.norm(
#     feature_vectors[:, np.newaxis, :, np.newaxis] - big_T[:, :, np.newaxis, :], 
#     axis=-1
# )
# # Minimum distance vectors for every superpixel, with shape (n_superpixels, max(lens))
# minimum_distance_vectors = cp.min(distance_matrices, axis=(-1, 1))

# # Matrix which correlates texture texton distances and minimum distances of every
# # pixel, for every superpixel
# A = np.sum(
#     cp.isclose(
#         minimum_distance_vectors[:, np.newaxis, :, np.newaxis], distance_matrices, 
#         rtol=1e-16
#     ),
#     axis=(-1, 2)
# )
# # The new segments are created, i.e, actual segmentation.
# # S_segmented = {}
# # class_matrix = cp.zeros(X.shape, dtype=int)
# # for superpixel in cp.unique(segments):
# #     pixels = cp.argwhere(segments == superpixel)
# #     i = pixels[:, 0]
# #     j = pixels[:, 1]

# #     feature_vectors = responses[i, j]
# #     predicted_class_idx = self._class_of(feature_vectors)
# #     S_segmented[int(superpixel)] = self.classes[predicted_class_idx.get()]
# #     class_matrix[i, j] = int(predicted_class_idx)

# superpixel_classes = cp.argmax(A, axis=1)
# class_matrix = cp.zeros(X.shape[0], dtype=int)
# for i in range(superpixel_classes.shape[0]):
#     class_matrix[:-2][superpixel_booleanmasks[:, i]] = superpixel_classes[i]

# class_matrix[-2::] = cp.asarray(original_shape)
# return class_matrix.get()