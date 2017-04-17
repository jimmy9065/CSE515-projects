import numpy as np
import tensorflow as tf


class K_means:
    maxiter = 100
    dim = 1
    n_clusters = 1
    centers = []

    def __init__(self, n_clusters, maxiter):
        self.maxiter = maxiter
        self.n_clusters = n_clusters

    def fit(self, X):
        [best_centroids, min_loss] = self.cluster(X)
        for i in range(10):
            [centroids, loss] = self.cluster(X)
            if loss < min_loss:
                best_centroids = centroids
                min_loss = loss
        self.centers = best_centroids

    def cluster(self, X):
        X_size = X.shape[0]
        dim = X.shape[1]
        points = tf.Variable(X)

        # initial the centroids
        centroids = tf.Variable(tf.slice(tf.random_shuffle(
                                points.initialized_value()),
                                [0, 0],
                                [self.n_clusters, dim]))
        assignments = tf.Variable(tf.zeros([X_size], dtype=tf.int64))

        # expand the points and centoids so we can calculate dists
        points_expand = tf.expand_dims(points, 0)
        centroids_expanded = tf.expand_dims(centroids, 1)
        dists = tf.reduce_sum(
                tf.square(tf.subtract(points_expand, centroids_expanded)), 2)

        sum_dists = tf.reduce_sum(tf.reduce_mean(dists, 1))

        # update assignment
        new_assignments = tf.argmin(dists, 0)

        # see if anything changed
        change_assignments = tf.reduce_any(tf.not_equal(new_assignments,
                                                        assignments))

        # calculate the ave dists
        def bucket_mean(data, bucket_ids, num_buckets):
            total = tf.unsorted_segment_sum(data, bucket_ids, num_buckets)
            count = tf.unsorted_segment_sum(tf.ones_like(data),
                                            bucket_ids,
                                            num_buckets) + 1e-10
            return total / count

        means = bucket_mean(points, new_assignments, self.n_clusters)

        with tf.control_dependencies([change_assignments]):
            centroids_update = tf.group(
                    centroids.assign(means),
                    assignments.assign(new_assignments))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            changed = True
            iters = int(0)
            while changed and (iters < self.maxiter):
                [changed, _] = sess.run(
                        [change_assignments, centroids_update])
                iters += 1

            [centers, loss] = sess.run([centroids, sum_dists])

            # if not changed:
            #     print('stop earlier', loss)

        return centers, loss

    def get_centers(self):
        return self.centers


if __name__ == '__main__':
    data = np.load('./train.npz')
    X = data[data.files[0]]
    X = X[:5000, :]
    print(X.shape)
    kmeans = K_means(n_clusters=128, maxiter=1000)
    kmeans.fit(X)
    centers = kmeans.centers
    print('centers:', centers)
