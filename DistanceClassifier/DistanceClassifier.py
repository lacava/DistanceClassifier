# -*- coding: utf-8 -*-

"""
Permission is hereby granted, free of charge, to any person obtaining a copy of this software
and associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from __future__ import print_function
from sklearn.base import BaseEstimator
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from ._version import __version__

class DistanceClassifier(BaseEstimator):

    """Multifactor Dimensionality Reduction (DistanceClassifier) for feature construction in machine learning"""

    def __init__(self, d='mahalanobis'):
        """Sets up the DistanceClassifier algorithm

        Parameters
        ----------
        d: ('mahalanobis' or 'euclidean')
            Type of distance calculation to use

        Returns
        -------
        None

        """
        # Save params to be recalled later by get_params()
        self.params = locals()  # Must be placed before any local variable definitions
        self.params.pop('self')

        self.d = d


    def fit(self, features, classes):
        """Constructs the DistanceClassifier from the provided training data

        Parameters
        ----------
        features: array-like {n_samples, n_features}
            Feature matrix
        classes: array-like {n_samples}
            List of class labels for prediction

        Returns
        -------
        None

        """
        # group the data by class label
        X = np.empty(np.unique(classes)) #,features.shape[0],features.shape[1]])
        self.mu = np.empty(np.unique(classes))
        self.Z = np.empty(np.unique(classes))
        for i in classes:
            X[i] = features[classes == i]
            self.mu[i] = np.mean(X[i])
            if self.d == 'mahalanobis':
                self.Z[i] = np.cov(X[i])


    def predict(self, features):
        """Predict class outputs for an unlabelled feature set"""

        for i in np.arange(self.mu.shape[0]):
            if self.d == 'mahalanobis':
                distance[i] = (features - mu[i])*np.linalg.inv(Z[i])*(features - mu[i]).transpose()
            else:
                distance[i] = (features - mu[i])**2

    def fit_predict(self, features, classes):
        """Convenience function that fits the provided data then predicts the class labels
        Parameters
        ----------
        features: array-like {n_samples, n_features}
            Feature matrix
        classes: array-like {n_samples}
            List of true class labels

        Returns
        ----------
        array-like: {n_samples}
            Constructed features from the provided feature matrix

        """
        self.fit(features, classes)
        return self.predict(features)

    def score(self, features, classes, scoring_function=None, **scoring_function_kwargs):
        """Estimates the accuracy of the predictions from the constructed feature

        Parameters
        ----------
        features: array-like {n_samples, n_features}
            Feature matrix to predict from
        classes: array-like {n_samples}
            List of true class labels

        Returns
        -------
        accuracy_score: float
            The estimated accuracy based on the constructed feature

        """
        if len(self.feature_map) == 0:
            raise ValueError('The DistanceClassifier model must be fit before score() can be called')

        new_feature = self.transform(features)

        if scoring_function is None:
            results = (new_feature == classes)
            score = np.sum(results)
            return float(score) / classes.size
        else:
            return scoring_function(classes, new_feature, **scoring_function_kwargs)

    def get_params(self, deep=None):
        """Get parameters for this estimator

        This function is necessary for DistanceClassifier to work as a drop-in feature constructor in,
        e.g., sklearn.cross_validation.cross_val_score

        Parameters
        ----------
        deep: unused
            Only implemented to maintain interface for sklearn

        Returns
        -------
        params: mapping of string to any
            Parameter names mapped to their values
        """
        return self.params

def main():
    """Main function that is called when DistanceClassifier is run on the command line"""
    parser = argparse.ArgumentParser(description='DistanceClassifier for classification based on distance measure in feature space.',
                                     add_help=False)

    parser.add_argument('INPUT_FILE', type=str, help='Data file to perform DistanceClassifier on; ensure that the class label column is labeled as "class".')

    parser.add_argument('-h', '--help', action='help', help='Show this help message and exit.')

    parser.add_argument('-is', action='store', dest='INPUT_SEPARATOR', default='\t',
                        type=str, help='Character used to separate columns in the input file.')

    parser.add_argument('-d', action='store', dest='D', default='mahalanobis',choices = ['mahalanobis','euclidean'],
                        type=str, help='Distance metric to use.')

    parser.add_argument('-v', action='store', dest='VERBOSITY', default=1, choices=[0, 1, 2],
                        type=int, help='How much information DistanceClassifier communicates while it is running: 0 = none, 1 = minimal, 2 = all.')

    parser.add_argument('--version', action='version', version='DistanceClassifier {version}'.format(version=__version__),
                        help='Show DistanceClassifier\'s version number and exit.')

    args = parser.parse_args()

    if args.VERBOSITY >= 2:
        print('\nDistanceClassifier settings:')
        for arg in sorted(args.__dict__):
            print('{}\t=\t{}'.format(arg, args.__dict__[arg]))
        print('')

    input_data = pd.read_csv(args.INPUT_FILE, sep=args.INPUT_SEPARATOR)

    if 'Class' in input_data.columns.values:
        input_data.rename(columns={'Class': 'class'}, inplace=True)

    RANDOM_STATE = args.RANDOM_STATE if args.RANDOM_STATE > 0 else None

    training_indices, testing_indices = train_test_split(input_data.index,
                                                         stratify=input_data['class'].values,
                                                         train_size=0.75,
                                                         test_size=0.25,
                                                         random_state=RANDOM_STATE)

    training_features = input_data.loc[training_indices].drop('class', axis=1).values
    training_classes = input_data.loc[training_indices, 'class'].values

    testing_features = input_data.loc[testing_indices].drop('class', axis=1).values
    testing_classes = input_data.loc[testing_indices, 'class'].values

    # Run and evaluate DistanceClassifier on the training and testing data
    dc = DistanceClassifier()
    dc.fit(training_features, training_classes)

    if args.VERBOSITY >= 1:
        print('\nTraining accuracy: {}'.format(dc.score(training_features, training_labels)))
        print('Holdout accuracy: {}'.format(dc.score(testing_features, testing_labels)))



if __name__ == '__main__':
    main()
