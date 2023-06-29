from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
import pandas
import logging
import sys



logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'),
    stream=sys.stdout)
LOGGER = logging.getLogger(__name__)


def main():
    table = pandas.read_csv('modified_training.csv')
    clf = svm.SVC()
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        table[['qgram', 'cosine', 'smith_waterman', 'lcs', 'len_a', 'len_b']],
        table['category'], test_size=0.2, random_state=42)

    reg = LinearRegression().fit(X_train, y_train)
    clf.fit(X_train, y_train)
    for x_set, y_set in [(X_train, y_train), (X_holdout, y_holdout)]:
        y_pred = clf.predict(x_set)
        svm_accuracy = accuracy_score(y_set, y_pred)
        LOGGER.info(f"SVM Accuracy: {svm_accuracy}")
        y_pred = reg.predict(x_set)
        reg_accuracy = reg.score(x_set, y_pred)
        LOGGER.info(f"REG Accuracy: {reg_accuracy}")
        LOGGER.info('\n'.join(str(x) for x in sorted(zip(y_pred, y_set))))



if __name__ == '__main__':
    main()
