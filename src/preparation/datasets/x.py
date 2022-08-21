import os
import datetime
import numpy as np
import pandas as pd
from scipy import sparse
from config.constants import DATASET_PATH, MIN_INTERACTIONS_PER_USER

TIME_REGEX = "%Y-%m-%d %H:%M:%S.%f"


###############################################################################
# Set timestamps
###############################################################################
def format_unix_time(row):
    return datetime.datetime.strptime(row['ImputedEventTime'], TIME_REGEX).timestamp()


def format_course_id(row):
    return row['CourseCode'][:6]

###############################################################################
# Process question meta-data
###############################################################################


def hash_q_mat(Q_mat):
    item_dic = {}
    indices_dic = {}
    for row_idx, row in enumerate(Q_mat):
        indices = tuple(row.indices.tolist())
        index = indices_dic.setdefault(indices, len(indices_dic))
        item_dic[row_idx] = index
    return item_dic


def prepare_question_meta_data(dataset):
    question_df = pd.read_csv(os.path.join(DATASET_PATH[dataset],
                                           "CS-Questions.csv"))

    question_to_id = {}
    vocabulary = {}
    indptr = [0]
    indices = []
    data = []

    for question_id, kcs in question_df[["QuestionId", "KC"]].values:
        for kc in kcs.split(';'):
            index = vocabulary.setdefault(kc, len(vocabulary))
            indices.append(index)
            data.append(1)
        indptr.append(len(indices))
        question_to_id.setdefault(question_id, len(question_to_id))

    Q_mat = sparse.csr_matrix((data, indices, indptr))
    return question_to_id, Q_mat


###############################################################################
# Prepare dataset
###############################################################################

def prepare_x(n_splits, dataset):

    print("Prepare question meta-data")
    print("-----------------------------------------------")

    q_to_id, Q_mat = prepare_question_meta_data(dataset)

    print("\nPreparing interaction data")
    print("-----------------------------------------------")
    base = DATASET_PATH[dataset]
    df = pd.read_csv(base + "CS-Question-Results6.csv")

    # Rename standard columns
    df.rename(columns={
        "UserId": "user_id",
        "QuizName": "bundle_id",
        "IsCorrect": "correct"
    }, inplace=True)

    # timestamps
    df['ImputedEventTime'] = df['EventTime']
    df['ImputedEventTime'].fillna(df['TimeCompleted'], inplace=True)
    df['ImputedEventTime'].fillna(df['AttemptCompleted'], inplace=True)
    df['ImputedEventTime'].fillna(df['AttemptStarted'], inplace=True)
    df['unix_time'] = df.apply(format_unix_time, axis=1)
    df['timestamp'] = df['unix_time'] - np.min(df['unix_time'])

    # Use first 5 digits of CourseId as course_id to group different presentations of course together
    df['course_id'] = df.apply(format_course_id, axis=1)

    # Some of the feature extraction modules check for nan in the pre-processed data so
    # remove some of the time columns that contain nan that we have now replaced with
    # the imputed timestamps
    df['Score'].fillna(0)
    df.dropna(axis=1, inplace=True)

    # item_id
    df["item_id"] = \
        np.array([q_to_id[q] for q in df["QuestionId"]])

    print("Checking consistency...")
    assert df.isnull().sum().sum() == 0, \
        "Error, found NaN values in the preprocessed data."

    print(df.head())

    print("Storing interaction data")
    print("------------------------------------------------------------------")

    # remove users with too few interactions
    def f(x): return len(x) >= MIN_INTERACTIONS_PER_USER
    df = df.groupby("user_id").filter(f)
    df = df.sort_values(["user_id", "timestamp"])
    df.reset_index(inplace=True, drop=True)

    # create splits for cross validation
    from src.preparation.prepare_data import determine_splits
    determine_splits(df, dataset, n_splits=n_splits)

    # save results
    pp = os.path.join(DATASET_PATH[dataset], "preparation")

    # Save the Q-matrix and hash the skill ids
    sparse.save_npz(os.path.join(pp, "q_mat.npz"), Q_mat)

    hashed_q = hash_q_mat(Q_mat)
    skill_id = [hashed_q[item_id] for item_id in df.item_id]
    pp = os.path.join(pp, "preprocessed_data.csv")
    df = df.assign(hashed_skill_id=skill_id)
    df.to_csv(pp, sep="\t", index=False)

