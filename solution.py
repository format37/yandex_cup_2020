import sys
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
import numpy as np


def read_train_data(target_count, class_count, budget):
    print('req 0 0 {}'.format(budget))  # requesting all possible data
    sys.stdout.flush()
    train_count = int(input())
    train_data_targets = []
    train_data_categories = []
    train_data_vectors = []
    for _ in range(train_count):
        input_line = input().strip().split()
        train_data_targets.append(list(map(int, input_line[:target_count])))
        train_data_categories.append(list(map(int, input_line[target_count:target_count + class_count])))
        train_data_vectors.append(list(map(float, input_line[target_count + class_count:])))
    return train_data_targets, train_data_categories, train_data_vectors


def train_model(train_data_targets,
                train_data_categories,
                train_data_vectors,
                target_count,
                class_count):
    # Convert to df
    df_targets = pd.DataFrame(train_data_targets)
    df_categories = pd.DataFrame(train_data_categories)
    df_vectors = pd.DataFrame(train_data_vectors)
    # Merge targets
    y = sum([df_targets[i] * (i + 1) for i in range(0, target_count)])
    # Merge categories and vectors
    df_categories.columns = [int(i) for i in df_categories.columns]
    df_vectors.columns = [i for i in
                          range(len(df_categories.columns),
                                len(df_vectors.columns) + len(df_categories.columns)
                                )]
    for col in df_vectors.columns:
        df_categories[col] = df_vectors[col]
    # Categorical features declaration
    cat_features = [i for i in range(class_count)]
    # Split data into train and validation
    X_train, X_validation, y_train, y_validation = train_test_split(
        df_categories,
        y,
        train_size=0.8,
        random_state=1234
    )
    # TODO: play with optimization
    model = CatBoostClassifier(
        iterations=5,
        learning_rate=0.5,
        loss_function='MultiClass'
    )
    model.fit(
        X_train, y_train,
        cat_features=cat_features,
        eval_set=(X_validation, y_validation),
        verbose=False
    )
    return model


def get_predicted_indexes(test_count, target_count, model):
    input_lines = []

    for i in range(test_count):
        input_line = input().strip().split()
        input_lines.append(input_line)

    user_requests = pd.DataFrame(input_lines)

    prediction = pd.DataFrame(model.predict_proba(data=user_requests))
    prediction['i'] = prediction.index
    mask = ((prediction[1] > prediction[0]) | (prediction[2] > prediction[0]) | (prediction[3] > prediction[0]))
    total_good_items = [str(len(prediction[prediction[i + 1] > prediction[0]])) for i in range(target_count)]
    return list(prediction[mask]['i']), total_good_items


def main():
    target_count, class_count, enbedding_dim = map(int, input().strip().split())  # T, C, F
    class_counts = list(map(int, input().strip().split()))  # C_i
    budget = int(input())  # B

    train_data_targets, train_data_categories, train_data_vectors = read_train_data(
        target_count,
        class_count,
        budget
    )

    model = train_model(train_data_targets,
                        train_data_categories,
                        train_data_vectors,
                        target_count,
                        class_count)

    print('test')
    sys.stdout.flush()

    test_count = int(input())

    all_good_indexes, total_good_items = get_predicted_indexes(test_count, target_count, model)

    print(' '.join(total_good_items))

    for good_index in all_good_indexes:
        print(good_index)

    sys.stdout.flush()


if __name__ == '__main__':
    main()
