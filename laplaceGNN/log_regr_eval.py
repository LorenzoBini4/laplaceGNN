import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, ShuffleSplit, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import OneHotEncoder, normalize
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score

def log_regr(X, y, data_random_seed=1, repeat=1):
    # Initialize the OneHotEncoder
    one_hot_encoder = OneHotEncoder(categories='auto', sparse=False)
    y = one_hot_encoder.fit_transform(y.reshape(-1, 1)).astype(bool)

    # Normalize the input features
    X = normalize(X, norm='l2')
    # Set random state for reproducibility
    rng = np.random.RandomState(data_random_seed)  # This ensures the dataset will be split exactly the same throughout training

    best_accuracy = 0.0
    accuracies = []
    test_size = 0.80  # Test set size, standars evaluation protocol

    for _ in range(repeat):
        # Perform a random split after each repeat
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=rng)
        print(f"Test set size: {test_size}")

        logreg = LogisticRegression(solver='liblinear')
        c = 2.0 ** np.arange(-10, 11)  # It's the range of hyperparameters to be searched
        cv = ShuffleSplit(n_splits=5, test_size=0.5)
        clf = GridSearchCV(estimator=OneVsRestClassifier(logreg), param_grid=dict(estimator__C=c),
                           n_jobs=5, cv=cv, verbose=0)
        clf.fit(X_train, y_train)
        y_pred = clf.predict_proba(X_test)
        y_pred = np.argmax(y_pred, axis=1)
        y_pred = one_hot_encoder.transform(y_pred.reshape(-1, 1)).astype(bool)
        test_acc = metrics.accuracy_score(y_test, y_pred)
        accuracies.append(test_acc)
        best_accuracy = max(best_accuracy, test_acc)
        print(f"Best Accuracy: {best_accuracy}")

    return accuracies

def log_regr_preset_splits(X, y, train_masks, val_masks, test_mask):
    # transfrom targets to one-hot vector
    one_hot_encoder = OneHotEncoder(categories='auto', sparse=False)
    # y = one_hot_encoder.fit_transform(y.reshape(-1, 1)).astybool(bool)    
    y = one_hot_encoder.fit_transform(y.reshape(-1, 1)).astype(bool) # For WikiCS dataset only
    X = normalize(X, norm='l2')
    accuracies = []
    for split_id in range(train_masks.shape[1]):
        train_mask, val_mask = train_masks[:, split_id], val_masks[:, split_id]
        total_size = len(train_mask) + len(val_mask) + len(test_mask)
        train_percentage = (len(train_mask) / total_size) * 100
        val_percentage = (len(val_mask) / total_size) * 100
        test_percentage = (len(test_mask) / total_size) * 100
        print(f"Train mask: {train_percentage:.2f}%, Val mask: {val_percentage:.2f}%, Test mask: {test_percentage:.2f}%")

        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        best_test_acc, best_acc = 0, 0
        for c in 2.0 ** np.arange(-10, 11):
            clf = OneVsRestClassifier(LogisticRegression(solver='liblinear', C=c))
            clf.fit(X_train, y_train)

            y_pred = clf.predict_proba(X_val)
            y_pred = np.argmax(y_pred, axis=1)
            y_pred = one_hot_encoder.transform(y_pred.reshape(-1, 1)).astype(bool)
            val_acc = metrics.accuracy_score(y_val, y_pred)
            if val_acc > best_acc:
                best_acc = val_acc
                y_pred = clf.predict_proba(X_test)
                y_pred = np.argmax(y_pred, axis=1)
                y_pred = one_hot_encoder.transform(y_pred.reshape(-1, 1)).astype(bool)
                best_test_acc = metrics.accuracy_score(y_test, y_pred)

        accuracies.append(best_test_acc)
    print(np.mean(accuracies))
    return accuracies

def log_regr_ogbn_arxiv_liblinear(X, y, train_idx, val_idx, test_idx):
    X = normalize(X, norm='l2')
    # c_values = 2.0 ** np.arange(-10, 11)
    c_values = 2.0 ** np.arange(-7, 7)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    best_test_acc, best_acc = 0, 0
    for c in c_values:
        clf = OneVsRestClassifier(LogisticRegression(solver='liblinear', C=c))
        clf.fit(X_train, y_train)

        y_pred_val = clf.predict(X_val)
        val_acc = metrics.accuracy_score(y_val, y_pred_val)
        if val_acc > best_acc:
            best_acc = val_acc
            y_pred_test = clf.predict(X_test)
            best_test_acc = metrics.accuracy_score(y_test, y_pred_test)

    print(f"Best validation accuracy: {best_acc}")
    print(f"Test accuracy with best validation model: {best_test_acc}")
    return best_test_acc

def log_regr_ogbn_paper100M_liblinear(X, y, train_idx, val_idx, test_idx):
    X = normalize(X, norm='l2')
    one_hot_encoder = OneHotEncoder(categories='auto', sparse=False)
    y = one_hot_encoder.fit_transform(y.reshape(-1, 1)).astype(bool)
    c_values = 2.0 ** np.arange(-7, 7)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    best_test_acc, best_acc = 0, 0
    for c in c_values:
        clf = OneVsRestClassifier(LogisticRegression(solver='liblinear', C=c))
        clf.fit(X_train, y_train)

        y_pred_val = clf.predict_proba(X_val)
        y_pred_val = np.argmax(y_pred_val, axis=1)
        y_pred_val = one_hot_encoder.transform(y_pred_val.reshape(-1, 1)).astype(bool)
        val_acc = metrics.accuracy_score(y_val, y_pred_val)
        if val_acc > best_acc:
            best_acc = val_acc
            y_pred_test = clf.predict_proba(X_test)
            y_pred_test = np.argmax(y_pred_test, axis=1)
            y_pred_test = one_hot_encoder.transform(y_pred_test.reshape(-1, 1)).astype(bool)
            best_test_acc = metrics.accuracy_score(y_test, y_pred_test)

    print(f"Best validation accuracy: {best_acc}")
    print(f"Test accuracy with best validation model: {best_test_acc}")
    return best_test_acc

def log_regr_ogbn_arxiv_adam(X, y, train_idx, val_idx, test_idx):
    X = normalize(X, norm='l2')
    
    X_train, y_train = torch.tensor(X[train_idx], dtype=torch.float32), torch.tensor(y[train_idx], dtype=torch.long)
    X_val, y_val = torch.tensor(X[val_idx], dtype=torch.float32), torch.tensor(y[val_idx], dtype=torch.long)
    X_test, y_test = torch.tensor(X[test_idx], dtype=torch.float32), torch.tensor(y[test_idx], dtype=torch.long)
    
    weight_decay_values = 2.0 ** np.arange(-4, 4)
    learning_rate = 0.007
    num_epochs = 250
    best_test_acc, best_val_acc = 0, 0
    
    # Iterate over weight decay values
    for wd in weight_decay_values:
        model = torch.nn.Sequential(
            torch.nn.Linear(X_train.shape[1], 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, len(torch.unique(y_train))),
            torch.nn.Softmax(dim=1)
        )
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=wd)
        criterion = torch.nn.CrossEntropyLoss()
        
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_preds = torch.argmax(val_outputs, dim=1)
            val_acc = accuracy_score(y_val, val_preds)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_outputs = model(X_test)
                test_preds = torch.argmax(test_outputs, dim=1)
                best_test_acc = accuracy_score(y_test, test_preds)
    
    print(f"Best validation accuracy: {best_val_acc}")
    print(f"Test accuracy with best validation model: {best_test_acc}")
    
    return best_test_acc
