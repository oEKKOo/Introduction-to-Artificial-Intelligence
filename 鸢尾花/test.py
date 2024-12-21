# 导入必要的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.decomposition import PCA
from math import pi

# 设置Seaborn风格
sns.set(style="whitegrid")

# 加载鸢尾花数据集
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# 转换为DataFrame
df = pd.DataFrame(X, columns=feature_names)
df['species'] = y
df['species_name'] = df['species'].apply(lambda i: target_names[i])

# 查看前五行
print("数据集前五行：")
print(df.head())

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means 聚类
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)

# 计算轮廓系数
kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)
print(f"K-Means Silhouette Score: {kmeans_silhouette:.3f}")

# 层次聚类
hierarchical = AgglomerativeClustering(n_clusters=3, linkage='ward')
hierarchical_labels = hierarchical.fit_predict(X_scaled)

# 计算轮廓系数
hierarchical_silhouette = silhouette_score(X_scaled, hierarchical_labels)
print(f"Hierarchical Clustering Silhouette Score: {hierarchical_silhouette:.3f}")

# PCA降维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 聚类结果可视化
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=kmeans_labels, palette='viridis')
plt.title('K-Means Clustering')

plt.subplot(1, 2, 2)
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=hierarchical_labels, palette='viridis')
plt.title('Hierarchical Clustering')

plt.tight_layout()
plt.show()

# 聚类性能比较
ari_kmeans = adjusted_rand_score(y, kmeans_labels)
ari_hierarchical = adjusted_rand_score(y, hierarchical_labels)

print(f"K-Means Adjusted Rand Index: {ari_kmeans:.3f}")
print(f"Hierarchical Clustering Adjusted Rand Index: {ari_hierarchical:.3f}")

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 逻辑回归分类
log_reg = LogisticRegression(random_state=42, max_iter=200)
log_reg.fit(X_train_scaled, y_train)
y_pred_lr = log_reg.predict(X_test_scaled)

# 评估
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f"Logistic Regression Accuracy: {accuracy_lr:.3f}")
print("Classification Report (Logistic Regression):\n", classification_report(y_test, y_pred_lr, target_names=target_names))

# 决策树分类
dt_clf = DecisionTreeClassifier(random_state=42)
dt_clf.fit(X_train, y_train)
y_pred_dt = dt_clf.predict(X_test)

# 评估
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print(f"Decision Tree Accuracy: {accuracy_dt:.3f}")
print("Classification Report (Decision Tree):\n", classification_report(y_test, y_pred_dt, target_names=target_names))

# 混淆矩阵可视化
plt.figure(figsize=(12, 5))

# Logistic Regression 混淆矩阵
plt.subplot(1, 2, 1)
cm_lr = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.title('Logistic Regression Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')

# Decision Tree 混淆矩阵
plt.subplot(1, 2, 2)
cm_dt = confusion_matrix(y_test, y_pred_dt)
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Greens', xticklabels=target_names, yticklabels=target_names)
plt.title('Decision Tree Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')

plt.tight_layout()
plt.show()

# 参数选择与性能分析

# K-Means 簇数量分析
silhouette_scores = []
k_values = range(2, 7)
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)

plt.figure(figsize=(8,6))
plt.plot(k_values, silhouette_scores, marker='o')
plt.title('K-Means Silhouette Scores for different k')
plt.xlabel('Number of clusters k')
plt.ylabel('Silhouette Score')
plt.xticks(k_values)
plt.grid(True)
plt.show()

# 逻辑回归参数调优
param_grid_lr = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'saga']
}

grid_lr = GridSearchCV(LogisticRegression(max_iter=200, random_state=42), param_grid_lr, cv=5, scoring='accuracy')
grid_lr.fit(X_train_scaled, y_train)

print(f"Best Logistic Regression Params: {grid_lr.best_params_}")
print(f"Best CV Accuracy: {grid_lr.best_score_:.3f}")

# 使用最佳参数进行预测
best_lr = grid_lr.best_estimator_
y_pred_best_lr = best_lr.predict(X_test_scaled)
accuracy_best_lr = accuracy_score(y_test, y_pred_best_lr)
print(f"Best Logistic Regression Test Accuracy: {accuracy_best_lr:.3f}")

# 决策树参数调优
param_grid_dt = {
    'max_depth': [None, 2, 3, 4, 5],
    'min_samples_split': [2, 5, 10]
}

grid_dt = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid_dt, cv=5, scoring='accuracy')
grid_dt.fit(X_train, y_train)

print(f"Best Decision Tree Params: {grid_dt.best_params_}")
print(f"Best CV Accuracy: {grid_dt.best_score_:.3f}")

# 使用最佳参数进行预测
best_dt = grid_dt.best_estimator_
y_pred_best_dt = best_dt.predict(X_test)
accuracy_best_dt = accuracy_score(y_test, y_pred_best_dt)
print(f"Best Decision Tree Test Accuracy: {accuracy_best_dt:.3f}")

# 分类性能比较
models = {
    'Logistic Regression': accuracy_lr,
    'Decision Tree': accuracy_dt,
    'Best Logistic Regression': accuracy_best_lr,
    'Best Decision Tree': accuracy_best_dt
}

plt.figure(figsize=(8,6))
sns.barplot(x=list(models.keys()), y=list(models.values()), palette='viridis')
plt.ylim(0,1)
plt.ylabel('Accuracy')
plt.title('Classification Model Accuracy Comparison')
plt.xticks(rotation=45)
for index, value in enumerate(models.values()):
    plt.text(index, value + 0.01, f"{value:.2f}", ha='center')
plt.show()

# 额外可视化：箱线图、直方图和径向图

# 1. 箱线图
plt.figure(figsize=(12, 8))
for idx, feature in enumerate(feature_names):
    plt.subplot(2, 2, idx+1)
    sns.boxplot(x='species_name', y=feature, data=df)
    plt.title(f'Box Plot of {feature}')
plt.tight_layout()
plt.show()

# 2. 直方图
plt.figure(figsize=(12, 8))
for idx, feature in enumerate(feature_names):
    plt.subplot(2, 2, idx+1)
    sns.histplot(data=df, x=feature, hue='species_name', kde=True, bins=20, palette='Set2', element='step')
    plt.title(f'Histogram of {feature}')
plt.tight_layout()
plt.show()

# 3. 径向图（雷达图）
# 计算每个类别的特征平均值，仅选择特征列
species_means = df.groupby('species_name')[feature_names].mean()

# 设置雷达图
categories = feature_names
N = len(categories)

angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]  # 完成圆周

plt.figure(figsize=(8, 8))

for idx, species in enumerate(species_means.index):
    values = species_means.loc[species].values.flatten().tolist()
    values += values[:1]  # 完成圆周
    plt.subplot(111, polar=True)
    plt.xticks(angles[:-1], [cat.capitalize() for cat in categories])
    plt.plot(angles, values, linewidth=1, linestyle='solid', label=species)
    plt.fill(angles, values, alpha=0.1)

plt.title('Radar Chart of Iris Species Features', size=20, y=1.05)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.show()
