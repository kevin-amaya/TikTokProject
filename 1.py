import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

data = pd.read_csv("D:/portfolio/tiktok/tiktok_dataset.csv")
data.head(10)
print(data.head(10))
data.info()
data.describe()
data['claim_status'].value_counts()
print(data['claim_status'].value_counts())

claims = data[data['claim_status'] == 'claim']

print('Mean view count claims:', claims['video_view_count'].mean())

print('Median view count claims:', claims['video_view_count'].median())

data.groupby(['claim_status', 'author_ban_status']).count()[['#']]

data.groupby(['author_ban_status']).agg(
    {'video_view_count': ['mean', 'median'],
     'video_like_count': ['mean', 'median'],
     'video_share_count': ['mean', 'median']})

print(data.groupby(['claim_status', 'author_ban_status']).count()[['#']])
print(data.groupby(['author_ban_status']).agg(
    {'video_view_count': ['mean', 'median'],
     'video_like_count': ['mean', 'median'],
     'video_share_count': ['mean', 'median']}))

data.groupby(['author_ban_status']).agg(
    {'video_view_count': ['count', 'mean', 'median'],
     'video_like_count': ['count', 'mean', 'median'],
     'video_share_count': ['count', 'mean', 'median']
     })
print(data.groupby(['author_ban_status']).agg(
    {'video_view_count': ['count', 'mean', 'median'],
     'video_like_count': ['count', 'mean', 'median'],
     'video_share_count': ['count', 'mean', 'median']
     }))


# Create a likes_per_view column
data['likes_per_view'] = data['video_like_count'] / data['video_view_count']

# Create a comments_per_view column
data['comments_per_view'] = data['video_comment_count'] / data['video_view_count']

# Create a shares_per_view column
data['shares_per_view'] = data['video_share_count'] / data['video_view_count']

data.groupby(['claim_status', 'author_ban_status']).agg(
    {'likes_per_view': ['count', 'mean', 'median'],
     'comments_per_view': ['count', 'mean', 'median'],
     'shares_per_view': ['count', 'mean', 'median']})

print(data.groupby(['claim_status', 'author_ban_status']).agg(
    {'likes_per_view': ['count', 'mean', 'median'],
     'comments_per_view': ['count', 'mean', 'median'],
     'shares_per_view': ['count', 'mean', 'median']}))




# Create a boxplot to visualize distribution of `video_duration_sec`
plt.figure(figsize=(5,1))
plt.title('video_duration_sec')
sns.boxplot(x=data['video_duration_sec'])
plt.show()

plt.figure(figsize=(5,3))
sns.histplot(data['video_duration_sec'], bins=range(0,61,5))
plt.title('Video duration histogram')
plt.show()

# Create a boxplot to visualize distribution of `video_view_count`
plt.figure(figsize=(5, 1))
plt.title('video_view_count')
sns.boxplot(x=data['video_view_count']);

plt.figure(figsize=(5,3))
sns.histplot(data['video_view_count'], bins=range(0,(10**6+1),10**5))
plt.title('Video view count histogram');

# Create a boxplot to visualize distribution of `video_like_count`
plt.figure(figsize=(10,1))
plt.title('video_like_count')
sns.boxplot(x=data['video_like_count']);

# plt.figure(figsize=(5,3))
ax = sns.histplot(data['video_like_count'], bins=range(0,(7*10**5+1),10**5))
labels = [0] + [str(i) + 'k' for i in range(100, 701, 100)]
ax.set_xticks(range(0,7*10**5+1,10**5), labels=labels)
plt.title('Video like count histogram');

# Create a boxplot to visualize distribution of `video_comment_count`
plt.figure(figsize=(5,1))
plt.title('video_comment_count')
sns.boxplot(x=data['video_comment_count']);

plt.figure(figsize=(5,3))
sns.histplot(data['video_comment_count'], bins=range(0,(3001),100))
plt.title('Video comment count histogram');

plt.figure(figsize=(5,1))
plt.title('video_share_count')
sns.boxplot(x=data['video_share_count']);

plt.figure(figsize=(5,3))
sns.histplot(data['video_share_count'], bins=range(0,(270001),10000))
plt.title('Video share count histogram');

# Create a boxplot to visualize distribution of `video_download_count`
plt.figure(figsize=(5,1))
plt.title('video_download_count')
sns.boxplot(x=data['video_download_count']);

plt.figure(figsize=(5,3))
sns.histplot(data['video_download_count'], bins=range(0,(15001),500))
plt.title('Video download count histogram');

plt.figure(figsize=(7,4))
sns.histplot(data=data,
             x='claim_status',
             hue='verified_status',
             multiple='dodge',
             shrink=0.9)
plt.title('Claims by verification status histogram');

fig = plt.figure(figsize=(7,4))
sns.histplot(data, x='claim_status', hue='author_ban_status',
             multiple='dodge',
             hue_order=['active', 'under review', 'banned'],
             shrink=0.9,
             palette={'active':'green', 'under review':'orange', 'banned':'red'},
             alpha=0.5)
plt.title('Claim status by author ban status - counts');

ban_status_counts = data.groupby(['author_ban_status']).median(
    numeric_only=True).reset_index()

fig = plt.figure(figsize=(5,3))
sns.barplot(data=ban_status_counts,
            x='author_ban_status',
            y='video_view_count',
            order=['active', 'under review', 'banned'],
            palette={'active':'green', 'under review':'orange', 'banned':'red'},
            alpha=0.5)
plt.title('Median view count by ban status');

data.groupby('claim_status')['video_view_count'].median()

fig = plt.figure(figsize=(3,3))
plt.pie(data.groupby('claim_status')['video_view_count'].sum(), labels=['claim', 'opinion'])
plt.title('Total views by video claim status');

count_cols = ['video_view_count',
              'video_like_count',
              'video_share_count',
              'video_download_count',
              'video_comment_count',
              ]

for column in count_cols:
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    median = data[column].median()
    outlier_threshold = median + 1.5*iqr

    # Count the number of values that exceed the outlier threshold
    outlier_count = (data[column] > outlier_threshold).sum()
    print(f'Number of outliers, {column}:', outlier_count)

    # Create a scatterplot of `video_view_count` versus `video_like_count` according to 'claim_status'
sns.scatterplot(x=data["video_view_count"], y=data["video_like_count"],
                hue=data["claim_status"], s=10, alpha=.3)
plt.show()

# Create a scatterplot of `video_view_count` versus `video_like_count` for opinions only
opinion = data[data['claim_status']=='opinion']
sns.scatterplot(x=opinion["video_view_count"], y=opinion["video_like_count"],
                 s=10, alpha=.3)
plt.show()

# Visualization 1: Bar plot of claim_status counts
sns.countplot(x='claim_status', data=data)
plt.title('Count of Claim Status')
plt.xlabel('Claim Status')
plt.ylabel('Count')
plt.show()

# Visualization 2: Boxplot of video_view_count by claim_status
sns.boxplot(x='claim_status', y='video_view_count', data=data)
plt.title('Video View Count by Claim Status')
plt.xlabel('Claim Status')
plt.ylabel('Video View Count')
plt.yscale('log')  # Use log scale for better visualization if values vary widely
plt.show()

# Visualization 3: Scatter plot of likes_per_view vs shares_per_view
sns.scatterplot(x='likes_per_view', y='shares_per_view', hue='claim_status', data=data)
plt.title('Likes per View vs Shares per View')
plt.xlabel('Likes per View')
plt.ylabel('Shares per View')
plt.show()

# Visualization 4: Histogram of video_view_count
sns.histplot(data['video_view_count'], bins=30, kde=True)
plt.title('Distribution of Video View Count')
plt.xlabel('Video View Count')
plt.ylabel('Frequency')
plt.show()

# Visualization 5: Bar plot of author_ban_status counts
sns.countplot(x='author_ban_status', data=data)
plt.title('Count of Author Ban Status')
plt.xlabel('Author Ban Status')
plt.ylabel('Count')
plt.show()

# Visualization 6: Bar plot of mean video_view_count by claim_status
mean_views = data.groupby('claim_status')['video_view_count'].mean().reset_index()
sns.barplot(x='claim_status', y='video_view_count', data=mean_views)
plt.title('Mean Video View Count by Claim Status')
plt.xlabel('Claim Status')
plt.ylabel('Mean Video View Count')
plt.show()

# Visualization 7: Stacked bar chart of claim_status by author_ban_status
stacked_data = data.groupby(['author_ban_status', 'claim_status']).size().unstack()
stacked_data.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Claim Status by Author Ban Status')
plt.xlabel('Author Ban Status')
plt.ylabel('Count')
plt.legend(title='Claim Status')
plt.show()

# Visualization 8: Boxplot of video_like_count by author_ban_status
sns.boxplot(x='author_ban_status', y='video_like_count', data=data)
plt.title('Video Like Count by Author Ban Status')
plt.xlabel('Author Ban Status')
plt.ylabel('Video Like Count')
plt.yscale('log')  # Use log scale for better visualization
plt.show()

# Visualization 9: Pairplot of numerical columns
sns.pairplot(data[['video_view_count', 'video_like_count', 'video_share_count', 'likes_per_view', 'shares_per_view']])
plt.suptitle('Pairplot of Numerical Columns', y=1.02)
plt.show()

# Visualization 10: Heatmap of correlations between numerical columns
correlation_matrix = data[['video_view_count', 'video_like_count', 'video_share_count', 'likes_per_view', 'shares_per_view']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()



