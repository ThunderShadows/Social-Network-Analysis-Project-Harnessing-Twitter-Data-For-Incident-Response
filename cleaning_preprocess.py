import pandas as pd
import re
from collections import Counter

# Load the training dataset
print("Loading train.csv...")
df = pd.read_csv('~/Desktop/SNA_Project/tweet_data.csv')
print(f"Loaded {len(df)} tweets")

# Function to extract hashtags from tweets
def extract_hashtags(text):
    if pd.isna(text):
        return []
    return [tag.lower() for tag in re.findall(r'#\w+', str(text))]

# Function to extract mentions from tweets
def extract_mentions(text):
    if pd.isna(text):
        return []
    return [mention.lower() for mention in re.findall(r'@\w+', str(text))]

# Function to extract keywords (words longer than 3 characters, excluding URLs and mentions)
def extract_keywords(text):
    if pd.isna(text):
        return []
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', str(text))
    # Remove mentions and hashtags
    text = re.sub(r'@\w+|#\w+', '', text)
    # Extract words (alphanumeric, length > 3)
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    return words

print("\nExtracting hashtags, mentions, and keywords...")
df['hashtags'] = df['text'].apply(extract_hashtags)
df['mentions'] = df['text'].apply(extract_mentions)
df['keywords'] = df['text'].apply(extract_keywords)

# Count tweets with hashtags
tweets_with_hashtags = df[df['hashtags'].apply(len) > 0]
print(f"Tweets with hashtags: {len(tweets_with_hashtags)}")

# ========================================
# METHOD 1: HASHTAG CO-OCCURRENCE NETWORK
# ========================================
print("\n=== Creating Hashtag Co-occurrence Network ===")

# Create edges between hashtags that appear in the same tweet
edges = []
for idx, row in df.iterrows():
    tags = row['hashtags']
    if len(tags) > 1:
        # Create edges for all pairs of hashtags in the tweet
        for i in range(len(tags)):
            for j in range(i+1, len(tags)):
                edges.append({
                    'Source': tags[i],
                    'Target': tags[j],
                    'Type': 'Undirected',
                    'tweet_id': row['id'],
                    'is_disaster': row['target']
                })

# Create edge dataframe
if edges:
    edge_df = pd.DataFrame(edges)
    
    # Aggregate edges: count co-occurrences and calculate disaster proportion
    edge_stats = edge_df.groupby(['Source', 'Target']).agg({
        'Type': 'first',
        'is_disaster': ['sum', 'count', 'mean']
    }).reset_index()
    
    edge_stats.columns = ['Source', 'Target', 'Type', 'disaster_count', 'Weight', 'disaster_ratio']
    
    # Save edges
    edge_stats.to_csv('edges.csv', index=False)
    print(f"Created edges.csv with {len(edge_stats)} edges")
else:
    print("No hashtag co-occurrences found. Creating keyword-based network instead...")

# Create nodes from hashtags
all_hashtags = []
hashtag_disaster_map = {}

for idx, row in df.iterrows():
    for tag in row['hashtags']:
        all_hashtags.append(tag)
        if tag not in hashtag_disaster_map:
            hashtag_disaster_map[tag] = {'disaster': 0, 'non_disaster': 0, 'total': 0}
        
        hashtag_disaster_map[tag]['total'] += 1
        if row['target'] == 1:
            hashtag_disaster_map[tag]['disaster'] += 1
        else:
            hashtag_disaster_map[tag]['non_disaster'] += 1

# Count hashtag frequencies
hashtag_counts = Counter(all_hashtags)

# Create node list with attributes
nodes = []
for tag, count in hashtag_counts.items():
    disaster_count = hashtag_disaster_map[tag]['disaster']
    non_disaster_count = hashtag_disaster_map[tag]['non_disaster']
    disaster_ratio = disaster_count / count if count > 0 else 0
    
    nodes.append({
        'Id': tag,
        'Label': tag,
        'Frequency': count,
        'disaster_count': disaster_count,
        'non_disaster_count': non_disaster_count,
        'disaster_ratio': disaster_ratio,
        'node_type': 'hashtag'
    })

nodes_df = pd.DataFrame(nodes)

# Save nodes
nodes_df.to_csv('nodes.csv', index=False)
print(f"Created nodes.csv with {len(nodes_df)} nodes")

# ========================================
# ALTERNATIVE: KEYWORD CO-OCCURRENCE NETWORK
# (Uncomment if hashtag network is too sparse)
# ========================================

# Filter common disaster-related keywords
common_words = ['fire', 'flood', 'earthquake', 'storm', 'hurricane', 'disaster', 
                'emergency', 'damage', 'death', 'rescue', 'killed', 'injured',
                'explosion', 'crash', 'attack', 'threat', 'warning', 'evacuation']

print("\n=== Alternative: Creating Keyword Network ===")

# Create keyword edges
keyword_edges = []
for idx, row in df.iterrows():
    # Filter to only disaster-related keywords
    keywords = [k for k in row['keywords'] if k in common_words]
    
    if len(keywords) > 1:
        for i in range(len(keywords)):
            for j in range(i+1, len(keywords)):
                keyword_edges.append({
                    'Source': keywords[i],
                    'Target': keywords[j],
                    'Type': 'Undirected',
                    'is_disaster': row['target']
                })

if keyword_edges:
    keyword_edge_df = pd.DataFrame(keyword_edges)
    keyword_edge_stats = keyword_edge_df.groupby(['Source', 'Target']).agg({
        'Type': 'first',
        'is_disaster': ['sum', 'count', 'mean']
    }).reset_index()
    
    keyword_edge_stats.columns = ['Source', 'Target', 'Type', 'disaster_count', 'Weight', 'disaster_ratio']
    keyword_edge_stats.to_csv('edges_keywords.csv', index=False)
    print(f"Created edges_keywords.csv with {len(keyword_edge_stats)} edges")
    
    # Create keyword nodes
    all_keywords = []
    for keywords in df['keywords']:
        all_keywords.extend([k for k in keywords if k in common_words])
    
    keyword_counts = Counter(all_keywords)
    keyword_nodes = []
    
    for keyword, count in keyword_counts.items():
        disaster_tweets = df[df['keywords'].apply(lambda x: keyword in x) & (df['target'] == 1)]
        disaster_count = len(disaster_tweets)
        disaster_ratio = disaster_count / count if count > 0 else 0
        
        keyword_nodes.append({
            'Id': keyword,
            'Label': keyword,
            'Frequency': count,
            'disaster_count': disaster_count,
            'non_disaster_count': count - disaster_count,
            'disaster_ratio': disaster_ratio,
            'node_type': 'keyword'
        })
    
    keyword_nodes_df = pd.DataFrame(keyword_nodes)
    keyword_nodes_df.to_csv('nodes_keywords.csv', index=False)
    print(f"Created nodes_keywords.csv with {len(keyword_nodes_df)} nodes")

# ========================================
# STATISTICS SUMMARY
# ========================================
print("\n=== Network Statistics ===")
print(f"Total tweets analyzed: {len(df)}")
print(f"Disaster tweets: {len(df[df['target'] == 1])}")
print(f"Non-disaster tweets: {len(df[df['target'] == 0])}")
print(f"\nHashtag Network:")
print(f"  - Nodes (hashtags): {len(nodes_df)}")
print(f"  - Edges (co-occurrences): {len(edge_stats) if edges else 0}")
if keyword_edges:
    print(f"\nKeyword Network:")
    print(f"  - Nodes (keywords): {len(keyword_nodes_df)}")
    print(f"  - Edges (co-occurrences): {len(keyword_edge_stats)}")

print("\n=== Files Created ===")
print("1. nodes.csv - Hashtag nodes with disaster statistics")
print("2. edges.csv - Hashtag co-occurrence edges")
