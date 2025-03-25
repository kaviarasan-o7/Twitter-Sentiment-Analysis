import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style for better visualizations
plt.style.use('ggplot')
sns.set_palette("pastel")

# Load the data
def load_data():
    try:
        # Try with a more permissive encoding that handles special characters better
        train_df = pd.read_csv('c:/Users/kavia/OneDrive/Desktop/prodigy_task3/twitter_training.csv', encoding='latin1')
        valid_df = pd.read_csv('c:/Users/kavia/OneDrive/Desktop/prodigy_task3/twitter_validation.csv', encoding='latin1')
        
        # Print column names to debug
        print("Training columns:", train_df.columns.tolist())
        print("Validation columns:", valid_df.columns.tolist())
        
        # Combine datasets if both exist
        df = pd.concat([train_df, valid_df], ignore_index=True)
        print(f"Loaded combined dataset with {len(df)} records")
        return df
    except Exception as e:
        # If one approach fails, try loading files individually
        print(f"Warning: {e}")
        try:
            df = pd.read_csv('c:/Users/kavia/OneDrive/Desktop/prodigy_task3/twitter_validation.csv', encoding='latin1')
            print("Validation columns:", df.columns.tolist())
            print(f"Loaded validation dataset with {len(df)} records")
            return df
        except Exception:
            try:
                df = pd.read_csv('c:/Users/kavia/OneDrive/Desktop/prodigy_task3/twitter_training.csv', encoding='latin1')
                print("Training columns:", df.columns.tolist())
                print(f"Loaded training dataset with {len(df)} records")
                return df
            except Exception as e:
                print(f"Error: Could not load either dataset file. {e}")
                return None

# Analyze sentiment distribution
def analyze_sentiment(df):
    # Count sentiments
    sentiment_counts = df['Sentiment'].value_counts()
    
    # Calculate percentages
    sentiment_percentages = df['Sentiment'].value_counts(normalize=True) * 100
    
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar chart
    colors = ['#5ab4ac', '#d8b365', '#8c96c6', '#f4a582']
    bars = ax1.bar(sentiment_counts.index, sentiment_counts.values, color=colors)
    
    # Add count labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    ax1.set_title('Sentiment Distribution', fontsize=16)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.tick_params(axis='x', labelsize=10)
    
    # Pie chart
    wedges, texts, autotexts = ax2.pie(
        sentiment_percentages, 
        labels=sentiment_percentages.index,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        wedgeprops={'edgecolor': 'w', 'linewidth': 1},
        textprops={'fontsize': 12}
    )
    
    # Make percentage labels bold
    for autotext in autotexts:
        autotext.set_fontweight('bold')
    
    ax2.set_title('Sentiment Distribution', fontsize=16)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('c:/Users/kavia/OneDrive/Desktop/prodigy_task3/sentiment_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return sentiment_counts, sentiment_percentages

# Analyze sentiment by entity/brand - Modified to handle different column names
def analyze_by_entity(df):
    # Check if 'Entity' column exists, if not try to find an alternative
    entity_column = None
    if 'Entity' in df.columns:
        entity_column = 'Entity'
    elif 'Brand' in df.columns:
        entity_column = 'Brand'
    else:
        # Try to identify a column that might contain entity/brand information
        # This is a fallback and might not be accurate
        potential_columns = [col for col in df.columns if col not in ['Sentiment', 'Tweet', 'tweet_length']]
        if potential_columns:
            entity_column = potential_columns[0]
    
    if entity_column is None:
        print("Could not find an entity/brand column. Skipping entity analysis.")
        return None, None
    
    print(f"Using '{entity_column}' as the entity/brand column")
    
    # Get top 10 entities by frequency
    top_entities = df[entity_column].value_counts().head(10).index.tolist()
    
    # Filter for top entities
    top_entities_df = df[df[entity_column].isin(top_entities)]
    
    # Create a crosstab of entity and sentiment
    entity_sentiment = pd.crosstab(top_entities_df[entity_column], top_entities_df['Sentiment'])
    
    # Calculate percentages
    entity_sentiment_pct = entity_sentiment.div(entity_sentiment.sum(axis=1), axis=0) * 100
    
    # Plot
    plt.figure(figsize=(14, 8))
    entity_sentiment_pct.plot(kind='bar', stacked=True, colormap='viridis')
    plt.title(f'Sentiment Distribution by Top 10 {entity_column}s', fontsize=16)
    plt.xlabel(entity_column, fontsize=12)
    plt.ylabel('Percentage', fontsize=12)
    plt.legend(title='Sentiment')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'c:/Users/kavia/OneDrive/Desktop/prodigy_task3/{entity_column.lower()}_sentiment.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return entity_sentiment, entity_sentiment_pct

# Generate additional insights
def generate_insights(df):
    # 1. Tweet length analysis by sentiment
    df['tweet_length'] = df['Tweet'].str.len()
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Sentiment', y='tweet_length', data=df)
    plt.title('Tweet Length by Sentiment', fontsize=16)
    plt.xlabel('Sentiment', fontsize=12)
    plt.ylabel('Tweet Length (characters)', fontsize=12)
    plt.tight_layout()
    plt.savefig('c:/Users/kavia/OneDrive/Desktop/prodigy_task3/tweet_length.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Word cloud for each sentiment (if wordcloud package is available)
    try:
        from wordcloud import WordCloud
        import re
        from collections import Counter
        
        # Function to clean text and handle encoding issues
        def clean_text(text):
            if isinstance(text, str):
                # Remove mentions, URLs, and special characters
                text = re.sub(r'@\w+', '', text)
                text = re.sub(r'http\S+', '', text)
                # Replace problematic characters with spaces
                text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII characters
                text = re.sub(r'[^\w\s]', '', text)
                return text.lower()
            return ""
        
        # Create word clouds for each sentiment
        sentiments = df['Sentiment'].unique().tolist()
        
        for sentiment in sentiments:
            # Get tweets for this sentiment
            sentiment_tweets = df[df['Sentiment'] == sentiment]['Tweet'].apply(clean_text)
            
            if len(sentiment_tweets) > 0:
                # Combine all tweets
                all_words = ' '.join(sentiment_tweets)
                
                # Generate word cloud
                wordcloud = WordCloud(width=800, height=400, background_color='white', 
                                     max_words=100, contour_width=3, contour_color='steelblue')
                wordcloud.generate(all_words)
                
                # Plot
                plt.figure(figsize=(10, 6))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.title(f'Word Cloud - {sentiment} Tweets', fontsize=16)
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(f'c:/Users/kavia/OneDrive/Desktop/prodigy_task3/wordcloud_{sentiment.lower()}.png', 
                           dpi=300, bbox_inches='tight')
                plt.show()
    except ImportError:
        print("WordCloud package not installed. Skipping word cloud generation.")
    except Exception as e:
        print(f"Error generating word clouds: {e}")

# Main function
def main():
    # Load data
    df = load_data()
    
    if df is not None:
        try:
            # Print the actual number of columns to understand the structure
            print(f"Number of columns in the dataset: {len(df.columns)}")
            
            # Instead of renaming, let's extract the columns we need
            # First, check if we already have the expected column names
            if 'Sentiment' not in df.columns and 'Entity' not in df.columns:
                # Extract the columns we need based on position
                # Assuming columns 0, 1, 2, 3 are ID, Entity, Sentiment, Tweet
                new_df = pd.DataFrame()
                new_df['ID'] = df.iloc[:, 0]
                new_df['Entity'] = df.iloc[:, 1]
                new_df['Sentiment'] = df.iloc[:, 2]
                new_df['Tweet'] = df.iloc[:, 3]
                df = new_df
                print("Created new DataFrame with required columns")
            
            print("\nData Overview:")
            print(f"Total tweets: {len(df)}")
            print(f"Available columns: {df.columns.tolist()}")
            
            # Now check if Sentiment column exists
            if 'Sentiment' in df.columns:
                print(f"Sentiment classes: {df['Sentiment'].unique().tolist()}")
                
                # Basic data info
                print("\nSample data:")
                print(df.head(3).to_string(index=False))  # Using to_string to avoid encoding issues
                
                # Analyze sentiment distribution
                print("\nAnalyzing sentiment distribution...")
                sentiment_counts, sentiment_percentages = analyze_sentiment(df)
                print("\nSentiment counts:")
                print(sentiment_counts)
                print("\nSentiment percentages:")
                print(sentiment_percentages.round(2))
                
                # Analyze by entity
                print("\nAnalyzing sentiment by entity/brand...")
                entity_sentiment, entity_sentiment_pct = analyze_by_entity(df)
                
                # Generate additional insights
                print("\nGenerating additional insights...")
                generate_insights(df)
                
                print("\nAnalysis complete! Visualizations have been saved to your project directory.")
            else:
                print("Error: Could not find 'Sentiment' column in the data.")
                print("Available columns:", df.columns.tolist())
        except Exception as e:
            print(f"An error occurred during analysis: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()