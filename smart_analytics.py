import pandas as pd
import numpy as np                    #Customer Segmentation and Product Recommendations 
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('online_retail.csv') # loading our dataset

df.info()                                        # info about our dataset

print(f"shape of the dataset: {df.shape}")       # overview of our dataset
print(f"columns:{list(df.columns)}")
print("\nFirst few rows:")
print(df.head())


print("Missing values in columns:")               # missing values in the dataset 
print(df.isnull().sum())
print(f"sum of missing values : {df.isnull().sum().sum()}")


print("\n data types:")
print(df.dtypes)                                # datatypes in datasets



original_rows = len(df)                                   # removing the missing values from the dataset
print(f"original dataset size: {original_rows} rows")

print("removing rows with missing CustomerID")
rows_with_missing_customerid = df['CustomerID'].isnull().sum()
print(f"rows_with_missing_customerid : {rows_with_missing_customerid}")

df_cleaned = df.dropna(subset=['CustomerID'])
rows_after_customerid_removal = len(df_cleaned)

print(f"Rows after removing missing CustomerID: {rows_after_customerid_removal}")
print(f"Rows removed: {original_rows - rows_after_customerid_removal}")          # removed the missing from rows 



print("excluding cancelled invoices (invoices starting with the letter 'c')")

cancelled_invoices = df_cleaned['InvoiceNo'].astype(str).str.startswith('C')      # cancelling the invoiceno with c
cancelled_count = cancelled_invoices.sum()
print(f"cancelled invoice found:{cancelled_count}")


if cancelled_count > 0:                                              # checking the canclled samples 
    cancelled_samples = df_cleaned[cancelled_invoices]['InvoiceNo'].unique()[:5]
    print(cancelled_samples)


df_cleaned = df_cleaned[~cancelled_invoices]  # not cancelled invoices '~' used as not operator 
rows_after_cancellation_removal = len(df_cleaned)

print(f"Rows after removing cancelled invoices: {rows_after_cancellation_removal}")
print(f"cancelled invoice rows removed : {rows_after_customerid_removal - rows_after_cancellation_removal}")


print("removing negative or zero quantities")

negative_quantity = (df_cleaned['Quantity'] < 0).sum()      # Check for negative and zero quantities and prices
zero_quantity = (df_cleaned['Quantity'] == 0).sum()
negative_price = (df_cleaned['UnitPrice'] < 0).sum()
zero_price = (df_cleaned['UnitPrice'] == 0).sum()

print(f"Rows with negative quantity: {negative_quantity}")
print(f"Rows with zero quantity: {zero_quantity}")
print(f"Rows with negative unit price: {negative_price}")
print(f"Rows with zero unit price: {zero_price}")

total_invalid_rows = ((df_cleaned['Quantity'] <= 0) | (df_cleaned['UnitPrice'] <= 0)).sum()
print(f"Total rows with negative or zero quantities or prices:{total_invalid_rows}")

df_cleaned = df_cleaned[(df_cleaned['Quantity'] > 0) & (df_cleaned['UnitPrice'] > 0)]
rows_after_negative_removal = len(df_cleaned)
print(f"Rows after removing negative/zero quantities and prices: {rows_after_negative_removal}")
print(f"Invalid rows removed: {rows_after_cancellation_removal - rows_after_negative_removal}")


print("converting data types:")   # Data type conversions

df_cleaned['CustomerID'] = df_cleaned['CustomerID'].astype(int)
df_cleaned['InvoiceDate'] = pd.to_datetime(df_cleaned['InvoiceDate'])


# creating some useful columns 

df_cleaned['TotalAmount'] = df_cleaned['Quantity'] * df_cleaned['UnitPrice']
df_cleaned['Year'] = df_cleaned['InvoiceDate'].dt.year
df_cleaned['Month'] = df_cleaned['InvoiceDate'].dt.month

df_cleaned['Day'] = df_cleaned['InvoiceDate'].dt.day

print("Data types after conversion:")
print(df_cleaned.dtypes)

print(" PREPROCESSING SUMMARY ")
print(f"Original dataset: {original_rows} rows")
print(f"Final cleaned dataset: {len(df_cleaned)} rows")
print(f"Total rows removed: {original_rows - len(df_cleaned)}")
print(f"Data retention rate: {(len(df_cleaned)/original_rows)*100:.2f}%")


# getting unique values for overall view of the cleaned data which is preprocessed

print(f"unique customers : {df_cleaned['CustomerID'].nunique()}")
print(f"Unique Invoices : {df_cleaned['InvoiceNo'].nunique()}")
print(f"unique Unique products : {df_cleaned['StockCode'].nunique()}")
print(f"Date range: {df_cleaned['InvoiceDate'].min()} to {df_cleaned['InvoiceDate'].max()}")


# cleaned dataset overview
print("First few rows of cleaned data:")
print(df_cleaned.head())

#basic statistics
print(df_cleaned[['Quantity', 'UnitPrice', 'TotalAmount']].describe())

# check for remaining missing values 
print(f"remaining missing values :{ df_cleaned.isnull().sum().sum()}")

#save the cleaned dataset

df_cleaned.to_csv('online_retail.csv' , index = False)
print("Cleaned dataset saved as 'online_retail.csv'")


#visualizing the data

plt.subplot(2, 2, 2)
monthly_transactions = df_cleaned.groupby(['Year', 'Month']).size()
monthly_transactions.plot(kind='line', marker='o')
plt.title('Monthly Transaction Trends')
plt.xlabel('Time Period')
plt.ylabel('Number of Transactions')
plt.show()

plt.subplot(2, 2, 3)
top_customers = df_cleaned['CustomerID'].value_counts().head(10)
plt.bar(range(len(top_customers)), top_customers.values)
plt.title('Top 10 Customers by Transaction Count')
plt.xlabel('Customer Rank')
plt.ylabel('Number of Transactions')
plt.show()

# transaction count satisttics , 

plt.subplot(4 , 3 , 4)
top_countries = df_cleaned['Country'].value_counts().head(8)
plt.bar(top_countries.index , top_countries.values, color = 'skyblue' )
plt.title('Top contries by order' , fontweight = 'bold')
plt.xticks(rotation = 90)
plt.ylabel('Orders')
print(f"Top country: {top_countries.index[0]} with {top_countries.iloc[0]:,} orders")
plt.show()

print("\n2. Top-Selling Products...")

plt.subplot(2, 3, 2)
product_sales = df_cleaned.groupby('StockCode')['Quantity'].sum().sort_values(ascending=False).head(10)
plt.barh(range(len(product_sales)), product_sales.values, color='lightcoral')
plt.title('Top Products by Quantity', fontweight='bold')
plt.yticks(range(len(product_sales)), product_sales.index)
plt.xlabel('Quantity Sold')

print(f"Best seller: {product_sales.index[0]} with {product_sales.iloc[0]:,} units")
plt.show()


print("\n3. Purchase Trends Over Time...")

plt.subplot(4, 3, 3)
monthly_sales = df_cleaned.groupby(df_cleaned['InvoiceDate'].dt.to_period('M'))['TotalAmount'].sum()
plt.plot(range(len(monthly_sales)), monthly_sales.values, marker='o', color='green', linewidth=2)
plt.title('Monthly Sales Trend', fontweight='bold')
plt.xlabel('Months')
plt.ylabel('Sales ($)')
plt.grid(True, alpha=0.3)

print(f"Total revenue: ${monthly_sales.sum():,.2f}")
plt.show()


print("\n4. Monetary Distribution...")

plt.subplot(4, 3, 4)
plt.hist(df_cleaned['TotalAmount'], bins=30, color='purple', alpha=0.7)
plt.title('Transaction Amount Distribution', fontweight='bold')
plt.xlabel('Amount ($)')
plt.ylabel('Frequency')
plt.yscale('log')

avg_amount = df_cleaned['TotalAmount'].mean()
print(f"Average transaction: ${avg_amount:.2f}")
plt.show()

plt.subplot(4, 3, 5)
customer_spending = df_cleaned.groupby('CustomerID')['TotalAmount'].sum()
plt.hist(customer_spending, bins=30, color='red', alpha=0.7)
plt.title('Customer Spending Distribution', fontweight='bold')
plt.xlabel('Total Spending ($)')
plt.ylabel('Customers')

print(f"Average customer spending: ${customer_spending.mean():.2f}")
plt.show()

print("\n5. 📊 RFM Analysis...")




#Data Preprocessing → RFM Analysis → K-means Clustering → Model Saving → Streamlit App

# Calculate RFM
current_date = df_cleaned['InvoiceDate'].max() + pd.Timedelta(days=1)
rfm = df_cleaned.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (current_date - x.max()).days,  # Recency
    'InvoiceNo': 'nunique',  # Frequency
    'TotalAmount': 'sum'  # Monetary
}).rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'TotalAmount': 'Monetary'})

plt.subplot(4, 3, 6)
plt.hist(rfm['Recency'], bins=25, color='blue', alpha=0.7)
plt.title('Recency Distribution', fontweight='bold')
plt.xlabel('Days Since Last Purchase')
plt.ylabel('Customers')
plt.show()

plt.subplot(4, 3, 7)
plt.hist(rfm['Frequency'], bins=25, color='green', alpha=0.7)
plt.title('Frequency Distribution', fontweight='bold')
plt.xlabel('Number of Orders')
plt.ylabel('Customers')
plt.show()


plt.subplot(4, 3, 8)
plt.hist(rfm['Monetary'], bins=25, color='red', alpha=0.7)
plt.title('Monetary Distribution', fontweight='bold')
plt.xlabel('Total Spending ($)')
plt.ylabel('Customers')
plt.xscale('log')
plt.show()


print(f"RFM calculated for {len(rfm)} customers")


# data clsuterig using k means

print('elbow cuurve for clustering')

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)

scores = []
for k in [2, 3, 4, 5, 6]:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(rfm_scaled)
    scores.append(kmeans.inertia_)

    plt.subplot(4, 3, 9)
plt.plot([2, 3, 4, 5, 6], scores, 'ro-', linewidth=2)
plt.title('Elbow Method', fontweight='bold')
plt.xlabel('Clusters')
plt.ylabel('Score')
plt.grid(True)
plt.show()

optimal_k = 4
print(f"Using {optimal_k} clusters")


# customer clustering 

print("customer clustering")

kmeans = KMeans(n_clusters=4 , random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)   #add new column showing which group each customer belongs to


cluster_names = {
    0: 'Regular Customers',
    1: 'High-Value Customers', 
    2: 'At-Risk Customers',
    3: 'New Customers'
}   

plt.subplot(4, 3, 10)
cluster_counts = rfm['Cluster'].value_counts().sort_index()
plt.bar(cluster_counts.index, cluster_counts.values, color=['red', 'blue', 'green', 'orange'])
plt.title('Customer Groups', fontweight='bold')
plt.xlabel('Group')
plt.ylabel('Customers')
plt.show()

print('Groups Created')

print("Groups created:")
for i in range(4):
    count = (rfm['Cluster'] == i).sum()
    print(f"  Group {i}: {count} customers")



# product recommendation matrix 

top_products = df_cleaned['StockCode'].value_counts().head(10).index

# Simple correlation matrix
product_data = df_cleaned[df_cleaned['StockCode'].isin(top_products)]
product_matrix = product_data.pivot_table(index='CustomerID', columns='StockCode', values='Quantity', fill_value=0)
product_corr = product_matrix.corr()

plt.subplot(4, 3, 11)
plt.imshow(product_corr, cmap='coolwarm')
plt.title('Product Similarity', fontweight='bold')
plt.colorbar()
plt.show()

print(f"Matrix ready: {len(top_products)} products")



# saving the file data
import joblib
joblib.dump(kmeans, 'customer_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
rfm.to_csv('customers_with_groups.csv')
print(" Model saved!")

print("\n SIMPLE RECOMMENDATION SYSTEM")
print("="*40)

# 1️ Make customer product table
customer_product = df_cleaned.pivot_table(
    index='CustomerID', 
    columns='StockCode', 
    values='Quantity', 
    fill_value=0
)
print(f" Table made: {customer_product.shape[0]} customers, {customer_product.shape[1]} products")

# 2️ Find which products are similar
from sklearn.metrics.pairwise import cosine_similarity
product_similarity = cosine_similarity(customer_product.T)
print("Found similar products")

# 3️ Get product names
product_names = df_cleaned.groupby('StockCode')['Description'].first().fillna('Unknown')

# 4️ Simple recommendation function
def recommend_products(product_code, how_many=5):
    """Find similar products"""
    try:
        # Get all products
        all_products = list(customer_product.columns)
        
        # Check if product exists
        if product_code not in all_products:
            return f"Product {product_code} not found"
        
        # Find where this product is in our list
        product_index = all_products.index(product_code)
        
        # Get similarity scores for this product
        similarities = product_similarity[product_index]
        
        # Find most similar products
        similar_indices = similarities.argsort()[::-1][1:how_many+1]  # Skip first (itself)
        
        # Make recommendation list
        recommendations = []
        for idx in similar_indices:
            similar_product = all_products[idx]
            score = similarities[idx]
            name = product_names.get(similar_product, 'Unknown')
            
            recommendations.append({
                'Product': similar_product,
                'Name': name[:30],  # Short name
                'Score': round(score, 2)
            })
        
        return recommendations
    
    except:
        return "Error making recommendations"

# 5️ Test recommendations
popular_product = df_cleaned['StockCode'].value_counts().index[0]
print(f"\nRecommendations for product '{popular_product}':")

recs = recommend_products(popular_product)
if isinstance(recs, list):
    for i, rec in enumerate(recs, 1):
        print(f"{i}. {rec['Product']}: {rec['Name']} (Score: {rec['Score']})")

# 6️ Customer recommendations based on their group
def recommend_for_customer(customer_id, how_many=5):
    """Recommend products for a customer based on their group"""
    try:
        # Check if customer exists
        if customer_id not in rfm.index:
            return f"Customer {customer_id} not found"
        
        # Get customer's group
        customer_group = rfm.loc[customer_id, 'Cluster']
        
        # Find all customers in same group
        same_group_customers = rfm[rfm['Cluster'] == customer_group].index
        
        # See what products this group likes
        group_purchases = df_cleaned[df_cleaned['CustomerID'].isin(same_group_customers)]
        popular_in_group = group_purchases['StockCode'].value_counts()
        
        # See what this customer already bought
        customer_bought = df_cleaned[df_cleaned['CustomerID'] == customer_id]['StockCode'].unique()
        
        # Recommend products they haven't bought yet
        recommendations = []
        for product, count in popular_in_group.items():
            if product not in customer_bought and len(recommendations) < how_many:
                name = product_names.get(product, 'Unknown')
                recommendations.append({
                    'Product': product,
                    'Name': name[:30],
                    'PopularityInGroup': count
                })
        
        return recommendations
    
    except:
        return "Error making recommendations"

# Test customer recommendations
test_customer = rfm.index[0]
print(f"\nRecommendations for Customer {test_customer}:")
customer_recs = recommend_for_customer(test_customer)
if isinstance(customer_recs, list):
    for i, rec in enumerate(customer_recs, 1):
        print(f"{i}. {rec['Product']}: {rec['Name']} (Popular: {rec['PopularityInGroup']} times)")

# 7️ Save recommendation data
import numpy as np
np.save('product_similarities.npy', product_similarity)
customer_product.to_csv('customer_products.csv')
product_names.to_csv('product_names.csv')

print("Files saved:")
print("- customer_model.pkl (for grouping customers)")
print("- scaler.pkl (for processing new customers)")  
print("- customers_with_groups.csv (customer data)")
print("- product_similarities.npy (for recommendations)")
print("- customer_products.csv (purchase data)")
print("- product_names.csv (product info)")



# streamlit app 

print("\n🚀 CREATING STREAMLIT APP")

app_code = '''import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="🛍️ Shopper Scope ", layout="wide")
st.markdown('<h1 style="text-align: center; color: #FF6B6B;">🛍️ Shopper Scope </h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 20px;">AI-Powered Product Recommendations & Customer Intelligence</p>', unsafe_allow_html=True)

@st.cache_data
def load_data():
    return (joblib.load('customer_model.pkl'), joblib.load('scaler.pkl'), 
            pd.read_csv('customers_with_groups.csv', index_col=0),
            np.load('product_similarities.npy'),
            pd.read_csv('customer_products.csv', index_col=0),
            pd.read_csv('product_names.csv', index_col=0))

model, scaler, customers, similarities, customer_products, product_names = load_data()

tab1, tab2 = st.tabs([" Product Recommendations", "👤 Customer Segment"])

with tab1:
    st.markdown("### Find Products Your Customers Will Love")
    
    products = list(customer_products.columns)
    selected = st.selectbox("🔍 Choose a product:", products)
    
    if st.button(" Get Recommendations", type="primary"):
        idx = products.index(selected)
        scores = similarities[idx]
        top5 = scores.argsort()[::-1][1:6]
        
        st.markdown("#### 🏆 Top 5 Similar Products:")
        
        for i, product_idx in enumerate(top5):
            product_code = products[product_idx]
            name = product_names.loc[product_code].values[0] if product_code in product_names.index else "Unknown"
            score = scores[product_idx]
            
            st.markdown(f"""
            <div style="background: linear-gradient(90deg, #4CAF50, #45a049); 
                        color: white; padding: 15px; margin: 10px 0; 
                        border-radius: 10px; border-left: 5px solid #2E7D32;">
                <h4>{i+1} 🎁 {product_code}</h4>
                <p>📝 {name[:40]}...</p>
                <p>⭐ Similarity Score: {score:.2f}</p>
            </div>
            """, unsafe_allow_html=True)

with tab2:
    st.markdown("### Predict Customer Behavior")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        recency = st.slider("📅 Days Since Last Purchase", 0, 365, 30)
    with col2:
        frequency = st.slider("🔄 Number of Orders", 1, 50, 5)
    with col3:
        monetary = st.slider("💰 Total Spent ($)", 0, 5000, 200)
    
    if st.button("🔮 Predict Customer Type", type="primary"):
        data = scaler.transform([[recency, frequency, monetary]])
        cluster = model.predict(data)[0]
        
        segments = {
            0: ("🔵 Regular Customer", "Steady and reliable shoppers", "#2196F3"),
            1: ("💎 VIP Customer", "Your most valuable customers!", "#4CAF50"), 
            2: ("⚠️ At-Risk Customer", "Need immediate attention", "#FF5722"),
            3: ("🌟 New Customer", "Fresh potential to nurture", "#FF9800")
        }
        
        title, desc, color = segments[cluster]
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {color}, {color}AA); 
                    color: white; padding: 30px; text-align: center; 
                    border-radius: 20px; margin: 20px 0; box-shadow: 0 10px 30px rgba(0,0,0,0.2);">
            <h2>{title}</h2>
            <h4>{desc}</h4>
            <p>Cluster ID: {cluster}</p>
        </div>
        """, unsafe_allow_html=True)
        
        if cluster == 1:
            st.success(" This customer is gold! Offer premium products and exclusive deals.")
        elif cluster == 2:
            st.error(" Send them a special discount to win them back!")
        elif cluster == 3:
            st.info(" Perfect time for welcome offers and onboarding.")
        else:
            st.info(" Keep them engaged with regular promotions.")

st.markdown("---")
st.markdown('<p style="text-align: center; color: #666;">Powered by AI & Machine Learning  (Gowri Nandhan) </p>', unsafe_allow_html=True)'''

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(app_code)

print(" App created as 'app.py'")
print(" Run: streamlit run app.py")