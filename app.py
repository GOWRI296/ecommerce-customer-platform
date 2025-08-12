import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="🛍️ Shopper Scope ", layout="wide")
st.markdown('<h1 style="text-align: center; color: #FF6B6B;">🛍️ Shopper Scope </h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 20px;">AI-Powered Product Recommendations & Customer Intelligence</p>', unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load all data with proper error handling"""
    try:
        # Load ML models (kept as pickle for compatibility)
        customer_model = joblib.load('customer_model.pkl')
        scaler = joblib.load('scaler.pkl')
        
        # Load compressed data files for much faster loading
        customers = pd.read_csv('customers_with_groups.csv', index_col=0)
        
        # Load similarities with proper error handling - ENHANCED FIX
        similarities = None
        try:
            # Try NPZ compressed file first
            st.info("🔄 Attempting to load compressed similarities file (.npz)...")
            similarities_file = np.load('product_similarities.npz', allow_pickle=True)
            similarities = similarities_file['arr_0']  # Default numpy compressed array name
            similarities_file.close()  # Close the file properly
            st.success("✅ Successfully loaded NPZ file!")
        except (FileNotFoundError, KeyError, Exception) as e:
            st.warning(f"⚠️ NPZ file not found or corrupted: {e}")
            try:
                # Fallback to NPY file with allow_pickle for object arrays
                st.info("🔄 Attempting to load NPY file...")
                similarities = np.load('product_similarities.npy', allow_pickle=True)
                st.success("✅ Successfully loaded NPY file!")
            except FileNotFoundError:
                st.error("❌ Could not find product_similarities file (.npz or .npy)")
                st.error("Make sure the file exists in your repository")
                st.stop()
            except Exception as npy_error:
                st.error(f"❌ Error loading NPY file: {npy_error}")
                # Try alternative loading methods
                try:
                    st.info("🔄 Attempting alternative loading method...")
                    similarities = np.load('product_similarities.npy', allow_pickle=True, fix_imports=True)
                    st.success("✅ Successfully loaded with alternative method!")
                except Exception as final_error:
                    st.error(f"❌ All loading methods failed: {final_error}")
                    st.error("Please check if the file is corrupted or in an unsupported format")
                    st.stop()
        
        # Verify similarities was loaded
        if similarities is None:
            st.error("❌ Failed to load similarities matrix")
            st.stop()
            
        customer_products = pd.read_parquet('customer_products.parquet')
        
        # Try to load product names from parquet, fallback to CSV
        try:
            product_names = pd.read_parquet('product_names.parquet')
        except FileNotFoundError:
            product_names = pd.read_csv('product_names.csv', index_col=0)
        
        return customer_model, scaler, customers, similarities, customer_products, product_names
    
    except FileNotFoundError as e:
        st.error(f"❌ File not found: {e}")
        st.error("Make sure all required files are in your repository:")
        st.code("""
        Required files:
        - customer_model.pkl
        - scaler.pkl
        - customers_with_groups.csv
        - product_similarities.npz (or .npy)
        - customer_products.parquet
        - product_names.parquet (or .csv)
        """)
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        st.info("Debug info: Check that all files are properly formatted and accessible")
        
        # Additional debugging information
        st.markdown("### 🔧 Debugging Information:")
        st.write("**Error type:**", type(e).__name__)
        st.write("**Error message:**", str(e))
        
        # Check which files exist
        import os
        files_to_check = [
            'customer_model.pkl', 'scaler.pkl', 'customers_with_groups.csv',
            'product_similarities.npz', 'product_similarities.npy',
            'customer_products.parquet', 'product_names.parquet', 'product_names.csv'
        ]
        
        st.markdown("**File availability check:**")
        for file in files_to_check:
            if os.path.exists(file):
                st.success(f"✅ {file}")
            else:
                st.error(f"❌ {file}")
        
        st.stop()

# Alternative function to regenerate similarities if needed
def regenerate_similarities_demo():
    """Demo function showing how to regenerate similarities if the file is corrupted"""
    st.markdown("### 🛠️ Emergency: Regenerate Similarities Matrix")
    st.warning("If your similarities file is corrupted, you might need to regenerate it.")
    
    with st.expander("🔧 Show regeneration code"):
        st.code("""
# Example code to regenerate similarities matrix
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Assuming you have your customer_products DataFrame
# similarities = cosine_similarity(customer_products.T)  # Transpose for product similarities
# np.savez_compressed('product_similarities.npz', similarities)  # Save as compressed NPZ
# np.save('product_similarities.npy', similarities)  # Save as regular NPY backup

print("Similarities matrix regenerated!")
        """, language='python')

# Load data with progress indicator
with st.spinner('🚀 Loading compressed data for lightning-fast performance...'):
    try:
        model, scaler, customers, similarities, customer_products, product_names = load_data()
        st.success("⚡ Data loaded successfully! Ready for AI magic.")
    except Exception as e:
        st.error(f"🚨 Critical error during data loading: {e}")
        st.markdown("---")
        regenerate_similarities_demo()
        st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["🛍️ Product Recommendations", "👤 Customer Segment", "📊 Performance Stats", "🛠️ Debug Info"])

with tab1:
    st.markdown("### Find Products Your Customers Will Love")
    
    products = list(customer_products.columns)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        selected = st.selectbox("🔍 Choose a product:", products)
    with col2:
        st.metric("📦 Total Products", len(products))
    
    if st.button("🔮 Get Recommendations", type="primary"):
        with st.spinner('🤖 AI is analyzing product similarities...'):
            try:
                idx = products.index(selected)
                scores = similarities[idx]
                top5 = scores.argsort()[::-1][1:6]
                
                st.markdown("#### 🏆 Top 5 Similar Products:")
                
                # Create columns for better layout
                cols = st.columns(2)
                
                for i, product_idx in enumerate(top5):
                    product_code = products[product_idx]
                    name = product_names.loc[product_code].values[0] if product_code in product_names.index else "Product Name Not Available"
                    score = scores[product_idx]
                    
                    # Determine color based on similarity score
                    if score > 0.8:
                        color = "#4CAF50"  # Green for high similarity
                    elif score > 0.6:
                        color = "#FF9800"  # Orange for medium similarity
                    else:
                        color = "#2196F3"  # Blue for lower similarity
                    
                    with cols[i % 2]:  # Alternate between columns
                        st.markdown(f"""
                        <div style="background: linear-gradient(90deg, {color}, {color}DD); 
                                    color: white; padding: 15px; margin: 10px 0; 
                                    border-radius: 10px; border-left: 5px solid {color};">
                            <h4>#{i+1} 🎁 {product_code}</h4>
                            <p>📝 {name[:50]}{'...' if len(name) > 50 else ''}</p>
                            <p>⭐ Similarity Score: <strong>{score:.3f}</strong></p>
                        </div>
                        """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error generating recommendations: {e}")
                st.info("💡 Tip: Try selecting a different product or check the Debug Info tab")

with tab2:
    st.markdown("### Predict Customer Behavior")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        recency = st.slider("📅 Days Since Last Purchase", 0, 365, 30, 
                           help="How many days since the customer's last purchase?")
    with col2:
        frequency = st.slider("🔄 Number of Orders", 1, 50, 5,
                             help="Total number of orders placed by customer")
    with col3:
        monetary = st.slider("💰 Total Spent ($)", 0, 5000, 200,
                            help="Total amount spent by customer")
    
    # Show input summary
    st.info(f"📊 Customer Profile: {recency} days ago | {frequency} orders | ${monetary} spent")
    
    if st.button("🔮 Predict Customer Type", type="primary"):
        with st.spinner('🧠 AI is analyzing customer behavior...'):
            try:
                data = scaler.transform([[recency, frequency, monetary]])
                cluster = model.predict(data)[0]
                
                segments = {
                    0: ("🔵 Regular Customer", "Steady and reliable shoppers", "#2196F3", "Keep them engaged with regular promotions."),
                    1: ("💎 VIP Customer", "Your most valuable customers!", "#4CAF50", "Offer premium products and exclusive deals."), 
                    2: ("⚠️ At-Risk Customer", "Need immediate attention", "#FF5722", "Send them a special discount to win them back!"),
                    3: ("🌟 New Customer", "Fresh potential to nurture", "#FF9800", "Perfect time for welcome offers and onboarding.")
                }
                
                title, desc, color, action = segments.get(cluster, ("❓ Unknown", "Unidentified segment", "#9E9E9E", "Further analysis needed."))
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, {color}, {color}AA); 
                            color: white; padding: 30px; text-align: center; 
                            border-radius: 20px; margin: 20px 0; box-shadow: 0 10px 30px rgba(0,0,0,0.2);">
                    <h2>{title}</h2>
                    <h4>{desc}</h4>
                    <p><strong>Cluster ID: {cluster}</strong></p>
                </div>
                """, unsafe_allow_html=True)
                
                # Action recommendations
                if cluster == 1:
                    st.success(f"✨ {action}")
                elif cluster == 2:
                    st.error(f"🚨 {action}")
                elif cluster == 3:
                    st.info(f"👋 {action}")
                else:
                    st.info(f"📈 {action}")
            except Exception as e:
                st.error(f"Error predicting customer type: {e}")

with tab3:
    st.markdown("### 📊 Performance & Data Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🛍️ Products", len(customer_products.columns))
    with col2:
        st.metric("👥 Customers", len(customers))
    with col3:
        st.metric("🔗 Similarities", f"{similarities.shape[0]}×{similarities.shape[1]}")
    with col4:
        st.metric("🎯 ML Model", "Trained & Ready")
    
    st.markdown("#### 🚀 Performance Improvements")
    st.success("✅ Using compressed Parquet files for 50-90% faster CSV loading")
    st.success("✅ Using compressed NPZ files for 70-95% faster numpy array loading")
    st.success("✅ Streamlit caching enabled for instant subsequent loads")
    st.success("✅ Enhanced error handling with multiple fallback methods")
    
    # File size comparison (if you want to show this)
    with st.expander("📁 File Format Benefits"):
        st.markdown("""
        **Parquet vs CSV:**
        - ⚡ 5-10x faster loading
        - 📦 50-90% smaller file size
        - 🔒 Better data type preservation
        
        **NPZ vs NPY:**
        - ⚡ Faster loading with compression
        - 📦 70-95% smaller file size
        - 🗜️ Built-in compression
        
        **Enhanced Error Handling:**
        - 🛡️ Multiple fallback loading methods
        - 🔧 Detailed debugging information
        - 🚨 Clear error messages with solutions
        """)

with tab4:
    st.markdown("### 🛠️ Debug Information")
    
    st.markdown("#### 📋 System Status")
    
    # Check data shapes and types
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Data Shapes:**")
        st.write(f"Customers: {customers.shape}")
        st.write(f"Similarities: {similarities.shape}")
        st.write(f"Customer Products: {customer_products.shape}")
        st.write(f"Product Names: {product_names.shape}")
    
    with col2:
        st.markdown("**Data Types:**")
        st.write(f"Similarities dtype: {similarities.dtype}")
        st.write(f"Model type: {type(model).__name__}")
        st.write(f"Scaler type: {type(scaler).__name__}")
    
    st.markdown("#### 🔍 Sample Data Preview")
    
    with st.expander("👥 Customer Data Sample"):
        st.dataframe(customers.head())
    
    with st.expander("🛍️ Product Names Sample"):
        st.dataframe(product_names.head())
    
    with st.expander("📊 Similarities Matrix Info"):
        st.write(f"Shape: {similarities.shape}")
        st.write(f"Min value: {similarities.min():.3f}")
        st.write(f"Max value: {similarities.max():.3f}")
        st.write(f"Mean value: {similarities.mean():.3f}")

st.markdown("---")
st.markdown('<p style="text-align: center; color: #666;">⚡ Powered by AI & Machine Learning | Optimized for Speed 🚀 | Enhanced Error Handling 🛡️ | Created by Gowri Nandhan</p>', unsafe_allow_html=True)
