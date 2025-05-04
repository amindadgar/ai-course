import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from math import pi

def cosine_similarity(vec1, vec2):
    """
    Calculate the cosine similarity between two vectors.
    
    Parameters:
    vec1, vec2 (numpy.ndarray): Input vectors
    
    Returns:
    float: Cosine similarity value between -1 and 1
    """
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    # Avoid division by zero
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0
    
    return dot_product / (norm_vec1 * norm_vec2)

def get_vector_from_angle(angle, magnitude):
    """Convert angle (in radians) and magnitude to a 2D vector"""
    x = magnitude * np.cos(angle)
    y = magnitude * np.sin(angle)
    return np.array([x, y])

def plot_vectors(vec1, vec2, similarity):
    fig, ax = plt.subplots(figsize=(4, 4))
    
    # Plot the vectors
    ax.quiver(0, 0, vec1[0], vec1[1], angles='xy', scale_units='xy', scale=1, color='r', label='Vector A', width=0.02)
    ax.quiver(0, 0, vec2[0], vec2[1], angles='xy', scale_units='xy', scale=1, color='b', label='Vector B', width=0.02)
    
    # Set limits and labels
    max_val = max(np.max(np.abs(vec1)), np.max(np.abs(vec2)), 1) * 1.2
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)
    ax.grid(True)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    ax.set_title(f"2D Vectors Visualization\nCosine Similarity: {similarity:.4f}", fontsize=12)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    
    return fig

# Streamlit app
st.set_page_config(page_title="Vector Cosine Similarity", layout="wide")
st.title("Interactive Cosine Similarity Calculator")

st.write("""
Adjust the angles and magnitudes of the vectors to see how cosine similarity changes.
The cosine similarity ranges from -1 (opposite directions) to 1 (same direction).
""")

col1, col2 = st.columns(2)

# Vector A controls
with col1:
    st.subheader("Vector A")
    angle_a = st.slider("Angle (degrees)", 0, 360, 30, key="angle_a")
    magnitude_a = st.slider("Magnitude", 0.1, 5.0, 1.0, key="mag_a")

# Vector B controls
with col2:
    st.subheader("Vector B")
    angle_b = st.slider("Angle (degrees)", 0, 360, 60, key="angle_b")
    magnitude_b = st.slider("Magnitude", 0.1, 5.0, 1.0, key="mag_b")

# Convert angles to radians
angle_a_rad = angle_a * (pi / 180)
angle_b_rad = angle_b * (pi / 180)

# Calculate vectors
vector_a = get_vector_from_angle(angle_a_rad, magnitude_a)
vector_b = get_vector_from_angle(angle_b_rad, magnitude_b)

# Calculate similarity
similarity = cosine_similarity(vector_a, vector_b)

# Display vector values
col1, col2 = st.columns(2)
with col1:
    st.write(f"Vector A: [{vector_a[0]:.2f}, {vector_a[1]:.2f}]")
with col2:
    st.write(f"Vector B: [{vector_b[0]:.2f}, {vector_b[1]:.2f}]")

# Display similarity with a progress bar
st.subheader("Cosine Similarity")
# Map from [-1,1] to [0,1] for the progress bar
progress_value = (similarity + 1) / 2
st.progress(progress_value)
st.write(f"Similarity Value: {similarity:.4f}")

# Color coding based on similarity
if similarity > 0.7:
    st.success("Vectors are pointing in similar directions")
elif similarity < -0.7:
    st.error("Vectors are pointing in opposite directions")
elif -0.3 <= similarity <= 0.3:
    st.warning("Vectors are close to perpendicular")
else:
    st.info("Vectors are at an intermediate angle")

# Plot
fig = plot_vectors(vector_a, vector_b, similarity)
# Make the plot display in a smaller container
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.pyplot(fig, use_container_width=True) 