"""
Configuration file for Latent Semantic Analysis (LSA) project.

This file contains all configurable constants used throughout the analysis.
Modify these values to adjust the behavior of the LSA pipeline.
"""

# ============================================================================
# DOCUMENT PROCESSING CONFIGURATION
# ============================================================================

# Number of documents to process for each category
NUMBER_OF_SCIENCE_FILES = 20
NUMBER_OF_SOCIAL_SCIENCE_FILES = 20

# Number of words to skip from the beginning of each document file
SKIP_FIRST_N_WORDS = 4

# ============================================================================
# LATENT SEMANTIC ANALYSIS (LSA) CONFIGURATION
# ============================================================================

# Number of top words to extract for similarity analysis
TOP_N_WORDS = 5

# Number of top words per topic for word cloud generation
TOP_WORDS_PER_TOPIC = 30

# ============================================================================
# SVD (Singular Value Decomposition) CONFIGURATION
# ============================================================================

# Algorithm for SVD computation ('randomized' is faster for large matrices)
SVD_ALGORITHM = 'randomized'

# Number of iterations for randomized SVD algorithm
SVD_N_ITER = 100

# Random state for reproducibility
SVD_RANDOM_STATE = 122

# ============================================================================
# GRAPH VISUALIZATION CONFIGURATION
# ============================================================================

# Scaling factor for edge weights in semantic graphs (thicker edges = higher similarity)
EDGE_WEIGHT_SCALE = 5

# Node size for graph visualization
NODE_SIZE = 1500

# Node color for graph visualization
NODE_COLOR = 'lightblue'

# Edge label font color
EDGE_LABEL_COLOR = 'red'

# Graph figure size (width, height in inches)
GRAPH_FIGURE_SIZE = (10, 8)

# ============================================================================
# WORD CLOUD CONFIGURATION
# ============================================================================

# Background color for word clouds
WORDCLOUD_BACKGROUND_COLOR = 'white'

# Word cloud dimensions (width, height in pixels) - used when no mask is provided
WORDCLOUD_WIDTH = 800
WORDCLOUD_HEIGHT = 400

# Path to custom mask image for word clouds (optional)
WORDCLOUD_MASK_PATH = 'cloud.png'

# Word cloud contour settings (when using mask)
WORDCLOUD_CONTOUR_WIDTH = 3
WORDCLOUD_CONTOUR_COLOR = 'black'

# ============================================================================
# FILE PATHS
# ============================================================================

# Prefix for science document files (e.g., 'science-1.txt', 'science-2.txt', ...)
SCIENCE_FILE_PREFIX = 'science'

# Prefix for social science document files (e.g., 'social-science-1.txt', ...)
SOCIAL_SCIENCE_FILE_PREFIX = 'social-science'

# Output file prefixes
SCIENCE_WORDCLOUD_PREFIX = 'science'
SOCIAL_SCIENCE_WORDCLOUD_PREFIX = 'social-science'

